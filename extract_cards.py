# Copyright (C) 2026 IgnorantGrimyYak
"""
Detect Magic: The Gathering cards in a photo, correct perspective/rotation,
and export each card as an upright 5:7 ratio image.

Usage:
    python extract_cards.py [input] [output_dir]

`input` may be a folder of images, a full/relative path to a single image,
or just a filename located inside the default input folder. Defaults to
the input/ and output/ folders relative to this script.

Phase 4 (INTEGRATION_PLAN.md): same-artwork set disambiguation. When the
pHash-best card is reprinted in several sets sharing the same
illustration_id, the winning set is chosen with the set-symbol scoring layer
(`set_disambiguation`), falling back to the legacy pHash-best entry whenever
disambiguation is disabled or fails (plan assumption A7).
"""
import argparse
import sys
import os
import math
import re
import sqlite3
import csv
import pickle
import cv2
import numpy as np
import imagehash
from PIL import Image

import set_disambiguation as sdis

CARD_RATIO = 5.0 / 7.0  # width / height for a card in upright (portrait) orientation
OUTPUT_LONG_SIDE = 1400  # px, output card height (portrait)
SHOW_PRICE_TAG = True  # toggle to show/hide the USD price line under the card name label


def order_quad_points(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _quad_axes(quad):
    """(u, v, L, S, center) of a rectangular (4,2) quad: u/v unit vectors
    along the long/short axis, L/S the corresponding lengths."""
    e0 = quad[1] - quad[0]
    e1 = quad[3] - quad[0]
    n0, n1 = np.linalg.norm(e0), np.linalg.norm(e1)
    if n0 >= n1:
        return e0 / n0, e1 / n1, n0, n1, quad.mean(axis=0)
    return e1 / n1, e0 / n0, n1, n0, quad.mean(axis=0)


def _split_cells(quad, n):
    """Split a rectangular quad into n equal cells along its long axis.
    Returns n (4,2) float32 quads that keep the parent's corner winding
    (the same edge traversal order, with the long edge scaled): the
    perspective warp maps corner i to destination corner i, so a reversed
    winding would produce a mirrored card image and break matching."""
    e0 = quad[1] - quad[0]
    e1 = quad[3] - quad[0]
    if np.linalg.norm(e0) >= np.linalg.norm(e1):
        e0s, e1s = e0 / n, e1
        long_e = e0
    else:
        e0s, e1s = e0, e1 / n
        long_e = e1
    cells = []
    for i in range(n):
        a = quad[0] + long_e * (i / n)
        cells.append(np.array([a, a + e0s, a + e0s + e1s, a + e1s],
                              dtype=np.float32))
    return cells


def _contour_card_filter(area, rect, min_fill=0.6):
    """Geometric card-likeness of (area, minAreaRect): 5:7 aspect (either
    orientation) and solid fill (rejects thin fused-gap outlines)."""
    rw, rh = rect[1]
    if rw <= 0 or rh <= 0:
        return False
    if area / (rw * rh) < min_fill:
        return False
    return abs(min(rw, rh) / max(rw, rh) - CARD_RATIO) <= 0.18


def _crop_contour_candidates(crop, blur_ksize=5):
    """Every closed contour in `crop` passing the geometric card filters,
    as (area, rect, boxPoints). No dedup, no image-relative area bound -
    the caller applies its own size window. Reuses the detector's exact
    edge pipeline (gray+sat Canny, 5x5 close)."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat = cv2.GaussianBlur(hsv[:, :, 1], (blur_ksize, blur_ksize), 0)
    edges = cv2.bitwise_or(cv2.Canny(blurred, 40, 120),
                           cv2.Canny(sat, 40, 120))
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        rect = cv2.minAreaRect(cnt)
        if _contour_card_filter(area, rect):
            out.append((area, rect, cv2.boxPoints(rect)))
    return out


def _split_subcard_boxes(selected, all_contours):
    """Replace accepted boxes that actually contain smaller cards.

    A box that passes the top-level filters can still be a container: e.g.
    two small cards stacked on a playing mat, where the mat/fused outline
    is what passes and the individual cards are too small to be candidates
    (or their outer borders are invisible against the mat). The tell is
    closed card-like contours strictly INSIDE the box - each small card's
    crisp artwork frame closes even when the card's own border does not.
    A split into two cells (along the long axis) happens when there are
    two or more COMPARABLE-size sub-frames (the cards) whose centers form
    >= 2 clusters along the long axis spanning most of the box.
    A single card never matches this: the regions that close inside it
    (artwork box ~30-40% of the card, text box, power box) fail the
    aspect test, or one dominates the others in size (comparability
    test). Runs on the full-image contour list - deliberately crop-free:
    any crop of a single card's half closes fake 'card' loops along its
    boundary from that card's own far frame edges, so cropped evidence is
    not trustworthy.
    """
    out = []
    for box in selected:
        u, v, L, S, c = _quad_axes(box)
        cf = np.array(c, dtype=np.float64)
        uf = np.array(u, dtype=np.float64)
        subs = []
        for area, rect, _bp in all_contours:
            if not (0.05 * L * S <= area <= 0.5 * L * S):
                continue
            if not _contour_card_filter(area, rect):
                continue
            corners = cv2.boxPoints(rect)
            if not all(cv2.pointPolygonTest(box, (float(x), float(y)), False) >= 0
                       for x, y in corners):
                continue
            ctr = np.array(rect[0], dtype=np.float64)
            subs.append((area, float(np.dot(ctr - cf, uf))))
        split = False
        if len(subs) >= 2:
            areas = sorted(a for a, _p in subs)
            if areas[-2] >= 0.5 * areas[-1]:  # comparable sizes
                projs = sorted(p for _a, p in subs)
                groups = 1
                for i in range(1, len(projs)):
                    if projs[i] - projs[i - 1] > 0.5 * S:
                        groups += 1
                if groups >= 2 and (projs[-1] - projs[0]) > 0.45 * L:
                    split = True
        if split:
            out.extend(_split_cells(box, 2))
        else:
            out.append(box)
    return out


def _rescue_fusion_boxes(image, selected, rejected_fusions, max_cards, blur_ksize=5):
    """Recover cards hidden in REJECTED fused multi-card contours.

    The morphological close can bridge the gutter between aligned cards
    (a paper fold line, a mat edge, a narrow gap): one combined contour
    instead of one per card. The combined shape then usually fails the 5:7
    aspect or fill tests and is dropped, losing every card it covered.
    Rescue: split the rejected rect into card-sized cells along its long
    side and keep a cell only if a depth-1 mini-run of this detector on a
    crop around the cell finds a closed card-like contour of plausible
    size inside it (the cell's own card frame closes on its own in the
    crop; dead-space cells - fold lines, empty mat - yield nothing).
    """
    selected = list(selected)
    med_short = float(np.median([_quad_axes(b)[3] for b in selected]))
    med_area = float(np.median([_quad_axes(b)[2] * _quad_axes(b)[3] for b in selected]))
    img_area = image.shape[0] * image.shape[1]

    def is_dup(b):
        cc = b.mean(axis=0)
        for ob in selected:
            if cv2.pointPolygonTest(ob, (float(cc[0]), float(cc[1])), False) >= 0:
                return True
            oc = ob.mean(axis=0)
            if math.hypot(cc[0] - oc[0], cc[1] - oc[1]) < 50:
                return True
        return False

    for area, rect, fill in sorted(rejected_fusions, key=lambda r: -r[0])[:8]:
        if len(selected) >= max_cards:
            break
        if area > 0.45 * img_area:  # too big to be a card fusion
            continue
        rw, rh = rect[1]
        short, long_ = min(rw, rh), max(rw, rh)
        if not (0.4 * med_short <= short <= 1.6 * med_short):
            continue
        est_card_long = short / CARD_RATIO
        ns = set()
        n = int(round(long_ / est_card_long))
        if 2 <= n <= 4:
            ns.add(n)
        if fill < 0.6:
            ns.add(2)
        for n in sorted(ns):
            added = False
            for cell in _split_cells(cv2.boxPoints(rect), n):
                for m in (0.1, 0.2, 0.5):
                    h, w = image.shape[:2]
                    x0, y0 = cell[:, 0].min(), cell[:, 1].min()
                    x1, y1 = cell[:, 0].max(), cell[:, 1].max()
                    cw, ch = x1 - x0, y1 - y0
                    cx0 = max(0, int(x0 - m * cw)); cy0 = max(0, int(y0 - m * ch))
                    cx1 = min(w, int(x1 + m * cw)); cy1 = min(h, int(y1 + m * ch))
                    if cx1 - cx0 < 40 or cy1 - cy0 < 40:
                        continue
                    crop = image[cy0:cy1, cx0:cx1]
                    cands = [cnd for cnd in _crop_contour_candidates(crop, blur_ksize)
                             if 0.25 * med_area <= cnd[0]
                             <= 0.85 * (cx1 - cx0) * (cy1 - cy0)]
                    # closest to the median accepted-card area first: a
                    # real card frame, not a crop-boundary closure fake
                    for _a, _r, bp in sorted(cands, key=lambda t: abs(t[0] - med_area)):
                        full = bp + np.array([cx0, cy0], dtype=np.float32)
                        if is_dup(full):
                            continue
                        selected.append(full)
                        added = True
                        break
                    if added:
                        break
                if added:
                    break
            if added:  # this contour is explained; do not re-split it with another n
                break
    return selected


def _find_card_quads_impl(image, blur_ksize=5, max_cards=50, min_fill=0.6):
    """Find quadrilateral contours likely to be cards using a given blur
    kernel size. `min_fill` rejects contours whose area is a thin fraction
    of their minAreaRect (fused gap outlines between adjacent cards); real
    cards fill most of their bounding rectangle.
    Returns (selected, all_contours, rejected_fusions): the accepted
    card boxes plus the two evidence lists the recovery passes in
    `find_card_quads` consume (see _split_subcard_boxes /
    _rescue_fusion_boxes) - they run AFTER grid repair + hole synthesis
    there, so a grid-snapped quad always wins over a raw rescued box at
    the same position."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # Grayscale Canny alone can miss the card's outer edge when a black
    # card border sits on a similarly dark/textured background (e.g. a
    # wooden table): the brightness contrast is too low there, even though
    # the color is very different. Add edges from the HSV saturation
    # channel too: a black/white card border has near-zero saturation while
    # wood, cloth, etc. backgrounds usually have much higher saturation, so
    # combining both makes the outer boundary far more likely to be a fully
    # closed contour.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = cv2.GaussianBlur(hsv[:, :, 1], (blur_ksize, blur_ksize), 0)

    edges_gray = cv2.Canny(blurred, 40, 120)
    edges_sat = cv2.Canny(sat, 40, 120)
    edges = cv2.bitwise_or(edges_gray, edges_sat)

    # Morphological closing to bridge small gaps so outer borders form a
    # single continuous contour instead of leaking into the background.
    # Keep this mild: too aggressive and touching/adjacent cards can fuse
    # into a single blob.
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img_area = image.shape[0] * image.shape[1]
    candidates = []
    rejected_fusions = []   # (area, rect, fill) - dropped by the fill/aspect tests
    all_contours = []       # (area, rect, boxPoints) - card-like closed contours

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= 500:
            rect0 = cv2.minAreaRect(cnt)
            if _contour_card_filter(area, rect0):
                all_contours.append((area, rect0, cv2.boxPoints(rect0)))
        # A card should be a reasonably large chunk of the image, but not
        # literally the entire photo (background usually shows around the
        # edges). A single card photographed alone can still fill most of
        # the frame, so allow up to 95% of the image area.
        if area < img_area * 0.01 or area > img_area * 0.95:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) != 4:
            # try the minAreaRect as a fallback quad
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            approx = box.reshape(-1, 1, 2).astype(np.float32)

        quad = approx.reshape(4, 2).astype(np.float32)

        # Check aspect ratio roughly matches a card (either orientation)
        rect = cv2.minAreaRect(cnt)
        (rw, rh) = rect[1]
        if rw == 0 or rh == 0:
            continue
        # Fused gap outlines (two adjacent cards' outer edges merged by the
        # morphological close) trace a thin strip between the cards: their
        # area is a small fraction of the bounding rectangle, while a real
        # card fills ~0.9+. Rejecting fill < min_fill drops these fused
        # duplicates without touching genuine cards.
        if area / (rw * rh) < min_fill:
            rejected_fusions.append((area, rect, area / (rw * rh)))
            continue
        long_side = max(rw, rh)
        short_side = min(rw, rh)
        ratio = short_side / long_side
        expected_ratio = CARD_RATIO  # ~0.714
        if abs(ratio - expected_ratio) > 0.18:
            rejected_fusions.append((area, rect, area / (rw * rh)))
            continue

        candidates.append((area, quad, cv2.boxPoints(rect)))

    # Sort by area descending so outer card boundaries (larger) are considered
    # before inner regions like the artwork box (smaller, nested inside).
    candidates.sort(key=lambda c: -c[0])

    selected = []
    for area, quad, box in candidates:
        cx, cy = box.mean(axis=0)
        is_dup = False
        for sbox in selected:
            # If this box's center falls inside an already-selected box, it's
            # a nested/duplicate detection (e.g. the card artwork inside the
            # full card), not a separate card.
            if cv2.pointPolygonTest(sbox.astype(np.float32), (float(cx), float(cy)), False) >= 0:
                is_dup = True
                break
            scx, scy = sbox.mean(axis=0)
            dist = math.hypot(cx - scx, cy - scy)
            if dist < 50:
                is_dup = True
                break
        if is_dup:
            continue
        selected.append(box)
        if len(selected) >= max_cards:
            break

    return selected, all_contours, rejected_fusions


def _row_clusters(boxes):
    """Cluster box centers into grid rows (>= 3 boxes; y tolerance is 0.35x
    the median long side). Returns (centers, med_long, rows) where rows is a
    list of index lists, each sorted by x, rows sorted top to bottom."""
    centers = np.array([b.mean(axis=0) for b in boxes], dtype=np.float64)
    longs = []
    for b in boxes:
        e0 = np.linalg.norm(b[1] - b[0])
        e1 = np.linalg.norm(b[3] - b[0])
        longs.append(max(e0, e1))
    med_long = float(np.median(longs))
    row_tol = 0.35 * med_long
    order = np.argsort(centers[:, 1])
    rows = []
    for i in order:
        for r in rows:
            if abs(centers[i, 1] - centers[r, 1].mean()) <= row_tol:
                r.append(i)
                break
        else:
            rows.append([i])
    rows = [sorted(r, key=lambda j: centers[j, 0]) for r in rows if len(r) >= 3]
    rows.sort(key=lambda r: centers[r[0], 1])
    return centers, med_long, rows


def _rebuild_quad(b, cy_new, long_new, short_new=None, cx_new=None):
    """Rebuild a box quad keeping its orientation; optionally replace the
    center (x and/or y) and the short and/or long side (grid repair of
    corrupted contours)."""
    tl, tr, br, bl = (np.asarray(p, dtype=np.float64) for p in b)
    u = tr - tl
    u = u / np.linalg.norm(u)
    vr = bl - tl
    perp = np.array([-u[1], u[0]])
    v = perp if np.dot(vr, perp) > 0 else -perp
    c = b.mean(axis=0)
    w = float(np.linalg.norm(tr - tl))
    if cy_new is not None:
        c = np.array([c[0], cy_new])
    if cx_new is not None:
        c = np.array([cx_new, c[1]])
    l = long_new if long_new is not None else float(np.linalg.norm(bl - tl))
    if short_new is not None:
        w = float(short_new)
    hw, hh = w / 2.0, l / 2.0
    return np.array([
        c - u * hw - v * hh, c + u * hw - v * hh,
        c + u * hw + v * hh, c - u * hw + v * hh,
    ], dtype=np.float32)


def correct_boxes(boxes):
    """Grid-consistency repair for corrupted contours (card outline merged
    with a shadow/fold edge in a gap): minAreaRect is shifted ~50-80px off
    the card AND inflated ~8-16% in the long direction. Both together put a
    hard floor of 72-88 on the pHash distance even though the card is
    clearly visible.

    A box is flagged only when BOTH signals fire (each alone is too weak):
      - center: >5% of the median card height away from the local row line
        (interpolated from its row neighbors - robust to sheet waviness;
        edge boxes skipped, one-sided extrapolation is unreliable).
        False positives: perspective/curvature can legitimately displace a
        single card ~30-80px from its neighbors' line (matching confirms
        those cards are fine).
      - size: long side deviates >5% from the row's robust median (median
        over the non-deviant boxes). Perspective alone gives up to ~10%
        within a row, but only with a small center residual; contamination
        couples a large size excess with the center shift.
    Flagged boxes are rebuilt: center-y snapped to the local line and each
    edge snapped to the median of the dimension it actually spans -
    horizontal edge -> row short-side median, vertical edge -> row
    long-side median (orientation detected from the quad's edge
    directions, because minAreaRect corner order flips when the rect
    angle crosses 45 degrees, so b0->b1 can be the vertical long edge of
    a portrait card); an edge already within 5% of its median is left
    at its own length. The center-x is then refined against the
    same-rank boxes of the sibling rows
    (the column position), accepted only when it is beyond the noise
    band yet still within half a card width, so a rank misalignment can
    never drag the box to a wrong column. Orientation is kept.
    Non-flagged boxes are never modified.

    Works on 2+-row grids: 2-row sheets get the same row-local repair
    (the flag rule - both signals must fire - is unchanged, so a 2-row
    layout cannot false-positive on perspective alone).
    """
    if not boxes:
        return boxes
    centers, med_long, rows = _row_clusters(boxes)
    if len(rows) < 2:
        return boxes
    th = 0.05 * med_long
    out = list(boxes)
    fixed = []
    for row in rows:
        n = len(row)
        row = sorted(row, key=lambda j: centers[j, 0])
        longs = np.array([max(np.linalg.norm(boxes[j][1] - boxes[j][0]),
                              np.linalg.norm(boxes[j][3] - boxes[j][0])) for j in row])
        rowmed = float(np.median(longs))
        size_dev = np.abs(longs - rowmed) > 0.05 * rowmed
        if size_dev.any():
            good = longs[~size_dev]
            if len(good) >= 3:
                rowmed = float(np.median(good))
                size_dev = np.abs(longs - rowmed) > 0.05 * rowmed
        shorts = np.array([min(np.linalg.norm(boxes[j][1] - boxes[j][0]),
                               np.linalg.norm(boxes[j][3] - boxes[j][0]))
                           for j in row])
        shortmed = float(np.median(shorts))
        short_dev = np.abs(shorts - shortmed) > 0.05 * shortmed
        if short_dev.any():
            good = shorts[~short_dev]
            if len(good) >= 3:
                shortmed = float(np.median(good))
                short_dev = np.abs(shorts - shortmed) > 0.05 * shortmed
        for t in range(n):
            j = row[t]
            xj, yj = centers[j]
            yexp = None
            if 0 < t < n - 1:
                x1, y1 = centers[row[t - 1]]
                x2, y2 = centers[row[t + 1]]
                yexp = y1 + (y2 - y1) * (xj - x1) / (x2 - x1)
                if t >= 2 and t <= n - 3:
                    x3, y3 = centers[row[t - 2]]
                    x4, y4 = centers[row[t + 2]]
                    yexp2 = y3 + (y4 - y3) * (xj - x3) / (x4 - x3)
                    yexp = (yexp + yexp2) / 2
            center_dev = yexp is not None and abs(yexp - yj) > th
            if not (center_dev and size_dev[t]):
                continue
            cy_new = yexp
            # minAreaRect corner order flips when the rect angle crosses 45
            # degrees: b0->b1 can be the vertical (long) edge of a portrait
            # card. _rebuild_quad assigns short_new to the b0->b1 (u) edge
            # and long_new to the b0->b3 (v) edge, so pick each target by
            # the dimension that edge actually spans - otherwise the
            # inflated vertical edge of a flipped-order contour stays
            # inflated AND the normal width gets snapped to the long-side
            # median (red_05/red_06 row-3/card-5: vertical +14/+17%,
            # width normal).
            u = np.asarray(boxes[j][1], dtype=np.float64) \
                - np.asarray(boxes[j][0], dtype=np.float64)
            u_horiz = abs(u[0]) >= abs(u[1])
            u_target = shortmed if u_horiz else rowmed
            v_target = rowmed if u_horiz else shortmed
            u_cur = float(np.linalg.norm(u))
            short_new = float(u_target) \
                if abs(u_cur - u_target) > 0.05 * u_target else None
            long_new = float(v_target)
            # Refine the center-x against the sibling rows' same-rank box
            # (the column position). Accept a consensus only when it is
            # off by more than the noise band yet still within half a
            # card width: beyond that the rank is likely misaligned (the
            # row lost a box elsewhere) and the box's own x is kept.
            cx_new = None
            analogs = []
            row_set = set(row)  # `row` was re-bound by sorted() above, so
                                # identity against `rows` no longer works
            for r in rows:
                if set(r) == row_set or len(r) <= t:
                    continue
                jr = sorted(r, key=lambda jj: centers[jj, 0])[t]
                analogs.append(float(centers[jr, 0]))
            if analogs:
                xc = float(np.median(analogs))
                if th < abs(xc - xj) < 0.5 * med_long:
                    cx_new = xc
            out[j] = _rebuild_quad(boxes[j], cy_new, long_new, short_new, cx_new)
            fixed.append((j,
                          round(float(cy_new - yj), 1) if cy_new is not None else 0.0,
                          round(float(long_new - longs[t]), 1) if long_new is not None else 0.0,
                          round(float(short_new - shorts[t]), 1) if short_new is not None else 0.0,
                          round(float(cx_new - xj), 1) if cx_new is not None else 0.0))
    if fixed:
        print("   corrected (idx, dy, dlong, dshort, dx):", fixed)
    return out


def _complete_grid(boxes):
    """Synthesize the quad of a missing card in a grid photo. A doubled
    horizontal gap between two row neighbors (1.6-2.8x the median of the
    row's other gaps) marks a hole only when at least one other row keeps a
    card near the gap midpoint (a structural gutter is wide in every row
    and has no box anywhere near the midpoint). The synthesized x starts at
    the gap midpoint, refined against the column consensus of sibling rows
    (a hole spanning a structural gutter has a midpoint that lands in the
    gutter; the column line of the other rows keeps it on the card).
    A hole at the END of a row (no doubled gap inside the row) is also
    filled: when every sibling row that has one more box keeps exactly
    one box beyond half a card width from all of this row's boxes, and
    all such orphans lie past the same end of the row, the row lost its
    end card and it is synthesized on the row line at the sibling
    column's x. Returns a list of (quad, center) tuples; [] when the
    image is not a 2+-row grid."""
    if len(boxes) < 6:
        return []
    centers, med_long, rows = _row_clusters(boxes)
    if len(rows) < 2:
        return []
    synth = []

    def half(b):
        e0 = np.linalg.norm(b[1] - b[0])
        e1 = np.linalg.norm(b[3] - b[0])
        return min(e0, e1) / 2, max(e0, e1) / 2
    for row in rows:
        xs = centers[row, 0]
        sx = np.sort(xs)
        gaps = np.diff(sx)
        idx = np.argsort(xs)
        A = np.vstack([xs[idx], np.ones(len(xs))]).T
        coef = np.linalg.lstsq(A, centers[row, 1][idx], rcond=None)[0]
        for k in range(len(row) - 1):
            g = gaps[k]
            # A missing card doubles the local pitch. Compare against the
            # median of the OTHER gaps: normal gaps vary ~30% between rows
            # (perspective) so a global min/max would false-positive.
            others = np.delete(gaps, k)
            pitch = float(np.median(others))
            if pitch <= 0:
                continue
            if not (1.6 * pitch <= g <= 2.8 * pitch):
                continue
            xm = (sx[k] + sx[k + 1]) / 2
            # Verify against other rows: a structural gutter (wide spacing
            # present in every row) has no box near this midpoint anywhere,
            # whereas a true hole does (sibling rows keep that column).
            verified = 0
            for r in rows:
                if r is row:
                    continue
                for j in r:
                    cxj, cyj = centers[j]
                    if abs(cyj - centers[row[0], 1]) > 0.5 * med_long and abs(cxj - xm) < 0.5 * pitch:
                        verified += 1
                        break
            if verified < 1:
                continue
            # The midpoint can land in a structural gutter: a hole spans a
            # normal gap plus the wide gutter, so the midpoint drifts
            # toward the gutter. Refine xm with the column consensus: every
            # sibling row that keeps one more card (i.e. has no hole) has a
            # box at this same column rank, and those boxes trace the column
            # line. Fit x(y) through them (perspective drifts the column
            # across the photo) and evaluate at the row line. Fall back to
            # the midpoint when there is no analog or the consensus would
            # overlap a flanking card.
            analogs = []
            for r in rows:
                if r is row or len(r) != len(row) + 1:
                    continue
                # The hole sits BETWEEN ranks k and k+1 of this row, so the
                # missing card is at rank k+1 of a full (sibling) row.
                jr = sorted(r, key=lambda j: centers[j, 0])[k + 1]
                analogs.append((float(centers[jr, 1]), float(centers[jr, 0])))
            if analogs:
                ay = np.array([p[0] for p in analogs])
                ax = np.array([p[1] for p in analogs])
                if len(analogs) == 1:
                    xc = float(ax[0])
                else:
                    acoef = np.linalg.lstsq(
                        np.vstack([ay, np.ones(len(ay))]).T, ax, rcond=None)[0]
                    xc = float(np.polyval(acoef, float(np.polyval(coef, xm))))
                if sx[k] < xc < sx[k + 1]:
                    xm = xc
            ym = float(np.polyval(coef, xm))
            b1, b2 = boxes[row[k]], boxes[row[k + 1]]
            hw = (half(b1)[0] + half(b2)[0]) / 2
            hh = (half(b1)[1] + half(b2)[1]) / 2
            # Row direction from flanking centers (dx > 0 => no 0/pi
            # line-aliasing, unlike median of per-box edge angles).
            p1, p2 = centers[row[k]], centers[row[k + 1]]
            ang = float(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
            ca, sa = np.cos(ang), np.sin(ang)
            ux, uy = ca, sa
            vx, vy = -sa, ca
            q = np.array([
                (xm - ux * hw - vx * hh, ym - uy * hw - vy * hh),
                (xm + ux * hw - vx * hh, ym + uy * hw - vy * hh),
                (xm + ux * hw + vx * hh, ym + uy * hw + vy * hh),
                (xm - ux * hw + vx * hh, ym - uy * hw + vy * hh),
            ], dtype=np.float32)
            synth.append((q, (xm, ym)))
        # End hole: no doubled gap inside the row, but a sibling row with
        # exactly one more box keeps a column this row does not. When
        # EACH such sibling has exactly one box beyond half a card width
        # from every box of this row and all orphans lie past the same
        # end of the row, the row lost its end card. Synthesize it on the
        # row line at the sibling column's x. (A structural gutter can
        # never produce such an orphan: its extra width is present in
        # every row, so no row keeps a lone box out there.)
        orphans = []
        for r in rows:
            if r is row or len(r) != len(row) + 1:
                continue
            rorph = [j for j in r
                     if float(np.min(np.abs(centers[row, 0] - centers[j, 0])))
                     > 0.5 * med_long]
            if len(rorph) != 1:
                orphans = None
                break
            orphans.append(rorph[0])
        if orphans:
            ox = np.array([float(centers[j, 0]) for j in orphans])
            if all(o > sx[-1] + 0.3 * med_long for o in ox):
                side = 1
            elif all(o < sx[0] - 0.3 * med_long for o in ox):
                side = -1
            else:
                side = 0
            if side:
                xc = float(np.median(ox))
                keep = (xc - sx[-1]) if side == 1 else (sx[0] - xc)
                if keep > 0.75 * med_long:
                    ym = float(np.polyval(coef, xc))
                    flank = row[-1] if side == 1 else row[0]
                    jn = orphans[int(np.argmin(np.abs(ox - xc)))]
                    hw = (half(boxes[flank])[0] + half(boxes[jn])[0]) / 2
                    hh = (half(boxes[flank])[1] + half(boxes[jn])[1]) / 2
                    ang = float(np.arctan2(coef[0], 1.0))
                    ca, sa = np.cos(ang), np.sin(ang)
                    ux, uy = ca, sa
                    vx, vy = -sa, ca
                    q = np.array([
                        (xc - ux * hw - vx * hh, ym - uy * hw - vy * hh),
                        (xc + ux * hw - vx * hh, ym + uy * hw - vy * hh),
                        (xc + ux * hw + vx * hh, ym + uy * hw + vy * hh),
                        (xc - ux * hw + vx * hh, ym - uy * hw + vy * hh),
                    ], dtype=np.float32)
                    synth.append((q, (xc, ym)))
    return synth


def _synth_has_card_content(image, quads, ref_boxes):
    """Content check for synthesized grid-hole quads: the grid geometry
    alone cannot tell "a card the detector missed" from "an empty slot" -
    both sit in a plausible grid position. What separates them is image
    content: a card's printed matter (art, frame, text) leaves a dense
    Canny edge signature, while an empty slot is bare table with almost no
    edges. Each quad's central 50% is scored with the same edge pipeline
    as the detector (gray + saturation Canny, 5x5 close) and kept only
    when its density reaches max(0.05, 0.25x the median central density of
    the real detected boxes): the absolute floor keeps a faint sheet from
    dragging the relative threshold into the noise, and 0.25x the real
    median stays far below the weakest real card (measured on the
    validated sheets: real cards 0.15-0.63, an empty slot 0.00). Returns
    one bool per quad, True = card content present, keep the quad."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]
    edges = cv2.bitwise_or(
        cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120),
        cv2.Canny(cv2.GaussianBlur(sat, (5, 5), 0), 40, 120))
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.erode(cv2.dilate(edges, kernel, iterations=1), kernel, iterations=1)
    h, w = edges.shape

    def density(q):
        c = q.mean(axis=0)
        inner = (c + 0.5 * (q - c)).astype(np.int32)
        x0, y0 = inner.min(axis=0)
        x1, y1 = inner.max(axis=0)
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(w, int(x1) + 1), min(h, int(y1) + 1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        e = edges[y0:y1, x0:x1]
        m = np.zeros(e.shape, np.uint8)
        cv2.fillPoly(m, [inner - np.array([x0, y0])], 1)
        return float(np.count_nonzero(e[m > 0])) / max(1, int(m.sum()))

    refs = [density(b) for b in ref_boxes]
    med = float(np.median(refs)) if refs else 0.0
    thresh = max(0.05, 0.25 * med)
    return [density(q) >= thresh for q in quads]


def find_card_quads(image, max_cards=50):
    """
    Find quadrilateral contours likely to be cards. Uses a small blur kernel
    by default so thin gaps between tightly-packed cards in grid photos stay
    distinguishable. Two cases warrant a retry with a heavier blur that
    smooths noise and faint edges away:

    - A single card fills most of the frame (a close-up shot) and sensor/JPEG
      noise on high-res photos makes the light blur trace an imprecise,
      oversized border. Retry only if the heavier blur cleanly resolves to
      one card, otherwise keep the original box.
    - No card is found at all. This happens when the card's outer contour
      fuses with faint edges (e.g. a small dark card on a white table whose
      soft shadow streak or scattered noise edges connect the card outline
      to other features): the fused contour's minAreaRect aspect ratio no
      longer looks card-like, so every candidate is rejected. The heavier
      blur erases the faint edges and restores the card's clean closed
      contour. Since the light blur found nothing, any card-like quad the
      heavier blur finds is adopted.
    """
    boxes, all_contours, rejected_fusions = _find_card_quads_impl(
        image, blur_ksize=5, max_cards=max_cards)

    if len(boxes) == 1:
        img_area = image.shape[0] * image.shape[1]
        area_fraction = cv2.contourArea(boxes[0].astype(np.float32)) / img_area
        if area_fraction > 0.35:
            refined, all_contours, rejected_fusions = _find_card_quads_impl(
                image, blur_ksize=9, max_cards=max_cards)
            if len(refined) == 1:
                boxes = refined
    elif not boxes:
        refined, all_contours, rejected_fusions = _find_card_quads_impl(
            image, blur_ksize=9, max_cards=max_cards)
        if refined:
            boxes = refined

    # Grid repair + completion: snap gap-fused corrupted boxes back onto the
    # sheet grid, then synthesize the quad of any card a doubled row gap
    # implies is missing (a row of cards with one hole, interior or at an
    # end). A synthesized quad is kept only when its content actually looks
    # like a card - an empty slot is just bare table (see
    # _synth_has_card_content). No-op for non-grid images (fewer than 2 rows
    # of 3+ boxes).
    boxes = correct_boxes(boxes)
    synth = _complete_grid(boxes)
    if synth:
        keep = _synth_has_card_content(image, [q for q, _c in synth], boxes)
        boxes += [q for (q, _c), ok in zip(synth, keep) if ok]

    # Recovery passes run LAST (after grid repair + synthesis): a box they
    # add at a grid-hole position would pre-empt the higher-quality
    # row/column-snapped synthesized quad, so the grid logic gets first
    # claim. _rescue_fusion_boxes' duplicate check (cell center inside an
    # existing box) then skips any rescued cell a synthesized quad already
    # covers. No-op when there are no containers / rejected fusions.
    boxes = _split_subcard_boxes(boxes, all_contours)
    boxes = _rescue_fusion_boxes(image, boxes, rejected_fusions, max_cards)

    return boxes


def warp_card(image, box):
    """Warp the quadrilateral `box` into an upright 5:7 rectangle."""
    quad = order_quad_points(box)
    tl, tr, br, bl = quad

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    src_w = (width_top + width_bottom) / 2.0
    src_h = (height_left + height_right) / 2.0

    # Determine output size preserving true card ratio (5:7), oriented to
    # match the longer measured side as height (portrait) by default;
    # we'll fix final orientation (portrait, right-side up) afterwards.
    if src_w >= src_h:
        out_w, out_h = OUTPUT_LONG_SIDE, int(round(OUTPUT_LONG_SIDE * CARD_RATIO))
    else:
        out_w, out_h = int(round(OUTPUT_LONG_SIDE * CARD_RATIO)), OUTPUT_LONG_SIDE

    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(image, M, (out_w, out_h))
    return warped


def to_portrait(img):
    """Rotate a landscape image 90deg so result is portrait (5:7, tall)."""
    h, w = img.shape[:2]
    if w > h:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def orient_upright(img):
    """
    Best-effort orientation fix: a MTG card's title bar / text lives in the
    top ~15% of the card. We can't read text reliably without OCR, so we use
    a simple heuristic based on image content brightness/edge density isn't
    reliable either. Instead we just ensure portrait orientation (tall) and
    leave 180-degree flips for manual review, since detecting "up vs down"
    without OCR or template matching is unreliable in general.
    """
    return to_portrait(img)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class CardDatabase:
    """Interface for data/scryfall_db.sqlite perceptual hash lookups."""

    def __init__(self, db_path="data/scryfall_db.sqlite", enable_disambiguation=True,
                 staged_band=True):
        self.db_path = db_path
        self.db_entries = []
        self.ill_index = {}
        self.set_names = {}
        self.templates = None
        self.plst_logo_tpl = None
        self.disambiguation_enabled = False
        self.staged_band = staged_band
        self._ref_img_cache = {}   # scryfall_id -> BGR 488x680 (or None)
        self._disambig_src_mtime = None   # set_disambiguation.py mtime at load
        self._disambig_drift_warned = False
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT scryfall_id, name, phash, price_usd, price_usd_foil, price_usd_etched, scryfall_uri, tcgplayer_purchase_uri, card_type, edition, set_code, illustration_id, phash_back, type_line FROM cards WHERE phash IS NOT NULL AND phash != ''"
                )
                rows = cursor.fetchall()
                for scryfall_id, name, phash_hex, price_usd, price_usd_foil, price_usd_etched, scryfall_uri, tcgplayer_purchase_uri, card_type, edition, set_code, illustration_id, phash_back_hex, type_line in rows:
                    if len(phash_hex) == 64:
                        # 13-tuple row: index 12 is the back-face hash of dual-face
                        # cards (transform/dual_face), None for all other cards.
                        phash_back = int(phash_back_hex, 16) if phash_back_hex and len(phash_back_hex) == 64 else None
                        self.db_entries.append((scryfall_id, name, int(phash_hex, 16), price_usd, price_usd_foil, price_usd_etched, scryfall_uri, tcgplayer_purchase_uri, card_type, edition, set_code, illustration_id, phash_back, type_line))
                # set_code -> set_name (for the same-artwork candidate chips)
                for code, sname in conn.execute(
                    "SELECT DISTINCT set_code, set_name FROM cards "
                    "WHERE set_code IS NOT NULL AND set_name IS NOT NULL"
                ):
                    if code not in self.set_names or not self.set_names[code]:
                        self.set_names[code] = sname
                conn.close()
                print(f"Loaded {len(self.db_entries)} card phashes from {os.path.basename(db_path)}.")
                # illustration_id -> [entry index, ...] (same-artwork groups)
                for idx, entry in enumerate(self.db_entries):
                    ill = entry[11]
                    if ill:
                        self.ill_index.setdefault(ill, []).append(idx)
                # Vectorized pHash matching cache: all front/back 256-bit
                # hashes as (N, 4) uint64 planes (index-aligned with
                # db_entries) plus a 16-bit popcount table, so a query's
                # Hamming distance to the whole DB is a handful of numpy ops
                # instead of a Python loop over every entry.
                n_db = len(self.db_entries)
                self._ph_front_u64 = np.zeros((n_db, 4), dtype=np.uint64)
                self._ph_back_u64 = np.zeros((n_db, 4), dtype=np.uint64)
                self._ph_has_back = np.zeros(n_db, dtype=bool)
                for i, entry in enumerate(self.db_entries):
                    v = entry[2]
                    for j, sh in enumerate((0, 64, 128, 192)):
                        self._ph_front_u64[i, j] = (v >> sh) & 0xFFFFFFFFFFFFFFFF
                    pb = entry[12]
                    if pb is not None:
                        self._ph_has_back[i] = True
                        for j, sh in enumerate((0, 64, 128, 192)):
                            self._ph_back_u64[i, j] = (pb >> sh) & 0xFFFFFFFFFFFFFFFF
                self._ph_pop16 = np.array(
                    [bin(i).count("1") for i in range(65536)], dtype=np.uint8)
                self._load_disambiguation_assets(enable_disambiguation)
            except Exception as e:
                print(f"Error loading card database {db_path}: {e}")
        else:
            print(f"Card database not found at {db_path}.")

    def _load_disambiguation_assets(self, enable=True):
        """Load the set-disambiguation artifacts (plan step 10 / A7).

        ``data/set_symbol_templates.pkl`` must exist and be fresh (its cache
        key is the mtime of ``set_disambiguation.py``) and
        ``data/plst_logo_template.npy`` must load; otherwise disambiguation
        stays
        disabled with a warning naming the offending artifact.  The rest of
        the pipeline (pHash matching) is unaffected either way.
        """
        root = os.path.dirname(os.path.abspath(__file__))
        pkl_path = os.path.join(root, "data", "set_symbol_templates.pkl")
        npy_path = os.path.join(root, "data", "plst_logo_template.npy")
        if not enable:
            print("Set disambiguation: disabled (enable_disambiguation=False).")
            return
        if not os.path.exists(pkl_path):
            print(f"Warning: set disambiguation disabled - missing "
                  f"'data/set_symbol_templates.pkl' (build it with "
                  f"scripts/build_set_symbol_templates.py).")
            return
        try:
            with open(pkl_path, "rb") as f:
                obj = pickle.load(f)
            if not (isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[1], dict)):
                raise ValueError("unexpected pickle format (cache_key, templates dict)")
            cache_key, templates = obj
            expected_key = os.path.getmtime(os.path.join(root, "set_disambiguation.py"))
            if cache_key != expected_key:
                print(f"Warning: set disambiguation disabled - stale "
                      f"'data/set_symbol_templates.pkl' (cache key {cache_key} != "
                      f"set_disambiguation.py mtime {expected_key}); rebuild it "
                      f"with scripts/build_set_symbol_templates.py.")
                return
            templates.pop("plst", None)  # non-templatable; logo detector only
            self.templates = templates
            self._disambig_src_mtime = expected_key
        except Exception as exc:
            print(f"Warning: set disambiguation disabled - cannot load "
                  f"'data/set_symbol_templates.pkl': {exc}")
            return
        if not os.path.exists(npy_path):
            print(f"Warning: set disambiguation disabled - missing "
                  f"'data/plst_logo_template.npy' (static asset shipped with "
                  f"the project; restore it, e.g. copy "
                  f"prototype/plst_logo_template.npy to data/).")
            return
        try:
            self.plst_logo_tpl = np.load(npy_path)
        except Exception as exc:
            print(f"Warning: set disambiguation disabled - cannot load "
                  f"'data/plst_logo_template.npy': {exc}")
            return
        self.disambiguation_enabled = True
        print(f"Set disambiguation: enabled "
              f"({len(self.templates)} set-symbol templates, "
              f"plst corner-logo template {self.plst_logo_tpl.shape}).")

    def _warn_disambig_drift(self):
        """One-time warning if ``set_disambiguation.py`` changed after the
        in-memory code + set-symbol templates were loaded (long-running
        process drift, e.g. app.py started before a code edit + pkl
        rebuild).  The running process stays self-consistent - it keeps
        using the startup versions of both - but results may not reflect
        the current code on disk.  Never disables disambiguation.
        """
        if self._disambig_src_mtime is None or self._disambig_drift_warned:
            return
        try:
            now = os.path.getmtime(os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "set_disambiguation.py"))
        except OSError:
            return
        if now != self._disambig_src_mtime:
            self._disambig_drift_warned = True
            print(f"WARNING: set_disambiguation.py changed after startup "
                  f"(loaded mtime {self._disambig_src_mtime}, now {now}). "
                  f"The in-memory set-disambiguation code and set-symbol "
                  f"templates are the startup versions - results stay "
                  f"self-consistent but may not reflect the current code. "
                  f"Restart the app (app.py) to pick up changes.")

    def _ref_image(self, scryfall_id):
        """Load a cached Scryfall reference image (488x680 BGR) by
        scryfall_id from ``cache/scryfall_images/``, or None when not cached.
        Used for the bottom set-name band evidence in disambiguation.
        """
        if scryfall_id in self._ref_img_cache:
            return self._ref_img_cache[scryfall_id]
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "cache", "scryfall_images", f"{scryfall_id}.jpg")
        img = cv2.imread(path) if os.path.exists(path) else None
        self._ref_img_cache[scryfall_id] = img
        return img

    @staticmethod
    def clean_card_name(name):
        """
        Remove special character tags at the end (e.g. parenthetical/bracket/brace tags)
        to keep only the real name, and sanitize for use as a valid filename.
        """
        # Remove trailing tags in parentheses/brackets/braces, e.g. "Card Name (Extended Art)" -> "Card Name"
        real_name = re.sub(r'\s*[\(\[\{<].*?[\)\]\}>]\s*$', '', name).strip()

        # For split/adventure cards with " // ", extract the primary card name,
        # e.g. "Silverflame Squire // On Alert" -> "Silverflame Squire"
        if ' // ' in real_name:
            primary_name = real_name.split(' // ')[0].strip()
        else:
            primary_name = real_name

        # Sanitize for valid filename across OSes
        clean_fn = re.sub(r'[<>:"/\\|?*]', '', primary_name).strip()
        return primary_name, clean_fn

    # Columns shared by the name/ID lookups below.
    _CARD_SELECT = (
        "SELECT scryfall_id, name, price_usd, price_usd_foil, price_usd_etched, "
        "illustration_id, scryfall_uri, tcgplayer_purchase_uri, card_type, edition, set_code, mana_cost, phash_back, type_line FROM cards"
    )

    # db_entries tuple layout: 14 elements, index 12 = back-face hash
    # (dual-face entries only), index 13 = type_line (raw Scryfall type line).
    @staticmethod
    def _face_dist(query_val, entry):
        """Min Hamming distance from a query hash to any face hash of an entry.

        Dual-face entries carry a back-face hash at index 12; single-face
        entries only use the front hash at index 2.
        """
        dist = (query_val ^ entry[2]).bit_count()
        back_val = entry[12]
        if back_val is not None:
            back_dist = (query_val ^ back_val).bit_count()
            if back_dist < dist:
                dist = back_dist
        return dist

    def _popcount_planes(self, x):
        """Per-row popcount of a (N, 4) uint64 array: four 16-bit lookup
        passes summed per row (256-bit Hamming counts)."""
        p = self._ph_pop16
        return (p[(x & 0xFFFF).astype(np.intp)]
                + p[((x >> 16) & 0xFFFF).astype(np.intp)]
                + p[((x >> 32) & 0xFFFF).astype(np.intp)]
                + p[((x >> 48) & 0xFFFF).astype(np.intp)]
                ).sum(axis=1, dtype=np.uint32)

    def _phash_dists(self, query_val):
        """Hamming distance from one 256-bit query hash to every DB entry's
        front and (when present) back hash, vectorized. Returns
        (front_dists, back_dists, min_dists) uint32 arrays of shape (N,);
        entries without a back hash get back distance 256 so their front
        always wins."""
        q = np.array([(query_val >> s) & 0xFFFFFFFFFFFFFFFF
                      for s in (0, 64, 128, 192)], dtype=np.uint64)
        df = self._popcount_planes(self._ph_front_u64 ^ q)
        db = self._popcount_planes(self._ph_back_u64 ^ q)
        db[~self._ph_has_back] = 256
        return df, db, np.minimum(df, db)

    def _multi_shift_search(self, warped_card):
        """Staged multi-shift pHash search over the whole warped card.

        For each of the 0/180 deg orientations and two artwork margins
        (standard 0.11-0.54 Y / 0.08-0.92 X, inner 0.11-0.55 Y / 0.10-0.90
        X) the artwork crop is pHash-probed on a coarse 3x3 crop-shift grid
        (sx +/-0.05, sy +/-0.04), refined on a 3x3 fine grid (sx +/-0.025,
        sy +/-0.02) around the local best, and - when the best distance is
        still above 60 - refined once more with halved steps (the coarse
        grid can straddle a narrow pHash valley) plus 3x3 neighborhoods
        around other coarse points within 10 of the best.

        Returns (best_dist, entry_index, face, query_val, card_img): the
        winning probe's minimum front/back distance, the winning db_entries
        index, the winning face ('back' only when the back hash is strictly
        closer), the winning query hash, and the full rotated card image of
        the winning rotation. best_dist stays 256 / entry_index None when
        nothing matched.
        """
        h, w = warped_card.shape[:2]
        best = 256
        best_idx = None
        best_face = 'front'
        best_query_val = 0
        best_card_img = warped_card
        for rot in (0, 180):
            card_img = warped_card if rot == 0 else cv2.rotate(warped_card, cv2.ROTATE_180)
            for (y0, y1, x0, x1) in ((0.11, 0.54, 0.08, 0.92), (0.11, 0.55, 0.10, 0.90)):
                ay1, ax1 = int(h * y0), int(w * x0)
                ay2, ax2 = int(h * y1), int(w * x1)
                hy, hx = ay2 - ay1, ax2 - ax1

                def q_at(sx, sy):
                    cy1 = ay1 + int(round(hy * sy))
                    cx1 = ax1 + int(round(hx * sx))
                    cy1 = max(0, min(h - hy, cy1))
                    cx1 = max(0, min(w - hx, cx1))
                    c = card_img[cy1:cy1 + hy, cx1:cx1 + hx]
                    return int(str(imagehash.phash(Image.fromarray(c), hash_size=16)), 16)

                local = None
                coarse_pts = []

                def probe(sx, sy):
                    nonlocal best, best_idx, best_face, best_query_val, best_card_img, local
                    q = q_at(sx, sy)
                    df, db, d = self._phash_dists(q)
                    i = int(d.argmin())
                    dv = int(d[i])
                    if dv < best:
                        best = dv
                        best_idx = i
                        best_face = 'back' if (self._ph_has_back[i] and db[i] < df[i]) else 'front'
                        best_query_val = q
                        best_card_img = card_img
                    # Tie-break toward the center so refinement does not
                    # anchor on a spurious equal-distance point far from
                    # (0,0).
                    if (local is None or dv < local[2]
                            or (dv == local[2]
                                and abs(sx) + abs(sy) < abs(local[0]) + abs(local[1]))):
                        local = (sx, sy, dv)
                    return dv

                for sy in (-0.04, 0.0, 0.04):
                    for sx in (-0.05, 0.0, 0.05):
                        v = probe(sx, sy)
                        coarse_pts.append((sx, sy, v))
                        if v == 0 or best == 0:
                            return best, best_idx, best_face, best_query_val, best_card_img
                clocal = (local[0], local[1])
                for dsy in (-0.02, 0.0, 0.02):
                    for dsx in (-0.025, 0.0, 0.025):
                        if dsx == 0.0 and dsy == 0.0:
                            continue
                        if probe(local[0] + dsx, local[1] + dsy) == 0 or best == 0:
                            return best, best_idx, best_face, best_query_val, best_card_img
                # Third stage: still not a confident match. The coarse grid
                # can straddle a narrow pHash valley (valley minimum between
                # grid points), so refine with a halved step around the
                # local best, and also around any other coarse point within
                # 10 dist of the current best (its neighborhood may hold the
                # true valley).
                if best > 60:
                    for dsy in (-0.01, 0.0, 0.01):
                        for dsx in (-0.0125, 0.0, 0.0125):
                            if dsx == 0.0 and dsy == 0.0:
                                continue
                            if probe(local[0] + dsx, local[1] + dsy) == 0 or best == 0:
                                return best, best_idx, best_face, best_query_val, best_card_img
                    for (px, py, pv) in coarse_pts:
                        if pv > best + 10 or (px, py) == clocal:
                            continue
                        for dsy in (-0.02, 0.0, 0.02):
                            for dsx in (-0.025, 0.0, 0.025):
                                if dsx == 0.0 and dsy == 0.0:
                                    continue
                                if probe(px + dsx, py + dsy) == 0 or best == 0:
                                    return best, best_idx, best_face, best_query_val, best_card_img
        return best, best_idx, best_face, best_query_val, best_card_img

    @staticmethod
    def _escape_like(value):
        """Escape LIKE wildcards (%, _, \\) so user input is matched literally."""
        return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    @staticmethod
    def _card_row_to_dict(row):
        scryfall_id, name, price_usd, price_usd_foil, price_usd_etched, illustration_id, scryfall_uri, tcgplayer_purchase_uri, card_type, edition, set_code, mana_cost, phash_back, type_line = row
        return {
            'scryfall_id': scryfall_id,
            'name': name,
            'price_usd': price_usd,
            'price_usd_foil': price_usd_foil,
            'price_usd_etched': price_usd_etched,
            'illustration_id': illustration_id,
            'scryfall_uri': scryfall_uri,
            'tcgplayer_purchase_uri': tcgplayer_purchase_uri,
            'card_type': card_type,
            'type_line': type_line,
            'edition': edition,
            'set_code': set_code,
            'mana_cost': mana_cost,
            'dual_face': phash_back is not None,
        }

    def get_by_scryfall_id(self, scryfall_id):
        """Return the card with this exact Scryfall id, or None if absent."""
        if not os.path.exists(self.db_path):
            return None
        try:
            conn = sqlite3.connect(self.db_path)
            row = conn.execute(self._CARD_SELECT + " WHERE scryfall_id = ?", (scryfall_id,)).fetchone()
            conn.close()
            return self._card_row_to_dict(row) if row else None
        except Exception as e:
            print(f"Error searching card database: {e}")
            return None

    def search_by_name(self, card_name):
        """
        Search for a card by name in the database. Prefers an exact
        (case-insensitive) match, then a "starts with" prefix match, then any
        partial match. Returns the card data or None if not found.
        """
        if not os.path.exists(self.db_path):
            return None
        try:
            conn = sqlite3.connect(self.db_path)
            escaped = self._escape_like(card_name)
            row = None
            # 1) Exact match (case-insensitive)
            row = conn.execute(
                self._CARD_SELECT + " WHERE name = ? COLLATE NOCASE LIMIT 1",
                (card_name,),
            ).fetchone()
            if row is None:
                # 2) "Starts with" (case-insensitive; uses the idx_name index)
                row = conn.execute(
                    self._CARD_SELECT + " WHERE name LIKE ? ESCAPE '\\' ORDER BY name LIMIT 1",
                    (escaped + "%",),
                ).fetchone()
            if row is None:
                # 3) Anywhere in the name (legacy behaviour)
                row = conn.execute(
                    self._CARD_SELECT + " WHERE name LIKE ? ESCAPE '\\' LIMIT 1",
                    ("%" + escaped + "%",),
                ).fetchone()
            conn.close()
            return self._card_row_to_dict(row) if row else None
        except Exception as e:
            print(f"Error searching card database: {e}")
            return None

    def suggest_names(self, prefix, limit=15):
        """
        Return cards whose name *starts with* `prefix` (case-insensitive),
        for typeahead autocomplete. Runs on demand against SQLite so nothing
        needs to be preloaded. Returns {"suggestions": [...], "truncated": bool}
        where each suggestion holds scryfall_id, name, edition and set_code.
        One entry per distinct name + edition (set) is returned, preferring a
        row that has a regular price.
        """
        prefix = (prefix or "").strip()
        if not prefix or not os.path.exists(self.db_path):
            return {"suggestions": [], "truncated": False}
        try:
            conn = sqlite3.connect(self.db_path)
            # 1) Distinct name + edition groups matching the prefix, in name
            #    order. GROUP BY (name, set_code) rides the idx_name_set index
            #    and stops as soon as enough groups are collected, so even a
            #    single-letter prefix stays fast.
            groups = conn.execute(
                "SELECT name, set_code FROM cards "
                "WHERE name LIKE ? ESCAPE '\\' "
                "GROUP BY name, set_code ORDER BY name, set_code LIMIT ?",
                (self._escape_like(prefix) + "%", limit + 1),
            ).fetchall()
            truncated = len(groups) > limit
            groups = groups[:limit]
            if groups:
                # 2) Representative row per group: the same card can have
                #    several rows (foil/etched/etc. variants); prefer a row
                #    that has a regular price.
                pairs = " OR ".join(["(name = ? AND set_code = ?)"] * len(groups))
                params = [v for group in groups for v in group]
                rows = conn.execute(
                    "SELECT scryfall_id, name, set_code, edition FROM ("
                    "  SELECT scryfall_id, name, set_code, edition,"
                    "    ROW_NUMBER() OVER ("
                    "      PARTITION BY name, set_code"
                    "      ORDER BY (price_usd IS NULL), price_usd, scryfall_id"
                    "    ) AS variant_rank"
                    "  FROM cards WHERE " + pairs +
                    ") WHERE variant_rank = 1",
                    params,
                ).fetchall()
            else:
                rows = []
            conn.close()
            by_key = {
                (name, set_code): {
                    "scryfall_id": scryfall_id, "name": name,
                    "edition": edition, "set_code": set_code,
                }
                for scryfall_id, name, set_code, edition in rows
            }
            return {
                "suggestions": [by_key[g] for g in groups if g in by_key],
                "truncated": truncated,
            }
        except Exception as e:
            print(f"Error searching card database: {e}")
            return {"suggestions": [], "truncated": False}

    def _resolve_set(self, best_entry, best_query_val, best_card_img,
                     progress_cb=None, frac_fn=None):
        """Phase 4: pick the set for a pHash-best match whose artwork is
        reprinted in several sets (same illustration_id).  Plan 4.4-4.5.

        ``progress_cb``/``frac_fn`` (plan §3, feature A): when both are given,
        the band sub-progress is reported per candidate as
        ``"{name} — verifying set k/n (set_code)"`` with absolute fraction
        ``frac_fn(k, n)``.

        Returns ``(set_status, candidates, chosen_entry, symbol_margin,
        logo_score)``:
          set_status    'unique' (single-set artwork) | 'resolved' |
                        'ambiguous' | 'unverified'
          candidates    [{set_code, set_name, scryfall_id, phash_dist,
                          score}] per candidate set (score None when the set
                          is unscored); [] for single-set artworks
          chosen_entry  db row (13-tuple) whose values the match reports
          symbol_margin top1 - top2 score margin (None when < 2 scored sets)
          logo_score    plst corner-logo score (None when not computed)
        """
        self._warn_disambig_drift()
        ill = best_entry[11]
        if not ill:
            return "unique", [], best_entry, None, None
        group = self.ill_index.get(ill)
        if not group:
            return "unique", [], best_entry, None, None
        # Representative row per candidate set: the group entry with the
        # smallest pHash distance to the winning crop; ties go to the earlier
        # entry index (deterministic).
        reps = {}  # set_code -> (phash_dist, entry_index)
        for idx in group:
            entry = self.db_entries[idx]
            d = self._face_dist(best_query_val, entry)
            cur = reps.get(entry[10])
            if cur is None or d < cur[0]:
                reps[entry[10]] = (d, idx)
        if len(reps) <= 1:
            return "unique", [], best_entry, None, None
        # 2+ sets share the artwork: score the set symbols.  Disabled or
        # failing disambiguation -> 'unverified', keep the pHash-best entry.
        if self.disambiguation_enabled:
            # Per-candidate reference images enable the bottom set-name band
            # evidence (separates sets with identical corner symbols, e.g.
            # gk2 vs prw2).  Missing cached images are simply skipped.
            ref_imgs = {}
            for sc in sorted(reps):
                ref = self._ref_image(self.db_entries[reps[sc][1]][0])
                if ref is not None:
                    ref_imgs[sc] = ref
            band_progress = None
            if progress_cb is not None and frac_fn is not None:
                name = best_entry[1]
                def band_progress(k, n, sc):
                    if k <= n:
                        msg = f"{name} — verifying set {k}/{n} ({sc})"
                    else:  # staged-band stage-2 re-report (virtual k = n + j)
                        j = k - n
                        m = min(sdis.BAND_FINE_LIMIT, n)
                        msg = f"{name} — refining set {j}/{m} ({sc})"
                    progress_cb(msg, frac_fn(k, n))
            try:
                winner, scores, logo_score, state = sdis.disambiguate(
                    best_card_img, sorted(reps), self.templates,
                    self.plst_logo_tpl, ref_imgs=ref_imgs,
                    band_progress=band_progress,
                    staged_band=self.staged_band)
            except Exception as exc:
                print(f"  Set disambiguation failed ({exc}); "
                      f"using the pHash best entry instead.")
                winner, scores, logo_score, state = None, {}, None, "unverified"
        else:
            winner, scores, logo_score, state = None, {}, None, "unverified"
        if state not in ("resolved", "ambiguous") or winner not in reps:
            chosen = best_entry
        else:
            chosen = self.db_entries[reps[winner][1]]
        margin = None
        if len(scores) >= 2:
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            margin = ranked[0][1] - ranked[1][1]
        candidates = [
            {
                "set_code": sc,
                "set_name": self.set_names.get(sc),
                "scryfall_id": self.db_entries[reps[sc][1]][0],
                "phash_dist": reps[sc][0],
                "score": scores.get(sc),
            }
            for sc in sorted(reps)
        ]
        candidates.sort(key=lambda c: (c["score"] is None,
                                       -(c["score"] or 0.0),
                                       c["phash_dist"], c["set_code"]))
        if "plst" not in reps:
            logo_score = None
        return state, candidates, chosen, margin, logo_score

    def find_match(self, warped_card, max_dist=75, finish="nonfoil", label=None,
                   progress_cb=None, frac_fn=None):
        """
        Extract artwork zone from warped card, compute 256-bit pHash, and find
        best match in database. The search is a staged multi-shift probe
        (see ``_multi_shift_search``): 0/180 deg orientations x two artwork
        margins (standard and inner frame), each probed on a coarse 3x3
        crop-shift grid and refined around the local best. The shift grids
        absorb residual warp/outline error that would leave a fixed margin a
        pHash valley away from the true artwork. `finish` selects
        which price to fetch ("nonfoil", "foil" or "etched"), falling back to the
        non-foil price if no price is known for that finish. If `label` is given,
        prints the match attempt and result (with pHash distance) to the console.

        Phase 4: when the winning card's illustration_id is shared by several
        sets, the set is resolved with the set-symbol scorer
        (``set_disambiguation.disambiguate``) and the result carries the extra
        fields ``scryfall_id``, ``set_status``, ``candidates``,
        ``symbol_margin`` and ``logo_score`` (plan 4.6).

        ``progress_cb``/``frac_fn`` are passed through to the set
        disambiguation band sub-progress (plan §3, feature A); ``None`` = no
        reporting.
        """
        if not self.db_entries:
            return None

        # Staged multi-shift pHash search (see _multi_shift_search): 0/180
        # deg x two artwork margins, each probed on a coarse 3x3 crop-shift
        # grid and refined around the local best. Replaces the old
        # fixed-margin crop loop: pHash distance is very sensitive to crop
        # placement, and the shift grid absorbs warp/outline errors that
        # used to leave clearly-visible cards above max_dist.
        best_dist, best_idx, best_face, best_query_val, best_card_img = \
            self._multi_shift_search(warped_card)
        best_name = self.db_entries[best_idx][1] if best_idx is not None else None
        best_entry = self.db_entries[best_idx] if best_idx is not None else None

        if best_dist <= max_dist and best_name:
            # Phase 4: 'unique' keeps the legacy pHash-best values; multi-set
            # artworks are resolved via the set-symbol scorer (plan 4.4).
            set_status, candidates, chosen, symbol_margin, logo_score = \
                self._resolve_set(best_entry, best_query_val, best_card_img,
                                  progress_cb=progress_cb, frac_fn=frac_fn)
            scryfall_id, name, _db_val, price_usd, price_usd_foil, price_usd_etched, scryfall_uri, tcgplayer_purchase_uri, card_type, edition, set_code, _illustration_id, _db_val_back, type_line = chosen
            # When the back face of a dual-face card matched, report that
            # face's name (card identity/price stay the dual card's).
            display_name = name
            if best_face == 'back' and ' // ' in name:
                display_name = name.split(' // ', 1)[1].strip()
            real_name, clean_fn = self.clean_card_name(display_name)
            if finish == "foil":
                real_name += " (Foil)"
                clean_fn += "_foil"
                price = price_usd_foil if price_usd_foil else price_usd
            elif finish == "etched":
                real_name += " (Etched Foil)"
                clean_fn += "_etched"
                price = price_usd_etched if price_usd_etched else price_usd
            else:
                price = price_usd
            if label:
                suffix = ""
                if set_status in ("resolved", "ambiguous"):
                    suffix = f" [{set_status}: {set_code}"
                    if symbol_margin is not None:
                        suffix += f", margin {symbol_margin:.2f}"
                    suffix += "]"
                elif set_status == "unverified":
                    suffix = f" [unverified: {set_code}]"
                print(f"{label}: matched '{real_name}' (distance {best_dist}/{max_dist}){suffix}")
            return {
                'raw_name': name,
                'real_name': real_name,
                'clean_filename': clean_fn,
                'matched_face': best_face,
                'dual_face': _db_val_back is not None,
                'distance': best_dist,
                'card_image': best_card_img,
                'price_usd': price,
                # Raw per-finish prices for this edition (before the
                # requested-finish selection above). Lets the web app re-price
                # the card for any finish on the per-card finish toggle.
                'price_usd_nonfoil': price_usd,
                'price_usd_foil': price_usd_foil,
                'price_usd_etched': price_usd_etched,
                'scryfall_uri': scryfall_uri,
                'tcgplayer_purchase_uri': tcgplayer_purchase_uri,
                'finish': finish,
                'card_type': card_type,
                'type_line': type_line,
                'edition': edition,
                'set_code': set_code,
                'scryfall_id': scryfall_id,
                'set_status': set_status,
                'candidates': candidates,
                'symbol_margin': symbol_margin,
                'logo_score': logo_score,
            }

        if label:
            closest = f" (closest: '{best_name}', distance {best_dist}/{max_dist})" if best_name else ""
            print(f"{label}: no match found{closest}")

        return None


def wrap_text_lines(text, font, font_scale, thickness, max_width):
    """Greedy word-wrap `text` into lines that each fit within max_width px."""
    words = text.split(' ')
    lines = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip() if current else word
        width = cv2.getTextSize(candidate, font, font_scale, thickness)[0][0]
        if width <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_card_label(debug_image, box, text, price_text=None):
    """
    Draw a semi-transparent gray label box with a thin black border and
    black text showing the card name, centered on the card and rotated to
    match its orientation. Sized relative to the card so it stays readable
    whether there's a single large card or many small ones in the photo,
    and shrunk/wrapped so it never crosses the green outline. If
    `price_text` is given, it's appended as its own line below the name.
    """
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(box)
    if rw == 0 or rh == 0:
        return
    if rw < rh:
        box_w, box_h, draw_angle = rw, rh, angle
    else:
        box_w, box_h, draw_angle = rh, rw, angle + 90

    font = cv2.FONT_HERSHEY_SIMPLEX
    max_text_width = max(10, int(box_w * 0.75))
    max_box_height = box_h * 0.4

    font_scale = max(0.3, box_w / 190.0)
    min_font_scale = max(0.25, box_w / 900.0)

    lines, sizes, line_height, widest, block_h = [text], [], 0, 0, 0
    while True:
        thickness = max(1, int(round(font_scale * 2)))
        padding = max(4, int(round(font_scale * 6)))
        line_gap = max(2, int(round(font_scale * 3)))
        lines = wrap_text_lines(text, font, font_scale, thickness, max_text_width)
        if price_text:
            lines.extend(wrap_text_lines(price_text, font, font_scale, thickness, max_text_width))
        sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
        line_height = max(s[1] for s in sizes)
        widest = max(s[0] for s in sizes)
        block_h = len(lines) * line_height + (len(lines) - 1) * line_gap + 2 * padding
        if (block_h <= max_box_height and widest <= max_text_width) or font_scale <= min_font_scale:
            break
        font_scale = max(min_font_scale, font_scale * 0.9)

    box_width = min(int(box_w * 0.92), widest + 2 * padding)
    box_height = block_h

    # Render the label on its own BGRA canvas so it can be rotated and
    # alpha-blended onto the debug image as a single warp.
    patch = np.zeros((box_height, box_width, 4), dtype=np.uint8)
    patch[:, :, :3] = 128
    patch[:, :, 3] = 190
    border_thickness = max(1, int(round(font_scale * 2)))
    cv2.rectangle(patch, (0, 0), (box_width - 1, box_height - 1), (0, 0, 0, 255), border_thickness)

    y = padding + line_height
    for line, size in zip(lines, sizes):
        x = (box_width - size[0]) // 2
        cv2.putText(patch, line, (x, y), font, font_scale, (0, 0, 0, 255), thickness, cv2.LINE_AA)
        y += line_height + line_gap

    xs, ys = box[:, 0], box[:, 1]
    margin = 5
    x_min = max(0, int(xs.min()) - margin)
    y_min = max(0, int(ys.min()) - margin)
    x_max = min(debug_image.shape[1], int(xs.max()) + margin)
    y_max = min(debug_image.shape[0], int(ys.max()) + margin)
    roi_w, roi_h = x_max - x_min, y_max - y_min
    if roi_w <= 0 or roi_h <= 0:
        return

    M = cv2.getRotationMatrix2D((box_width / 2.0, box_height / 2.0), draw_angle, 1.0)
    M[0, 2] += cx - x_min - box_width / 2.0
    M[1, 2] += cy - y_min - box_height / 2.0

    warped_patch = cv2.warpAffine(
        patch, M, (roi_w, roi_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
    )

    roi = debug_image[y_min:y_max, x_min:x_max].astype(np.float32)
    alpha = warped_patch[:, :, 3:4].astype(np.float32) / 255.0
    blended = alpha * warped_patch[:, :, :3].astype(np.float32) + (1 - alpha) * roi
    debug_image[y_min:y_max, x_min:x_max] = blended.astype(np.uint8)


def sort_boxes(boxes, image):
    """Sort card boxes left-to-right, top-to-bottom for consistent naming."""
    def box_center(b):
        c = b.mean(axis=0)
        return (round(c[1] / (image.shape[0] / 10)), c[0])
    boxes.sort(key=box_center)
    return boxes


def render_debug_overlay(image, boxes, card_labels, show_price=True):
    """
    Draw the green detection outlines and the rotated name/price labels.
    `card_labels` holds one entry per box (same order): either
    (label_text, price_value) or None for boxes without a label (e.g. an
    unmatched card). Returns the annotated image; `image` is not modified.
    """
    debug_image = image.copy()
    line_thickness = max(2, round(max(image.shape[:2]) / 500))
    for box in boxes:
        pts = box.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(debug_image, [pts], isClosed=True, color=(0, 255, 0), thickness=line_thickness)
    for box, label_spec in zip(boxes, card_labels):
        if not label_spec:
            continue
        label_text, price_value = label_spec
        price_text = f"${float(price_value):.2f}" if show_price and price_value else None
        draw_card_label(debug_image, box, label_text, price_text=price_text)
    return debug_image


def process_image(input_path, output_dir, db=None, finish="nonfoil", show_price=True, progress_cb=None):
    """
    Process a single image. `progress_cb`, if given, is called with
    (message, frac) as processing advances: `message` is a short
    human-readable step description and `frac` is this image's progress as a
    fraction 0..1 (a small slice for card detection, then one step per
    detected card). Returns (export_rows, info) where `info` holds
    paths/metadata useful for a web UI (debug image, per-card crops).
    """
    # In-image progress fractions: detection is a small slice, the rest is
    # spread over the detected cards so a UI bar advances per card.
    F_READ, F_DETECT, F_MATCH, F_OVERLAY, F_SAVE, F_DONE = 0.02, 0.10, 0.80, 0.90, 0.98, 1.0

    def report(msg, frac):
        if progress_cb:
            progress_cb(msg, frac)

    report("Reading image", 0.0)
    image = cv2.imread(input_path)
    if image is None:
        print(f"Could not read image: {input_path}")
        return [], {"debug_path": None, "cards": []}

    report("Detecting card boundaries", F_READ)
    boxes = find_card_quads(image)
    print(f"{os.path.basename(input_path)}: found {len(boxes)} card candidate(s).")

    if not boxes:
        print("Warning: no cards detected. Adjust thresholds if needed.")

    sort_boxes(boxes, image)

    stem = os.path.splitext(os.path.basename(input_path))[0]
    n = len(boxes)

    def match_frac(i):
        """Progress fraction after i of n cards have been matched."""
        return F_DETECT + (F_MATCH - F_DETECT) * (i / n) if n else F_MATCH

    def save_frac(i):
        """Progress fraction after i of n card crops have been saved."""
        return F_OVERLAY + (F_SAVE - F_OVERLAY) * (i / n) if n else F_SAVE

    report(f"Detected {n} card(s); matching against database", F_DETECT)
    warped_cards = []
    for i, box in enumerate(boxes, start=1):
        warped = warp_card(image, box)
        warped = orient_upright(warped)
        report(f"Matching card {i}/{n}", match_frac(i - 1))
        _bm_kwargs = {}
        if progress_cb is not None:
            lo, hi = match_frac(i - 1), match_frac(i)
            def _band_frac(k, n_, _lo=lo, _hi=hi):
                if k <= n_:
                    return _lo + (_hi - _lo) * ((k - 1) / max(n_, 1))
                # Staged-band stage-2 re-report (virtual k = n + j):
                # forward-map into (last stage-1 frac, _hi) so the bar never
                # steps back; the completion event stays the first thing AT _hi.
                j = k - n_
                m = min(sdis.BAND_FINE_LIMIT, n_)
                last1 = _lo + (_hi - _lo) * ((n_ - 1) / max(n_, 1))
                return last1 + (_hi - last1) * ((j - 0.5) / m)
            _bm_kwargs = {"progress_cb": report, "frac_fn": _band_frac}
        match = db.find_match(warped, finish=finish, label=f"  Card {i}/{n}",
                              **_bm_kwargs) if db else None
        if match:
            report(f"Card {i}/{n}: matched '{match['real_name']}' (distance {match['distance']})", match_frac(i))
        else:
            report(f"Card {i}/{n}: no match found", match_frac(i))
        warped_cards.append((box, warped, match))

    report("Rendering overlay", F_MATCH)
    card_labels = []
    for _box, _warped, match in warped_cards:
        if not match:
            card_labels.append(None)
            continue
        finish_tag = {"foil": " (Foil)", "etched": " (Etched Foil)"}.get(match.get('finish'), "")
        set_tag = f" [{match['set_code']}]" if match.get('set_code') else ""
        card_labels.append((match['raw_name'] + finish_tag + set_tag, match.get('price_usd')))
    debug_image = render_debug_overlay(image, boxes, card_labels, show_price=show_price)
    finish_suffix = {"foil": "_foil", "etched": "_etched"}.get(finish, "")
    debug_path = os.path.join(output_dir, f"{stem}-detected{finish_suffix}.jpg")
    cv2.imwrite(debug_path, debug_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Saved {debug_path}")

    report("Saving extracted card images", F_OVERLAY)
    export_rows = []
    card_infos = []
    for i, (_box, warped, match) in enumerate(warped_cards, start=1):
        if match:
            clean_fn = match['clean_filename']
            out_path = os.path.join(output_dir, f"{stem}-card{i}_{clean_fn}.png")
            cv2.imwrite(out_path, match['card_image'])
            print(f"Saved {out_path} ({match['card_image'].shape[1]}x{match['card_image'].shape[0]}, Matched: '{match['real_name']}')")
            # CSV reports only the selected (winning) set, not the full
            # same-artwork candidate list.
            export_rows.append((match['real_name'], os.path.basename(input_path), i, match['price_usd'], match['finish'], match.get('set_code') or "", match.get('type_line') or ""))
            card_infos.append({
                "name": match['real_name'],
                "price_usd": match['price_usd'],
                "price_usd_nonfoil": match['price_usd_nonfoil'],
                "price_usd_foil": match['price_usd_foil'],
                "price_usd_etched": match['price_usd_etched'],
                "finish": match['finish'],
                "distance": match['distance'],
                "scryfall_uri": match['scryfall_uri'],
                "tcgplayer_purchase_uri": match['tcgplayer_purchase_uri'],
                "crop_path": out_path,
                "card_type": match.get('card_type'),
                "type_line": match.get('type_line'),
                "edition": match.get('edition'),
                "set_code": match.get('set_code'),
                "scryfall_id": match.get('scryfall_id'),
                "set_status": match.get('set_status'),
                "candidates": match.get('candidates'),
                "symbol_margin": match.get('symbol_margin'),
                "dual_face": match.get('dual_face'),
                "matched_face": match.get('matched_face'),
            })
        else:
            out_path = os.path.join(output_dir, f"{stem}-card{i}.png")
            cv2.imwrite(out_path, warped)
            print(f"Saved {out_path} ({warped.shape[1]}x{warped.shape[0]})")
            export_rows.append(("", os.path.basename(input_path), i, None, finish, "", ""))
            card_infos.append({
                "name": None,
                "price_usd": None,
                "finish": finish,
                "distance": None,
                "scryfall_uri": None,
                "tcgplayer_purchase_uri": None,
                "crop_path": out_path,
                "card_type": None,
                "type_line": None,
                "edition": None,
                "set_code": None,
                "scryfall_id": None,
                "set_status": None,
                "candidates": [],
                "symbol_margin": None,
                "dual_face": None,
                "matched_face": None,
            })
        report(f"Saved card {i}/{n}", save_frac(i))

    report("Finished", F_DONE)
    info = {"debug_path": debug_path, "cards": card_infos}
    return export_rows, info


def write_csv_export(rows, output_dir, filename="cards_export.csv"):
    """Write the collected card rows (name, input file, card number, price usd,
    finish, set, type_line) to a CSV file.  ``set`` is the selected set code of
    the matched card — for same-artwork multi-set cards the disambiguation
    winner, e.g. ``one1`` ("" for unmatched cards).  ``type_line`` is the
    card's raw Scryfall type line ("" for unmatched cards).  Written as
    UTF-8 **with BOM** (``utf-8-sig``) so Windows Excel detects the encoding
    and renders the em-dash type lines correctly instead of as ``â€”``
    (BOM-less UTF-8 opens as cp1252); readers using ``utf-8-sig`` strip the
    BOM, and plain ``utf-8`` readers see it only in the first header cell."""
    csv_path = os.path.join(output_dir, filename)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["card_name", "input_file", "card_number", "price_usd", "finish", "set", "type_line"])
        for name, input_file, card_number, price_usd, finish, set_code, type_line in rows:
            writer.writerow([name, input_file, card_number, price_usd if price_usd is not None else "", finish, set_code or "", type_line or ""])
    print(f"Saved {csv_path} ({len(rows)} row(s))")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input_dir = os.path.join(script_dir, "input")

    parser = argparse.ArgumentParser(
        description="Detect and extract card images from photos in the input folder."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=default_input_dir,
        help="Input folder, or a single image file (path, or just a filename inside the input folder).",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=os.path.join(script_dir, "output"),
        help="Folder to write extracted card images to.",
    )
    finish_group = parser.add_mutually_exclusive_group()
    finish_group.add_argument(
        "--foil",
        action="store_true",
        help="Treat all cards in this run as foil: fetch the foil USD price and tag matches as foil.",
    )
    finish_group.add_argument(
        "--etched",
        action="store_true",
        help="Treat all cards in this run as etched foil: fetch the etched USD price and tag matches as etched.",
    )
    args = parser.parse_args()
    finish = "foil" if args.foil else "etched" if args.etched else "nonfoil"

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    db_path = os.path.join(script_dir, "data", "scryfall_db.sqlite")
    if not os.path.exists(db_path):
        db_path = "data/scryfall_db.sqlite"

    db = CardDatabase(db_path) if os.path.exists(db_path) else None

    input_path = args.input
    if not os.path.isfile(input_path) and not os.path.isdir(input_path):
        # Treat it as a filename relative to the default input folder.
        candidate = os.path.join(default_input_dir, input_path)
        if os.path.isfile(candidate):
            input_path = candidate

    if os.path.isfile(input_path):
        rows, _info = process_image(input_path, output_dir, db=db, finish=finish)
        write_csv_export(rows, output_dir)
        return

    if not os.path.isdir(input_path):
        print(f"Input not found: {input_path}")
        sys.exit(1)

    input_files = sorted(
        f for f in os.listdir(input_path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )

    if not input_files:
        print(f"No images found in {input_path}")
        sys.exit(1)

    all_rows = []
    for filename in input_files:
        rows, _info = process_image(os.path.join(input_path, filename), output_dir, db=db, finish=finish)
        all_rows.extend(rows)

    write_csv_export(all_rows, output_dir)


if __name__ == "__main__":
    main()
