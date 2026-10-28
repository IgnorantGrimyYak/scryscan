# Copyright (C) 2026 IgnorantGrimyYak
"""Same-artwork set disambiguation (production module).

Resolves which print run (Magic set code) a photo of a Magic: The Gathering
card belongs to when several sets share the same illustration.  Method:
score the card's set symbol (the small stamp below the art box, top-left)
against one majority-vote template per candidate set.

This module is a production port of the prototype (now in ``prototype/``)
that produced the run7 results (``prototype/evalA_run7.csv`` /
``prototype/evalB_run7.csv``, template cache
``prototype/tmpl_cache/full_sweep_run7.pkl``):

* Template extraction (``extract_template_multi``), scoring
  (``match_score`` / ``_resized`` / SCALES sweep) are ported verbatim from
  ``prototype/eval_within_group.py`` (run7).
* The ``plst`` (The List) corner-logo detector is ported verbatim from
  ``prototype/plst_logo_detector.py``, with the logo template passed in as an
  argument instead of a module-level global.
* ``stable_true`` / ``ALL_GROUPS_SQL`` are ported verbatim from
  ``prototype/full_sweep.py`` and exist only for the verification harness.

Reference geometry: templates live in the 488x680 reference image space
(``cache/scryfall_images/<scryfall_id>.jpg``).  Runtime card photos are the warped
portrait cards produced by ``extract_cards.warp_card`` (~1000x1400);
``symbol_zone`` crops the symbol zone in photo space and scoring uses
``scale_factor = PHOTO_W / REF_W`` to enlarge the template to photo scale
(before each SCALES pass), exactly as the prototype's experiment B did.

Conventions (all as in run7):
  * binary masks are 1 == ink (dark pixels below ``MASK_THR``);
  * ``match_score`` = (template-ink covered by query ink) - 1.2 * (uncovered
    query ink inside the template bbox), normalized by template ink area;
  * scores are clamped to ``[0, 1]`` by the 0.0 floor of the per-set sweep;
  * ``plst`` has no corner symbol (every card keeps its original set's
    frame and symbol) and is non-templatable: it is excluded from the
    templates and only reachable through the B-mode corner-logo boost
    (``logo_score_photo`` / ``PLST_LOGO_TH``).

The module is stateless (one internal resize cache) and has no CLI.  It does
not open ``data/scryfall_db.sqlite``: candidate sets, reference images and the
``plst`` logo template are all supplied by the caller (``CardDatabase``
builds the templates and loads the logo template).

Public API:
    build_templates(sets)               -> {set_code: (template_gray, mask)}
    extract_template_multi(sids)        -> (template_gray, mask) | None
    load_ref(ref)                       -> gray ndarray (ref = path or array)
    symbol_zone(card_bgr)               -> gray symbol-zone crop (photo space)
    binarize(img)                       -> uint8 mask (1 == ink)
    score_candidate_sets(zone_bin, scale_factor, codes, templates)
                                       -> {set_code: score in [0, 1]}
    band_score(photo_gray, ref_gray)  -> float (set-name band evidence)
    text_score(photo_gray, ref_gray)  -> float (set-code + copyright tiebreak)
    disambiguate(card_bgr, candidate_sets, templates, logo_tpl)
                                       -> (winner, scores, logo_score, state)
    logo_score_ref(ref, logo_tpl)       -> A-mode (488x680) plst logo score
    logo_score_photo(photo, logo_tpl)   -> B-mode (1000x1400) plst logo score
    has_plst_logo(card_bgr, logo_tpl)   -> bool
    make_photo(gray488, seed)           -> 1000x1400 simulated photo (harness)
    stable_true(ill, codes)             -> deterministic true-set label (harness)
    ALL_GROUPS_SQL                      -> multi-set group query (harness)

``disambiguate`` decision states:
    'resolved'    exactly one candidate scored, or top1-top2 >= AMBIGUOUS_MARGIN
    'ambiguous'   two+ candidates scored within AMBIGUOUS_MARGIN of each other
    'unverified'  no candidate could be scored at all
"""

import os
import random
import zlib

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Geometry / constants (ported verbatim from prototype/eval_within_group.py,
# prototype/full_sweep.py and prototype/plst_logo_detector.py)
# ---------------------------------------------------------------------------

# Reference image size (cache/scryfall_images/*.jpg)
REF_W, REF_H = 488, 680

# Photo size (warped card from extract_cards.warp_card, portrait)
PHOTO_W, PHOTO_H = 1000, 1400

# reference-image coordinates (488x680)
# Voting region: x starts after the type-line text (~x<=390), ends before the
# per-card card-edge shadow (x>=450 on 2021+ frames); y below the art bottom
# line / decor band, above the rule-box top line. Badge (58x25 max) fits.
VOTE_REGION = (392, 384, 450, 424)
ALIGN_WIN = 8                        # per-card shift search window (px)
# Query zone: symbol box (symbol at x392-450, y389-415, +-4px jitter),
# extended right/bottom to (468, 426) — photo x793-959, y788-877: some photo
# warps land the symbol ~+25 px right / +10 px down (c18-300 swamp) and the
# old tight edge (x457 / y421) clipped ~10 px of its ink.  The added right
# strip (card-edge shadow on 2021+ frames) and bottom strip (rule line) are
# background ink for nominally aligned photos; scoring is robust to it
# because match_score only penalizes query ink inside the matched template
# bbox.  Extending much further is filled with type-line text ink, which
# makes containment of ANY template ~0.7-0.9 (dense background) and kills
# discrimination.
ZONE = (387, 383, 468, 426)
SYMBOL_CENTER = (430.0, 402.0)
SCALES = (0.85, 0.95, 1.0, 1.05, 1.15)
MASK_THR = 150

# Sets with no corner symbol: never templatable; plst is answered via the
# B-mode corner-logo boost instead (see full_sweep.py Step 5/6 notes).
NO_TEMPLATE_SETS = {'plst'}

# plst (The List) corner-logo detector geometry (plst_logo_detector.py)
# A mode: clean 488x680 reference, left-zone search window.
A_ROI = (630, 670, 0, 60)
# B mode: intermediate 415-wide photo space (exact pre-upscale geometry of
# make_photo), slightly widened bottom-left window.
B_SPACE = (415, 582)
B_ROI = (530, 582, 0, 60)
_B_SCALE = 415.0 / 488.0
PLST_LOGO_TH = 0.80

# Production decision margin (INTEGRATION_PLAN phase 1)
AMBIGUOUS_MARGIN = 0.05

# Bottom set-name band discriminator.  Modern frames print the set name in a
# dark horizontal band near the card bottom; the printed text differs per set
# even when the corner symbol is identical (e.g. gk2 vs prw2, both Gruul).
# Reference-space rows of the band strip matched into the photo:
BAND_REF_Y = (624, 654)
# Horizontal extent of the strip: the set name starts at x~32; the collector
# number on the right (x>250) is common to all sets and dilutes the
# discrimination, so the strip is cropped before it (x 0-330 measured best).
BAND_REF_X = (0, 330)
# Photo-space search window: the perspective warp does not place the band at a
# fixed photo row (~20 px of slack, plus make_photo rotation in the harness),
# so the strip is matched across the bottom window instead of a fixed crop.
BAND_WINDOW_TOP = 1240
# The detected card quad usually sits slightly INSIDE the true card edge, so
# the perspective warp magnifies the card beyond the nominal ~2.05x (bottom
# band regions have been measured at ~2.2x) and the warped card can extend
# past the left/right photo edges.  The scale range must cover that stretch.
# Coarse grid: 0.05 scale step (was 0.1).  The 0.1 step undershot photos
# whose warp magnification falls between grid points in a way the fine pass
# cannot recover for non-finalists (token_03 Elf Warrior: true set tori
# peaked at s~2.045, the 0.1 grid best was s=2.1, coarse rank 8/15, no fine
# pass, correct razor auto-resolution lost).  11 scales x 5 angles = 55
# evaluations per candidate (~1.8x the old 30) — the user chose accuracy
# over speed (PLAN_BAND_IMPROVEMENTS.md §9 deviation log).
BAND_SCALES = (1.8, 1.85, 1.9, 1.95, 2.0, 2.05,
               2.1, 2.15, 2.2, 2.25, 2.3)
BAND_ANGLES = (-2.0, -1.0, 0.0, 1.0, 2.0)
# Pixels of BORDER_REPLICATE padding added to the left/right of the band
# search window so the template may be placed partly outside the photo
# (card clipped at the warp edge); otherwise the optimum hits the image
# boundary and the true set loses the band match.
BAND_WIN_PAD = 300
# Weight of the band evidence added to the per-set symbol score.  Full
# weight: the band is an independent per-set signal matched against the
# candidate's own reference card (its printed set name), so for a correct
# candidate it reinforces rather than competes with the symbol score.
BAND_WEIGHT = 1.0
# Number of finalists (by current symbol+coarse score) that receive the
# pass-2 fine band refinement in the staged protocol
# (``disambiguate(..., staged_band=True)``); the other candidates keep their
# coarse score.  See PLAN_BAND_IMPROVEMENTS.md §4.
BAND_FINE_LIMIT = 2

# Bottom text-region tiebreak (set-code/collector line + copyright line).
# Modern frames (2015+) print the set code below the collector number
# ("SET • LANG") and the copyright line to the right; the 2003 frame prints
# the copyright range + collector total.  Both regions are constant per set
# and carry the printed set identity even when the corner symbol is identical
# across sets and the set-NAME band does not separate them (the band strip
# BAND_REF_X/Y ends above the set-code line and before the copyright year,
# and per-card text in its window dilutes the match).  The reference strips
# are matched as plain image templates (no OCR): the printed text differs
# per set exactly where it needs to.
TEXT_L_BOX = (26, 632, 114, 666)    # collector number + set-code line (left)
TEXT_R_BOX = (316, 630, 472, 666)   # copyright line (right)
# Photo-space search window (rows): wider than the band window because the
# 2003-frame band sits higher and warp stretch moves the 2015+ band.
TEXT_WINDOW_TOP = 1080
TEXT_SCALES = (1.8, 1.85, 1.9, 1.95, 2.0, 2.05,
               2.1, 2.15, 2.2, 2.25, 2.3)
TEXT_ANGLES = (-2.0, -1.0, 0.0, 1.0, 2.0)
# Left/right BORDER_REPLICATE padding, same edge-clipping rationale as
# BAND_WIN_PAD (the warped card can extend past the photo edge).
TEXT_WIN_PAD = 300
# Weight of the text evidence added to the per-set score (same scale as the
# band evidence).
TEXT_WEIGHT = 1.0
# The tiebreak runs only when the top-2 margin of the symbol+COARSE-band
# scores is below this: near-ties are where the per-set text decides it,
# while high-margin cards (strong symbol match or 2021+ band resolution)
# are left untouched.  The gate is evaluated on the coarse scores (not the
# mode-dependent full-band scores) so that staged and legacy mode fire the
# same gate and apply the same offsets — see disambiguate().
TEXT_GATE_MARGIN = 0.10
# Top-N candidates (by symbol+coarse score) that receive the text evidence.
# Widened from 6 to 18: the true set of a fixable near-tie can rank as low as
# ~10-17 by symbol+coarse (its corner-symbol template is weak/corrupted, or a
# sibling set's symbol is stronger) and needs the text offset to win (e.g.
# c15 / gk2 ranking 10-17).  NOTE a smaller limit is NOT a free latency win:
# on the 260-card GT replay the COARSE-BAND pass over all ref-ready
# candidates dominates the per-near-tie cost, so cutting the limit 18->8
# saved only ~11% (9.72s -> 8.66s/card) while LOSING 2 fixes (the c15 / gk2
# cards above fall outside the top-8 and go unresolved).  18 is kept.
TEXT_CANDIDATE_LIMIT = 18
# Identical-reference-text guard threshold (no-false-match guard).  When the
# text offset is the SOLE reason a near-tie resolved (the symbol+band margin
# without text is below AMBIGUOUS_MARGIN) AND the top-2 candidate sets'
# reference bottom-text regions are MORE similar than this (refref_text_sim >
# threshold), the printed text is not a reliable discriminator — the two sets
# print nearly-identical set code / collector / copyright (a promo and its
# base set, or same-year same-artist reprints) — and the card is forced to
# 'ambiguous' so the user reviews it instead of receiving a silent false
# match.  The guard only ever downgrades resolved->ambiguous (the winner is
# unchanged), so it cannot break a correct pick.  Calibrated on known
# promo/base (high) vs different-set (low) reference pairs; see
# output/_match_probe/calibrate_refref.py.  NOTE the measure is
# layout-dominated (whole-region CCOEFF), so the gap is not clean; 0.72 sits
# between the different-set pairs (<=0.699) and the identical-text pairs
# (>=0.740) in the calibration set.
IDENTICAL_TEXT_THRESHOLD = 0.72
# Same-print guard threshold.  When the top-2 candidate sets' reference
# bottom-text regions are MORE similar than this, the two sets are the SAME
# physical print (a promo and its base set sharing the exact same collector
# number / artist / copyright, or two sets that render identically in the
# band).  Such a pair is indistinguishable by ANY CV signal — corner symbol,
# set-name band and printed text are all layout-dominated and near-equal —
# so a confident resolution between them is a false match no matter what
# drove the margin (symbol+band or text).  The guard forces 'ambiguous'
# regardless of the margin driver.  0.95 sits in the clean gap between
# same-physical-print pairs (measured ~0.988-0.989) and every different-set
# or same-set-different-card pair (measured <= 0.891).  This is the surgical
# catch for the psoi/soi-style false match, where the margin is
# symbol+band-driven so the text-sole-reason condition (IDENTICAL_TEXT_*)
# above does not fire on its own.
SAME_PRINT_THRESHOLD = 0.95
# Margin cap for the identical-text guard.  The guard only ever applies to
# near-ties whose post-text margin is at most this value.  A confidently
# resolved card (margin > cap) is left alone: its corner-symbol + band
# evidence strongly distinguishes the top-2 sets, so they are NOT the same
# print and a high refref_text_sim there is a layout-dominated false
# positive (forcing it 'ambiguous' would over-flag a correct, confident
# pick).  Calibrated to catch the psoi/soi silent false match (measured
# margin 0.215) while leaving confidently-resolved cards untouched; on the
# 260-card GT replay this drops guard-forced over-flags from 70 (uncapped)
# to the ~44 genuine near-ties.
GUARD_MARGIN_CAP = 0.25

__all__ = [
    'REF_W', 'REF_H', 'PHOTO_W', 'PHOTO_H',
    'VOTE_REGION', 'ALIGN_WIN', 'ZONE', 'SYMBOL_CENTER', 'SCALES', 'MASK_THR',
    'NO_TEMPLATE_SETS', 'A_ROI', 'B_SPACE', 'B_ROI', 'PLST_LOGO_TH',
    'AMBIGUOUS_MARGIN',
    'BAND_REF_Y', 'BAND_REF_X', 'BAND_WINDOW_TOP', 'BAND_SCALES',
    'BAND_ANGLES', 'BAND_WIN_PAD',
    'BAND_WEIGHT', 'BAND_FINE_LIMIT', 'band_score',
    'TEXT_L_BOX', 'TEXT_R_BOX', 'TEXT_WINDOW_TOP', 'TEXT_SCALES',
    'TEXT_ANGLES', 'TEXT_WIN_PAD', 'TEXT_WEIGHT', 'TEXT_GATE_MARGIN',
    'TEXT_CANDIDATE_LIMIT', 'text_score',
    'IDENTICAL_TEXT_THRESHOLD', 'refref_text_sim',
    'SAME_PRINT_THRESHOLD', 'GUARD_MARGIN_CAP',
    'load_ref', 'extract_template_multi', 'build_templates',
    'symbol_zone', 'binarize', 'score_candidate_sets', 'disambiguate',
    'logo_score_ref', 'logo_score_photo', 'has_plst_logo', 'make_photo',
    'stable_true', 'ALL_GROUPS_SQL',
]


# ---------------------------------------------------------------------------
# Template extraction (ported verbatim from prototype/eval_within_group.py)
# ---------------------------------------------------------------------------

def load_ref(ref):
    """Load a reference image as grayscale.

    Accepts a file path (str/Path, e.g. ``cache/scryfall_images/<scryfall_id>.jpg``)
    or an already-loaded grayscale ``np.ndarray``.  Returns None if the image
    cannot be read.
    """
    if isinstance(ref, np.ndarray):
        return ref
    if isinstance(ref, (str, os.PathLike)):
        img = cv2.imread(os.fspath(ref), cv2.IMREAD_GRAYSCALE)
        return None if img is None else img
    raise TypeError(f'load_ref expects a path or np.ndarray, got {type(ref)!r}')


def pick_blob(binary, x0, y0):
    """Best symbol blob in a binary region. Returns (bx0,by0,bx1,by1) globals or None."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    best = None
    best_d = 1e9
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 120 or w * h < 200 or w < 8 or h < 8:
            continue
        if w > 64 or h > 50:
            continue
        if area / (w * h) > 0.78:   # dense promo layouts (p05 f0.79, lea f0.70...) rejected
            continue
        cx, cy = x0 + x + w / 2.0, y0 + y + h / 2.0
        d = ((cx - SYMBOL_CENTER[0]) ** 2 + (cy - SYMBOL_CENTER[1]) ** 2) ** 0.5
        if d < best_d:
            best_d = d
            best = (x0 + x, y0 + y, x0 + x + w, y0 + y + h)
    return best


def _trim_frame_rows(mask, cy0, cx0):
    """Remove frame rows/columns shared by every set of the same frame
    generation (measured on 2021+, 2019, 2003 frames):
      - global y<=386: decor band below the art (partial-width fragments)
      - global y>=415: rule-box top line (full width; badges end by y416)
      - global x>=449 tall columns: 2003-frame vertical bar
    """
    m = mask.copy()
    w, h = m.shape[1], m.shape[0]
    for r in range(h):
        gy = cy0 + r
        if gy <= 386 or gy >= 415:
            m[r] = 0
    for c in range(w):
        gx = cx0 + c
        if gx >= 449 and int(m[:, c].sum()) >= 0.5 * h:
            m[:, c] = 0
    return m


def _aligned_region(img, dy, dx, box):
    """Crop box from img, compensating a card offset (dy,dx).
    (dy,dx) = card content position relative to nominal; vacated border=white."""
    x0, y0, x1, y1 = box
    H, W = y1 - y0, x1 - x0
    out = np.full((H, W), 255, np.uint8)
    sy, ty = max(0, dy), max(0, -dy)
    sx, tx = max(0, dx), max(0, -dx)
    out[ty:ty + H - abs(dy), tx:tx + W - abs(dx)] = \
        img[y0 + sy:y0 + sy + H - abs(dy), x0 + sx:x0 + sx + W - abs(dx)]
    return out


def extract_template_multi(sids):
    """Aligned majority-vote template over up to 16 cards of the set.
    The set symbol is constant; art/text vary. Reference cards sit at
    slightly different positions in the 488x680 frame (+/-7px), so each
    card is first aligned to the set mean card by full-image normalized
    cross-correlation, then regions are voted. Extended-art and overall-
    dark (promo) cards are excluded first.
    Returns (template_gray, mask) or None.

    Known limitation: sets whose printed symbol VARIES PER CARD (guild-sigil
    sets such as gk2 / prw2, where each card shows the sigil of its guild)
    violate the "constant symbol" premise — the vote mixes several different
    glyphs and the result is a small diagonal fragment (~17-18 px wide)
    instead of the full upright sigil (~32-33 px). Matching such templates
    against real photos scores low (0.05-0.10). ``disambiguate()``
    compensates with the bottom set-name band evidence (``band_score()``),
    which matches each candidate's own reference card."""
    x0, y0, x1, y1 = VOTE_REGION
    cands = []
    for sid in sids:
        img = load_ref(sid)
        if img is None:
            continue
        reg_full = img[y0:y1, x0:x1]
        art_in = float((img[384:392, 100:400] < MASK_THR).mean())  # extended art
        dark = float((reg_full < MASK_THR).mean())
        if art_in > 0.35 or dark > 0.60:
            continue
        cands.append((dark, img))
    # keep the 16 cleanest (least ink in the region -> least text/background)
    cands.sort(key=lambda t: t[0])
    cards = [img for _, img in cands[:16]]
    if len(cards) < 4:
        return None
    # Sharp template = per-pixel median (robust to per-card position spread;
    # a plain mean blurs all edges and flattens the correlation peak).
    median_img = np.median(np.stack(cards, axis=0), axis=0).astype(np.float32)
    # 3x3 dilation: keeps thin strokes alive under the +/-0.5px per-card
    # jitter (median vote over <=16 cards).
    kernel = np.ones((3, 3), np.uint8)
    regions = []
    for img in cards:
        patch = np.pad(img, ALIGN_WIN, mode='constant', constant_values=0).astype(np.float32)
        res = cv2.matchTemplate(patch, median_img, cv2.TM_CCOEFF_NORMED)
        idx = int(np.argmax(res))
        py, px = divmod(idx, res.shape[1])
        dy, dx = py - ALIGN_WIN, px - ALIGN_WIN
        reg = (_aligned_region(img, dy, dx, VOTE_REGION) < MASK_THR).astype(np.uint8)
        regions.append(cv2.dilate(reg, kernel))
    frac = np.mean(np.stack(regions), axis=0)
    # Higher thresholds keep only constant structures (symbol) and drop the
    # varying type-line text (frac ~0.4-0.6). Try from strict to loose.
    for T in (0.9, 0.8, 0.7, 0.6, 0.5):
        mask_region = (frac >= T).astype(np.uint8)
        box = pick_blob(mask_region, x0, y0)
        if box is None:
            continue
        m = 3
        bx0, by0, bx1, by1 = box
        cx0, cy0 = max(x0, bx0 - m), max(y0, by0 - m)
        cx1, cy1 = min(x1, bx1 + m), min(y1, by1 + m)
        crop_frac = frac[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
        mask = _trim_frame_rows((crop_frac >= T).astype(np.uint8), cy0, cx0)
        # Drop constant structures that are not part of the symbol (e.g. a
        # type-line character shared by the sampled cards): keep only mask
        # components located inside the picked blob bbox (+-2px). Inner marks
        # of the symbol stay (they are inside the blob bbox).
        ncomp, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        lb0, lb1 = bx0 - cx0, bx1 - cx0
        lt0, lt1 = by0 - cy0, by1 - cy0
        keep = np.zeros_like(mask)
        for lbl in range(1, ncomp):
            cx_l, cy_l, cw, ch, _a = stats[lbl]
            if cx_l >= lb0 - 2 and cx_l + cw <= lb1 + 2 and \
               cy_l >= lt0 - 2 and cy_l + ch <= lt1 + 2:
                keep[labels == lbl] = 1
        mask = keep
        if int(mask.sum()) < 120:
            continue
        # Re-crop to the (trimmed) mask ink bbox: empty frame-line rows must
        # not bloat the template, or it becomes larger than the query zone.
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        r0, r1 = max(0, rows[0] - 2), min(mask.shape[0], rows[-1] + 3)
        c0, c1 = max(0, cols[0] - 2), min(mask.shape[1], cols[-1] + 3)
        sub = mask[r0:r1, c0:c1]
        maxw = ZONE[2] - ZONE[0] - 2
        maxh = ZONE[3] - ZONE[1] - 2
        if sub.shape[1] > maxw or sub.shape[0] > maxh:
            continue  # still too large (frame element inside); try lower T
        crop = (crop_frac[r0:r1, c0:c1] * 255).astype(np.uint8)
        return crop, sub
    return None


def build_templates(sets):
    """Build set templates: ``{set_code: (template_gray, mask)}``.

    ``sets`` is an iterable of ``(set_code, sids)`` pairs where ``sids`` is an
    iterable of up to ~60 reference images (file paths or gray ndarrays,
    typically ``cache/scryfall_images/<scryfall_id>.jpg``).  Sets whose symbol
    cannot be extracted (``extract_template_multi`` -> None) and the
    non-templatable sets in ``NO_TEMPLATE_SETS`` are skipped, mirroring
    run7's build_templates + _drop_no_template.  No database access: the
    caller (CardDatabase) supplies the sids.
    """
    templates = {}
    for sc, sids in sets:
        if sc in NO_TEMPLATE_SETS:
            continue
        sids = list(sids)
        if not sids:
            continue
        tmpl = extract_template_multi(sids)
        if tmpl is not None:
            templates[sc] = tmpl
    return templates


# ---------------------------------------------------------------------------
# Scoring (ported verbatim from prototype/eval_within_group.py /
# prototype/full_sweep.py)
# ---------------------------------------------------------------------------

def match_score(query_bin, tmpl_mask, scale=1.0):
    """Max over shifts of containment minus a penalty for query ink inside
    the template bbox that is NOT covered by the template. The penalty
    separates symbols sharing the same outline (same bars, same position)
    but differing in inner strokes: at the true position the query's symbol
    ink is mostly covered, at a wrong symbol position much of it is not.
    Uncovered text/other ink elsewhere in the zone does not matter.
    Returns (score, (dx, dy))."""
    th, tw = tmpl_mask.shape
    nh, nw = query_bin.shape
    t_area = float(tmpl_mask.sum())
    if t_area < 10:
        return 0.0, (0, 0)
    if th >= nh or tw >= nw:
        return 0.0, (0, 0)  # template larger than zone: cannot match
    qf = query_bin.astype(np.float32)
    tf = tmpl_mask.astype(np.float32)
    ones = np.ones((th, tw), np.float32)
    # both filter2D with the same kernel size -> identical anchor convention
    corr = cv2.filter2D(qf, -1, tf, borderType=cv2.BORDER_CONSTANT)
    boxsum = cv2.filter2D(qf, -1, ones, borderType=cv2.BORDER_CONSTANT)
    sl = (slice(th // 2, nh - th // 2 + 1), slice(tw // 2, nw - tw // 2 + 1))
    corr = corr[sl]
    boxsum = boxsum[sl]
    LAMBDA = 1.2
    scoremap = corr - LAMBDA * np.maximum(0.0, boxsum - corr)
    ys, xs = np.unravel_index(int(np.argmax(scoremap)), scoremap.shape)
    inter = float(corr[ys, xs])
    extra = max(0.0, float(boxsum[ys, xs]) - inter)
    return (inter - LAMBDA * extra) / t_area, (xs, ys)


_res_cache = {}


def _resized(tmpl, tmask, scale_factor, s):
    key = (id(tmpl), s, scale_factor)
    r = _res_cache.get(key)
    if r is None:
        ts = scale_factor * s
        t = cv2.resize(tmpl, None, fx=ts, fy=ts, interpolation=cv2.INTER_AREA)
        tm = cv2.resize((tmask * 255).astype(np.uint8), None, fx=ts, fy=ts,
                        interpolation=cv2.INTER_AREA)
        r = (t, (tm > 100).astype(np.uint8))
        if len(_res_cache) > 40000:
            _res_cache.clear()
        _res_cache[key] = r
    return r


def symbol_zone(card_bgr):
    """Crop the set-symbol zone from a runtime card photo.

    ``card_bgr`` is the warped portrait card (~1000x1400, BGR or gray).  The
    reference-space ZONE is scaled to photo space and cropped (identical
    expression to the prototype's photo-mode zone).  Returns the grayscale
    (non-binarized) crop; pass it through ``binarize`` for scoring.
    """
    if card_bgr.ndim == 2:
        gray = card_bgr
    else:
        gray = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2GRAY)
    return gray[int(ZONE[1] * PHOTO_H / REF_H):int(ZONE[3] * PHOTO_H / REF_H),
                int(ZONE[0] * PHOTO_W / REF_W):int(ZONE[2] * PHOTO_W / REF_W)]


def binarize(img, thr=MASK_THR):
    """Binarize a gray image: 1 == ink (dark)."""
    return (img < thr).astype(np.uint8)


def score_candidate_sets(zone_bin, scale_factor, candidate_sets, templates):
    """Score candidate sets against a binarized symbol zone (run7 sweep).

    ``zone_bin`` is a binary mask at the input card's scale (photo space for
    ``symbol_zone`` output, reference space for 488x680 images);
    ``scale_factor`` is the zone's scale relative to reference space
    (``PHOTO_W / REF_W`` for photos, ``1.0`` for reference-scale images).
    Each candidate set present in ``templates`` is scored with the run7
    SCALES sweep (template enlarged by ``scale_factor * s``).  Returns
    ``{set_code: score}`` with scores in ``[0, 1]`` (1 == perfect).
    """
    scores = {}
    for sc in candidate_sets:
        if sc not in templates:
            continue
        tmpl, tmask = templates[sc]
        best = 0.0
        for s in SCALES:
            t, tm = _resized(tmpl, tmask, scale_factor, s)
            sc_i, _ = match_score(zone_bin, tm, s)
            best = max(best, sc_i)
        scores[sc] = best
    return scores


def band_score(photo_gray, ref_gray, coarse_only=False):
    """Score how well ``ref_gray``'s bottom set-name band matches the bottom
    region of a warped 1000x1400 photo.

    ``photo_gray``  grayscale warped card (PHOTO_W x PHOTO_H)
    ``ref_gray``    grayscale 488x680 reference image of a card from the set
    ``coarse_only`` when True, only the pass-1 coarse grid is searched and the
                    pass-2 fine refinement around the coarse best is skipped
                    (≈7× faster per candidate).  Used by the staged protocol
                    in ``disambiguate`` for the non-finalist candidates.
                    ``False`` (default) = unchanged full two-pass behavior.

    Two-pass search, identical protocol for every candidate:
      pass 1 (coarse): the band strip (reference rows BAND_REF_Y, columns
                       BAND_REF_X) is resized across BAND_SCALES, rotated
                       across BAND_ANGLES, and matched against the photo's
                       bottom window (rows BAND_WINDOW_TOP..PHOTO_H) with
                       TM_CCOEFF_NORMED.
      pass 2 (fine): a refinement grid around the coarse best — scale in
                     best_s +/- 0.15 (step 0.02), angle in best_a +/- 2.5
                     (step 0.5).  The coarse grid undershoots photos whose
                     warp magnification/tilt falls between grid points (e.g.
                     the c18-300 swamp photo, optimum s~2.17 / a~-0.5).
    The best score over both passes (or the coarse pass alone when
    ``coarse_only``) is returned, floored at 0 (the band never penalizes a
    candidate).  Position independent: the warped band row
    varies with the photo's perspective, so the match is searched over the
    whole window rather than a fixed crop.  The window is padded left/right
    (BAND_WIN_PAD) so the template may be placed partly outside the photo
    when the warped card is clipped at the image edge.
    """
    win = photo_gray[BAND_WINDOW_TOP:PHOTO_H, :]
    if BAND_WIN_PAD:
        win = cv2.copyMakeBorder(
            win, 0, 0, BAND_WIN_PAD, BAND_WIN_PAD, cv2.BORDER_REPLICATE)
    band = ref_gray[BAND_REF_Y[0]:BAND_REF_Y[1], BAND_REF_X[0]:BAND_REF_X[1]]
    best, best_s, best_a = -1.0, None, None

    def _search(scales, angles):
        nonlocal best, best_s, best_a
        for ang in angles:
            for s in scales:
                t = cv2.resize(band, None, fx=s, fy=s)
                if t.shape[1] > win.shape[1] - 2:
                    t = cv2.resize(
                        t,
                        (win.shape[1] - 2,
                         max(1, int(round(t.shape[0] * (win.shape[1] - 2) / t.shape[1])))))
                if t.shape[0] >= win.shape[0]:
                    continue
                if ang:
                    M = cv2.getRotationMatrix2D(
                        (t.shape[1] / 2, t.shape[0] / 2), ang, 1.0)
                    t = cv2.warpAffine(t, M, (t.shape[1], t.shape[0]),
                                       borderMode=cv2.BORDER_REPLICATE)
                v = float(cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED).max())
                if v > best:
                    best, best_s, best_a = v, s, ang

    _search(BAND_SCALES, BAND_ANGLES)          # pass 1: coarse grid
    if not coarse_only and best_s is not None:
        fine_s = np.arange(best_s - 0.15, best_s + 0.151, 0.02)
        fine_a = np.arange(best_a - 2.5, best_a + 2.51, 0.5)
        _search(list(np.round(fine_s, 3)), list(np.round(fine_a, 3)))  # pass 2
    return max(0.0, best)


def text_score(photo_gray, ref_gray):
    """Bottom text-region tiebreak score (set-code line + copyright line).

    ``photo_gray``  grayscale warped card (PHOTO_W x PHOTO_H)
    ``ref_gray``    grayscale 488x680 reference image of a card from the set

    For each of the two per-set text regions — TEXT_L_BOX (collector number +
    set-code line, "SET • LANG") and TEXT_R_BOX (copyright line) — the
    reference strip is resized across TEXT_SCALES, rotated across TEXT_ANGLES
    and matched against the photo's bottom window (rows
    TEXT_WINDOW_TOP..PHOTO_H, left/right padded by TEXT_WIN_PAD) with
    TM_CCOEFF_NORMED; the per-region bests are summed and returned.  Unlike
    the band score there is no 0 floor and no fine pass: the regions are
    small (a single coarse grid covers them) and a candidate whose reference
    text is absent from the photo should score below one whose is present,
    relative to its rivals.  On 2015+ frames the set code + collector number
    and the copyright year differ per set; on 2003 frames the copyright
    range and collector total do.  Returns a float (typically -0.5..2.0,
    1+ per region when both match well).
    """
    win = photo_gray[TEXT_WINDOW_TOP:PHOTO_H, :]
    if TEXT_WIN_PAD:
        win = cv2.copyMakeBorder(
            win, 0, 0, TEXT_WIN_PAD, TEXT_WIN_PAD, cv2.BORDER_REPLICATE)
    total = 0.0
    for box in (TEXT_L_BOX, TEXT_R_BOX):
        strip = ref_gray[box[1]:box[3], box[0]:box[2]]
        best = -1.0
        for ang in TEXT_ANGLES:
            for s in TEXT_SCALES:
                t = cv2.resize(strip, None, fx=s, fy=s)
                if t.shape[1] > win.shape[1] - 2:
                    t = cv2.resize(
                        t,
                        (win.shape[1] - 2,
                         max(1, int(round(t.shape[0] * (win.shape[1] - 2) / t.shape[1])))))
                if t.shape[0] >= win.shape[0]:
                    continue
                if ang:
                    M = cv2.getRotationMatrix2D(
                        (t.shape[1] / 2, t.shape[0] / 2), ang, 1.0)
                    t = cv2.warpAffine(t, M, (t.shape[1], t.shape[0]),
                                       borderMode=cv2.BORDER_REPLICATE)
                v = float(cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED).max())
                if v > best:
                    best = v
        total += best
    return total


def refref_text_sim(refA_gray, refB_gray):
    """Similarity in [-1,1] between two reference cards' bottom-text regions
    (TEXT_L_BOX + TEXT_R_BOX).  High when the two sets print nearly-identical
    band text (a promo and its base set share the same printed set code,
    collector number and copyright year); low when they print different text.
    Both inputs are 488x680 grayscale.  Per box the patch is zero-meaned and
    the normalized dot product (CCOEFF) is the similarity; the max over the
    two boxes is returned (0.0 when both patches are flat).  Used only by the
    identical-reference-text guard in disambiguate().  The measure is
    layout-dominated (the frame structure fills most of the region), so it is
    a "do these look alike" proxy, not an exact "same set code" test — the
    guard biases to forcing ambiguous on high values (a no-false-match
    priority: over-flagging a review is far cheaper than a silent wrong set).
    """
    sims = []
    for box in (TEXT_L_BOX, TEXT_R_BOX):
        x0, y0, x1, y1 = box
        a = refA_gray[y0:y1, x0:x1].astype(np.float32)
        b = refB_gray[y0:y1, x0:x1].astype(np.float32)
        a = a - a.mean()
        b = b - b.mean()
        denom = float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
        if denom > 0:
            sims.append(float((a * b).sum() / denom))
    return max(sims) if sims else 0.0


def disambiguate(card_bgr, candidate_sets, templates, logo_tpl, ref_imgs=None,
                 band_progress=None, staged_band=True):
    """Full production disambiguation for one photo.

    ``card_bgr``      warped portrait card (~1000x1400, BGR or gray)
    ``candidate_sets`` set codes sharing the artwork (iterable or mapping)
    ``templates``      {set_code: (template_gray, mask)} (plst is not templatable)
    ``logo_tpl``       plst corner-logo template (from CardDatabase, 13x14)
    ``ref_imgs``       optional {set_code: 488x680 reference image (BGR or
                       gray)}.  When given, every candidate with a reference
                       image also gets the bottom set-name band evidence:
                       score += BAND_WEIGHT * max(0, band_score(photo, that
                       set's reference image)).  The band is an independent
                       per-set signal (the set name printed on the band),
                       matched against the candidate's own reference card, so
                       it reinforces the true set; it is the rescue signal for
                       sets whose corner symbols are identical (per-card guild
                       sigils, e.g. gk2 vs prw2) and whose production symbol
                       templates are corrupted majority-vote fragments.
                       Flooring at 0 means the band never penalizes a
                       candidate.

                       After the COARSE band evidence and BEFORE the fine
                       pass (the gate is mode-independent by construction),
                       near-ties (top-2 margin of the symbol+coarse scores
                       below TEXT_GATE_MARGIN) additionally receive the
                       bottom text-region tiebreak: the top
                       TEXT_CANDIDATE_LIMIT candidates by symbol+coarse score
                       each get score += TEXT_WEIGHT * text_score(photo, that
                       set's reference image) — the printed set-code/
                       collector line and copyright line matched as image
                       templates.  The offsets are re-applied on top of the
                       full-band score of the fine-pass candidates.  This is
                       the per-set signal on 2003/2015/2019 frames (no set
                       name band) and on reprints whose set-name band is
                       identical to a sibling set's.  High-margin cards skip
                       the stage entirely.
    ``band_progress``  optional ``Callable[[int, int, str], None]``, called
                       immediately before each candidate's band score:
                       ``band_progress(k, n, set_code)`` where ``n`` is the
                       number of candidates with a cached reference image and
                       ``k`` = 1..``n`` in candidate order.  ``None`` (default)
                       = no reporting.  With ``staged_band=True`` the
                       finalists are reported a second time (stage 2) with a
                       VIRTUAL index ``k = n + j`` (``j`` = 1..number of
                       finalists, in fine-pass order), so a caller mapping
                       ``k`` -> fraction can forward-map those reports: the
                       reported sequence is non-decreasing and the bar never
                       steps back.  Legacy mode (``staged_band=False``) always
                       reports ``k`` <= ``n``.
    ``staged_band``    True (default): two-stage band protocol — pass-1
                       (coarse) band for ALL reference-ready candidates, then
                       the pass-2 fine refinement only for the top
                       ``BAND_FINE_LIMIT`` candidates by current
                       (symbol+coarse+text) score.  Winner/state/margin are
                       bit-identical to the legacy protocol whenever the
                       coarse and full rankings agree on that top pair
                       (checked by the Gate 2 parity gate, not assumed).
                       False: legacy protocol — full (coarse+fine) band for
                       every candidate (retained for parity checks and
                       instant A/B).  Known difference with True: the
                       returned ``scores`` for non-finalists are coarse-based
                       (≤ their full values); the top-2 are exact (plan §4.3).

    Residual edge case: on old/non-standard frames where every candidate's
    symbol score is 0.0 (dead templates, e.g. pcy Planechase 2012), the band
    is the only evidence and may not separate candidates cleanly (observed
    pcy vs clu at margin 0.043 on a simulated photo). Such frames are not
    expected to reach this path in production candidate formation (candidates
    come from same-art/name DB matches), so the close ranking is accepted
    rather than tuned for at the cost of the gk2/prw2 rescue.

    Returns ``(winner, scores, logo_score, state)``:
      winner      winning set code (top scorer; None if nothing scored)
      scores      {set_code: score}  (plst may carry 1.0 + logo_score via
                  boost; band evidence, when given, is folded in before
                  ranking)
      logo_score  B-mode plst logo score in [-1, 1] (0.0 if plst not a candidate)
      state       'resolved' | 'ambiguous' | 'unverified'
    """
    if card_bgr.ndim == 2:
        gray = card_bgr
    else:
        gray = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2GRAY)

    zone_bin = binarize(symbol_zone(card_bgr))
    scores = score_candidate_sets(zone_bin, PHOTO_W / REF_W,
                                  candidate_sets, templates)

    logo_score = 0.0
    if 'plst' in candidate_sets:
        logo_score = logo_score_photo(gray, logo_tpl)
        if logo_score >= PLST_LOGO_TH:
            scores['plst'] = 1.0 + logo_score

    # Bottom set-name band evidence: added to every candidate that has a
    # reference image.  It is independent of the corner-symbol templates and
    # matched against the candidate's own reference card, so the true set's
    # band always matches best; it rescues same-sigil sets (gk2 vs prw2) whose
    # symbol templates are corrupted majority-vote fragments.
    refs_gray = {}
    text_off = {}   # function-scope default so the identical-text guard can
                    # read it even when ref_imgs is None (stays empty -> no-op)
    if ref_imgs:
        # Precompute the reference-ready items (set code + prepared gray ref)
        # so n is known before the scoring loop (band_progress contract).
        band_items = []
        for sc in candidate_sets:
            ref = ref_imgs.get(sc)
            if ref is None:
                continue
            rg = ref if ref.ndim == 2 else cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
            if rg.shape != (REF_H, REF_W):
                rg = cv2.resize(rg, (REF_W, REF_H))
            band_items.append((sc, rg))
        with_ref = [sc for sc, _ in band_items]   # candidate order unchanged
        refs_gray = dict(band_items)
        base = dict(scores)  # symbol (+plst boost) scores, before any band
        # Stage 1 (BOTH modes): coarse band for ALL reference-ready
        # candidates.  The coarse scores are the mode-independent ranking:
        # the text tiebreak below gates on them and the staged protocol
        # picks its fine-pass finalists from them, so staged and legacy
        # apply the identical text offsets to the identical candidate subset
        # — the staged-vs-legacy difference stays the fine pass only (the
        # wide staged-vs-legacy gate's envelope).
        for k, sc in enumerate(with_ref, 1):
            if band_progress is not None:
                band_progress(k, len(with_ref), sc)
            scores[sc] = (base.get(sc, 0.0)
                          + BAND_WEIGHT * max(
                              0.0, band_score(gray, refs_gray[sc],
                                              coarse_only=True)))
        # Bottom text-region tiebreak (set-code/collector line + copyright
        # line), applied BETWEEN the coarse and the fine band pass: near-
        # ties only (top-2 margin of the symbol+coarse scores below
        # TEXT_GATE_MARGIN).  The top TEXT_CANDIDATE_LIMIT reference-ready
        # candidates each get the per-set printed text evidence (text_score).
        # High-margin cards never reach this stage, so confident picks
        # (strong symbol matches, 2021+ band resolutions, plst logo boosts)
        # are untouched.  Gating on the COARSE scores (mode-independent by
        # construction) rather than the mode-dependent full-band scores is
        # essential: staged and legacy must fire the same gate on the same
        # top-N subset, or the tiebreak itself becomes a spurious staged-vs-
        # legacy winner/state diverger (the pre-fix wide run showed exactly
        # that: a production-mode winner flip dc_01#5 ddj-vs-rtr).
        text_off = {}
        if len(scores) >= 2:
            ranked_pre = sorted(scores.items(), key=lambda kv: -kv[1])
            if ranked_pre[0][1] - ranked_pre[1][1] < TEXT_GATE_MARGIN:
                for sc, _ in ranked_pre[:TEXT_CANDIDATE_LIMIT]:
                    if sc in refs_gray:
                        text_off[sc] = TEXT_WEIGHT * text_score(
                            gray, refs_gray[sc])
                        scores[sc] += text_off[sc]
        if staged_band:
            # Stage 2: full (coarse+fine) band for the top BAND_FINE_LIMIT by
            # current (symbol+coarse+text) score.  Re-reports use a virtual
            # index (k = n + j, j = 1..m) so the caller's frac mapping can
            # forward-map stage 2: the bar never steps back (band_progress
            # contract).  Ranked among the reference-ready candidates (the
            # only ones that can receive a fine band); without the filter a
            # non-reference top-2 candidate would raise ValueError in
            # with_ref.index() and the whole disambiguation would degrade to
            # 'unverified'.
            withref = set(with_ref)
            ranked1 = sorted((kv for kv in scores.items() if kv[0] in withref),
                             key=lambda kv: -kv[1])
            nref = len(with_ref)
            for j, (sc, _) in enumerate(ranked1[:BAND_FINE_LIMIT], 1):
                if band_progress is not None:
                    band_progress(nref + j, nref, sc)  # virtual: k > n
                scores[sc] = (base.get(sc, 0.0)
                              + BAND_WEIGHT * max(0.0,
                                                  band_score(gray,
                                                             refs_gray[sc]))
                              + text_off.get(sc, 0.0))
        else:
            # Legacy: full (coarse+fine) band for every candidate, re-
            # applying the mode-independent text offsets.  band_score re-runs
            # its own coarse grid internally (the coarse scores above were
            # needed for the mode-independent text gate; legacy mode is
            # verification-only, so the double work is accepted).  No
            # band_progress here: stage 1 already reported 1..n and a second
            # sweep would step the bar back.
            for sc in with_ref:
                scores[sc] = (base.get(sc, 0.0)
                              + BAND_WEIGHT * max(0.0, band_score(gray,
                                                                 refs_gray[sc]))
                              + text_off.get(sc, 0.0))

    if not scores:
        return None, scores, logo_score, 'unverified'

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    winner = ranked[0][0]
    if len(ranked) == 1:
        state = 'resolved'
    else:
        margin_post = ranked[0][1] - ranked[1][1]
        second = ranked[1][0]
        state = 'resolved' if margin_post >= AMBIGUOUS_MARGIN else 'ambiguous'
        # FIX 3 identical-reference-text guard (no false match): if the text
        # offset is the SOLE reason this near-tie resolved (the symbol+band
        # margin WITHOUT text is below AMBIGUOUS_MARGIN) AND the top-2 sets
        # print nearly-identical band text (refref_text_sim above the
        # threshold), the text "separation" is not reliable evidence — force
        # 'ambiguous' so the user reviews it instead of receiving a silent
        # false match.  Only ever downgrades resolved->ambiguous (the winner
        # is unchanged), so it can never break a correct pick.  A no-op when
        # there are no reference images (refs_gray empty) or no text offset
        # was applied (text_off empty => margin_post == margin_no_text).
        # NO_TEMPLATE_SETS exclusion: plst (The List) is a digital-only set
        # whose reference image is the SAME illustration as the real set's
        # (refref_text_sim ~ 1.0 trivially, a reference compared to its own
        # identical twin).  That is not a same-physical-print ambiguity --
        # a physical card can never be plst -- so the guard must not fire on
        # a (real_set, plst) pair, or it over-flags a correct real-set pick.
        # Excluded by set membership (generic), not by per-card data.
        if (state == 'resolved' and margin_post <= GUARD_MARGIN_CAP
                and winner in refs_gray and second in refs_gray
                and winner not in NO_TEMPLATE_SETS
                and second not in NO_TEMPLATE_SETS):
            rsim = refref_text_sim(refs_gray[winner], refs_gray[second])
            # (a) spec mechanism: the text offset is the SOLE reason this
            #     near-tie resolved (symbol+band margin without text is below
            #     AMBIGUOUS_MARGIN) AND the two sets print similar text.
            text_sole = ((margin_post
                          - (text_off.get(winner, 0.0)
                             - text_off.get(second, 0.0)))
                         < AMBIGUOUS_MARGIN)
            # (b) same-print catch: the two sets' reference bottom-text
            #     regions are near-identical (the same physical print), so
            #     they are indistinguishable by ANY CV signal — force
            #     'ambiguous' regardless of what drove the margin.  Catches
            #     the psoi/soi false match where the margin is
            #     symbol+band-driven so (a) does not fire on its own.
            if rsim > SAME_PRINT_THRESHOLD or (
                    rsim > IDENTICAL_TEXT_THRESHOLD and text_sole):
                state = 'ambiguous'
    return winner, scores, logo_score, state


# ---------------------------------------------------------------------------
# plst (The List) B-mode corner-logo detector (ported verbatim from
# prototype/plst_logo_detector.py; the logo template is passed in instead of
# the module-level TPL/TPL_B globals)
# ---------------------------------------------------------------------------

def _match(roi, tpl):
    if roi.shape[0] <= tpl.shape[0] or roi.shape[1] <= tpl.shape[1]:
        return -1.0
    res = cv2.matchTemplate(roi.astype(np.float32), tpl, cv2.TM_CCOEFF_NORMED)
    if np.isnan(res).any():
        res = np.nan_to_num(res, nan=-1.0)
    return float(res.max())


def _tpl_b(logo_tpl):
    """B-mode logo template (13x14 -> 11x12), same derivation as the
    prototype's module-level TPL_B."""
    tpl = logo_tpl.astype(np.uint8)
    return cv2.resize(
        tpl,
        (int(round(tpl.shape[1] * _B_SCALE)), int(round(tpl.shape[0] * _B_SCALE))),
        interpolation=cv2.INTER_AREA).astype(np.float32)


def logo_score_ref(ref, logo_tpl):
    """A mode: score of the 488x680 clean reference."""
    y0, y1, x0, x1 = A_ROI
    return _match(ref[y0:y1, x0:x1], logo_tpl.astype(np.float32))


def logo_score_photo(photo, logo_tpl):
    """B mode: score of a 1000x1400 photo."""
    w, h = B_SPACE
    img = cv2.resize(photo, (w, h), interpolation=cv2.INTER_AREA)
    y0, y1, x0, x1 = B_ROI
    return _match(img[y0:y1, x0:x1], _tpl_b(logo_tpl))


def has_plst_logo(card_bgr, logo_tpl, th=PLST_LOGO_TH):
    """True if a runtime 1000x1400 card photo (BGR or gray) shows The List's
    bottom-left corner logo."""
    if card_bgr.ndim == 2:
        gray = card_bgr
    else:
        gray = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2GRAY)
    return logo_score_photo(gray, logo_tpl) >= th


# ---------------------------------------------------------------------------
# Verification-harness support (ported verbatim)
# ---------------------------------------------------------------------------

def make_photo(gray488, seed):
    """Simulate a 1000x1400 card photo from a 488x680 reference image
    (downscale, JPEG q82, Gaussian noise, +/-2.5 deg rotation, upscale).
    Verification harness only; never used on the production path."""
    rng = random.Random(seed)
    h, w = gray488.shape
    target_w = 415
    target_h = int(h * target_w / w)
    small = cv2.resize(gray488, (target_w, target_h), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 82])
    small = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    noise = np.random.default_rng(seed).normal(0, 2.2, small.shape).astype(np.float32)
    small = np.clip(small.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    angle = rng.uniform(-2.5, 2.5)
    M = cv2.getRotationMatrix2D((small.shape[1] / 2, small.shape[0] / 2), angle, 1.0)
    small = cv2.warpAffine(small, M, (small.shape[1], small.shape[0]),
                           borderMode=cv2.BORDER_REPLICATE)
    return cv2.resize(small, (PHOTO_W, PHOTO_H), interpolation=cv2.INTER_CUBIC)


def stable_true(ill, codes):
    """Deterministic per-group 'printed set' choice, independent of sampling."""
    return random.Random(zlib.crc32(ill.encode('utf-8'))).choice(codes)


ALL_GROUPS_SQL = """SELECT illustration_id, GROUP_CONCAT(DISTINCT set_code) AS sets, COUNT(*) AS n
       FROM cards WHERE illustration_id IS NOT NULL AND illustration_id != ''
       GROUP BY illustration_id HAVING COUNT(DISTINCT set_code) >= 2"""
