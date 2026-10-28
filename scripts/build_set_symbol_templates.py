# Copyright (C) 2026 IgnorantGrimyYak
"""Build data/set_symbol_templates.pkl — per-set set-symbol templates (Phase 2).

Offline one-time builder for the set-disambiguation layer
(INTEGRATION_PLAN.md phase 2):

    data/scryfall_db.sqlite + cache/scryfall_images/  ->  data/set_symbol_templates.pkl

For every set in the database, up to 60 reference cards (ascending
scryfall_id, the same sample as the prototype) are read from
``cache/scryfall_images/<scryfall_id>.jpg`` as 488x680 grayscale (resized if a
file deviates), and the set symbol is extracted with the run7
majority-vote method — ``set_disambiguation.extract_template_multi``, a
verbatim port of ``prototype/eval_within_group.py``.  The result is written
to data/ as:

    (cache_key, {set_code: (template_gray, mask)})

where ``cache_key`` is the mtime of ``set_disambiguation.py`` — the same
invalidation scheme as the prototype cache (``prototype/full_sweep.py``
``get_templates``), so a runtime loader can treat a stale key as
"rebuild recommended".

Sets whose symbol cannot be extracted (no usable blob, no readable
reference image) and the non-templatable sets in
``set_disambiguation.NO_TEMPLATE_SETS`` (``plst``) are omitted from the
dict.  Re-run after the reference images or the extraction module change;
the app never rebuilds this file at startup (plan assumption A7).

Usage (from the project root or anywhere):
    .venv/Scripts/python.exe scripts/build_set_symbol_templates.py
"""

import pickle
import sqlite3
import sys
import time
from pathlib import Path

import cv2

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import set_disambiguation as sdis  # noqa: E402

DATABASE_PATH = PROJECT_DIR / "data" / "scryfall_db.sqlite"
IMAGE_DIR = PROJECT_DIR / "cache" / "scryfall_images"
MODULE_PATH = PROJECT_DIR / "set_disambiguation.py"
OUTPUT_PKL = PROJECT_DIR / "data" / "set_symbol_templates.pkl"

SAMPLE_LIMIT = 60  # max reference cards per set (prototype build_templates)
PROGRESS_EVERY = 100


def load_sample_image(path):
    """Load a cached reference image as 488x680 grayscale, resizing if the
    file deviates from the reference size. Returns None when unreadable."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if img.shape[:2] != (sdis.REF_H, sdis.REF_W):
        img = cv2.resize(img, (sdis.REF_W, sdis.REF_H),
                         interpolation=cv2.INTER_AREA)
    return img


def extract_all_templates(conn):
    """Extract symbol templates for every set.

    Returns (rows, templates, n_no_template, n_no_image, n_no_blob) where
    ``rows`` is the ``(set_code, MIN(scryfall_id))`` row per set and the
    ``n_*`` values classify each skip:
      n_no_template  set is in NO_TEMPLATE_SETS (plst: no corner symbol)
      n_no_image     no readable reference image in cache/scryfall_images/
      n_no_blob      images exist but no usable symbol blob was found
    """
    rows = conn.execute(
        "SELECT set_code, MIN(scryfall_id) FROM cards GROUP BY set_code"
    ).fetchall()
    templates = {}
    n_no_template = n_no_image = n_no_blob = 0
    t0 = time.time()
    for i, (set_code, _rep) in enumerate(rows, start=1):
        if set_code in sdis.NO_TEMPLATE_SETS:
            n_no_template += 1
            continue
        arrays = []
        for (sid,) in conn.execute(
                "SELECT scryfall_id FROM cards WHERE set_code=? "
                "ORDER BY scryfall_id LIMIT ?",
                (set_code, SAMPLE_LIMIT)):
            img = load_sample_image(IMAGE_DIR / (sid + ".jpg"))
            if img is not None:
                arrays.append(img)
        if not arrays:
            n_no_image += 1
        else:
            tmpl = sdis.extract_template_multi(arrays)
            if tmpl is None:
                n_no_blob += 1
            else:
                templates[set_code] = tmpl
        if i % PROGRESS_EVERY == 0:
            print(f"  {i}/{len(rows)} sets  ({time.time() - t0:.0f}s elapsed)",
                  flush=True)
    return rows, templates, n_no_template, n_no_image, n_no_blob


def group_coverage(conn, templates):
    """Count how many multi-set illustration groups the templates cover:
    total groups, groups with >=1 templatable candidate (scoreable), and
    groups where every candidate is templatable."""
    n_groups = n_covered = n_full = 0
    for _ill, sets_csv, _n in conn.execute(sdis.ALL_GROUPS_SQL):
        codes = sets_csv.split(",")
        n_groups += 1
        have = [c for c in codes if c in templates]
        if have:
            n_covered += 1
        if len(have) == len(codes):
            n_full += 1
    return n_groups, n_covered, n_full


def main():
    if not DATABASE_PATH.exists():
        sys.exit(f"error: missing database {DATABASE_PATH}")

    el = 0.0
    with sqlite3.connect(str(DATABASE_PATH)) as conn:
        t0 = time.time()
        rows, templates, n_no_template, n_no_image, n_no_blob = \
            extract_all_templates(conn)
        n_groups, n_covered, n_full = group_coverage(conn, templates)
        el = time.time() - t0

    cache_key = MODULE_PATH.stat().st_mtime
    with OUTPUT_PKL.open("wb") as f:
        pickle.dump((cache_key, templates), f)

    print(f"Built {len(templates)} set-symbol templates ({len(rows)} sets in DB)")
    print(f"  skipped, non-templatable (NO_TEMPLATE_SETS): {n_no_template}")
    print(f"  skipped, no usable reference image: {n_no_image}")
    print(f"  skipped, no usable symbol blob: {n_no_blob}")
    print(f"Multi-set illustration groups: {n_groups} "
          f"({n_covered} with >=1 templatable candidate, "
          f"{n_full} fully templatable)")
    print(f"Wrote {OUTPUT_PKL} "
          f"(cache key = set_disambiguation.py mtime {cache_key}) in {el:.0f}s")


if __name__ == "__main__":
    main()
