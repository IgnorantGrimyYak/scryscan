# Copyright (C) 2026 IgnorantGrimyYak
"""
Flask web app around extract_cards.py: upload photos of MTG cards, detect
and identify them, and view/download the results in a browser.
"""
import copy
import json
import os
import secrets
import shutil
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file, abort
from werkzeug.utils import secure_filename

import cv2

from extract_cards import (
    CardDatabase,
    IMAGE_EXTENSIONS,
    find_card_quads,
    process_image,
    render_debug_overlay,
    sort_boxes,
    write_csv_export,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(SCRIPT_DIR, "webapp_jobs")
# Automatic cleanup of webapp_jobs/: a job directory is deleted when its own
# mtime is older than JOB_TTL_HOURS. The dir mtime is set when uploads/ +
# output/ are created (writing files *into* the subdirectories does not move
# it), so it reliably equals job-creation time. The built-in sweep runs
# in-process (no cron): first ~60 s after startup, then every
# CLEANUP_INTERVAL_HOURS.
JOB_TTL_HOURS = 48
CLEANUP_INTERVAL_HOURS = 24
DB_PATH = os.path.join(SCRIPT_DIR, "data", "scryfall_db.sqlite")
MAX_CONTENT_LENGTH = 300 * 1024 * 1024  # 300 MB upload cap

os.makedirs(JOBS_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Loading the phash database is somewhat expensive; do it once at startup
# and share it (read-only) across requests/jobs.
# Set SCRYSCAN_DISABLE_SET_DISAMBIGUATION (any non-empty value) to force
# pHash-only matching without touching the template artifacts.
# Set SCRYSCAN_STAGED_BAND=0 to use the legacy full-band-for-all protocol;
# any other value (or unset) keeps the default staged protocol (coarse band
# for all candidates, fine refinement for the top BAND_FINE_LIMIT).
_card_db = (
    CardDatabase(
        DB_PATH,
        enable_disambiguation=not os.environ.get("SCRYSCAN_DISABLE_SET_DISAMBIGUATION"),
        staged_band=os.environ.get("SCRYSCAN_STAGED_BAND", "1") != "0",
    )
    if os.path.exists(DB_PATH)
    else None
)


def _ensure_name_indexes(db_path):
    """Create the indexes used by typeahead prefix searches (idempotent)."""
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_name ON cards (name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_name_set ON cards (name, set_code)")
        conn.commit()
        conn.close()
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not create name indexes: {exc}")


if _card_db:
    _ensure_name_indexes(_card_db.db_path)

_jobs = {}
_jobs_lock = threading.Lock()


def find_expired_jobs(jobs_dir, now, ttl_hours):
    """Pure function: job subdirectories of `jobs_dir` older than `ttl_hours`
    (age = `now` - dir mtime, in seconds; `now` in epoch seconds). Returns
    [(job_id, path, age_seconds)] sorted by job id. Only subdirectories are
    considered; .gitkeep and any stray non-directory entry are never touched.
    A dir's mtime is its age source, not the mtimes of files inside it."""
    expired = []
    if not os.path.isdir(jobs_dir):
        return expired
    cutoff = ttl_hours * 3600.0
    for entry in sorted(os.listdir(jobs_dir)):
        path = os.path.join(jobs_dir, entry)
        if not os.path.isdir(path):
            continue
        age = now - os.path.getmtime(path)
        if age > cutoff:
            expired.append((entry, path, age))
    return expired


def cleanup_old_jobs(jobs_dir, jobs, lock, now=None, ttl_hours=None):
    """Delete expired job dirs (see find_expired_jobs) and prune their ids
    from the in-memory `jobs` dict under `lock` (so the owner's live tab gets
    a clean "Unknown job" on status/result instead of dangling image 404s).
    Returns the list of pruned job ids (empty when nothing expired)."""
    if now is None:
        now = time.time()
    if ttl_hours is None:
        ttl_hours = JOB_TTL_HOURS
    pruned = []
    for job_id, path, age in find_expired_jobs(jobs_dir, now, ttl_hours):
        shutil.rmtree(path, ignore_errors=True)
        pruned.append(job_id)
        print(f"cleanup: deleted job {job_id} (age {age / 3600.0:.1f} h)")
    if pruned:
        with lock:
            for job_id in pruned:
                jobs.pop(job_id, None)
        print(f"cleanup: removed {len(pruned)} job dir(s) older than {ttl_hours} h")
    return pruned


def _cleanup_sweep_loop():
    """Daemon sweep of webapp_jobs/: first run ~60 s after startup (catches
    leftovers from while the app was down), then every CLEANUP_INTERVAL_HOURS.
    Started only in the serving process (see the __main__ block below)."""
    print(f"cleanup: sweep thread started (first sweep in 60 s, then every "
          f"{CLEANUP_INTERVAL_HOURS} h)")
    time.sleep(60)
    while True:
        cleanup_old_jobs(JOBS_DIR, _jobs, _jobs_lock)
        time.sleep(CLEANUP_INTERVAL_HOURS * 3600.0)


# ---------------------------------------------------------------------------
# Share links: read-only, token-scoped snapshot of a finished job.
#
# A share is written at share time to webapp_jobs/<job_id>/share.json:
#     {"token": <token_urlsafe(16)>, "created_at", "refreshed_at",
#      "expires_hint", "payload": {...}}            (timestamps ISO 8601 UTC)
# - The token is the access control (128-bit, unguessable). No auth and no
#   revocation by design: a link dies with its job (the 48 h sweep deletes
#   the job dir that holds the snapshot).
# - Disk (share.json) is the source of truth; _shares (token -> job_id) is
#   only a lookup cache, rebuilt from disk at startup (_rebuild_share_index)
#   and kept in sync on create/refresh, so links survive app restarts.
# - Writes are atomic (tmp file + os.replace), followed by os.utime(job_dir)
#   so the cleanup's age clock restarts: re-sharing deterministically extends
#   the link's life (plan §3.1).
# - Re-sharing an already-shared job refreshes the snapshot under the SAME
#   token (new refreshed_at/expires_hint), so the URL stays stable.
_shares = {}  # token -> job_id (lookup cache; disk is the source of truth)
_shares_lock = threading.Lock()


def _share_path(job_id):
    return os.path.join(JOBS_DIR, job_id, "share.json")


def _utc_iso(dt):
    return dt.astimezone(timezone.utc).isoformat()


def _rebuild_share_index():
    """Repopulate _shares from webapp_jobs/*/share.json (disk is the source
    of truth). Called at startup; tests use it to simulate an app restart.
    Returns the number of links indexed."""
    found = 0
    with _shares_lock:
        _shares.clear()
    if os.path.isdir(JOBS_DIR):
        for entry in os.listdir(JOBS_DIR):
            path = os.path.join(JOBS_DIR, entry, "share.json")
            if not os.path.isfile(path):
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    token = json.load(f).get("token")
            except (OSError, ValueError):
                print(f"share: skipping unreadable share.json in job {entry}")
                continue
            if token:
                with _shares_lock:
                    _shares[token] = entry
                found += 1
    return found


def _load_share(token):
    """(job_id, share_dict) for a valid token, else (None, None). The cache is
    the fast path; disk is the source of truth, so a token the cache knows but
    whose share.json is gone (e.g. the job was just swept) is unknown. A cache
    miss is final: tokens only enter the cache at startup or on create/refresh,
    so a token absent from it cannot exist on disk."""
    if not token:
        return None, None
    with _shares_lock:
        job_id = _shares.get(token)
    if job_id is None:
        return None, None
    try:
        with open(_share_path(job_id), encoding="utf-8") as f:
            return job_id, json.load(f)
    except (OSError, ValueError):
        return None, None


# ---------------------------------------------------------------------------
# Session persistence: the full in-memory job state, mirrored to disk so a
# session (job) survives an app restart.
#
# A job's state is written at every state change to
# webapp_jobs/<job_id>/job.json:
#     {"version": 1, "status", "step", "current", "total", "progress",
#      "error", "csv_name", "show_price", "finish", "filenames",
#      "results", "rows", "overrides", "finishes"}
# - Disk (job.json) is the source of truth, _jobs the in-memory fast path;
#   jobs are restored at startup (_restore_all_jobs) and lazily on first
#   reference by a state-reading route (_restore_job), exactly like the
#   share token index (share.json).
# - Writes are atomic (tmp file + os.replace) and best-effort: a failed
#   persist only prints a warning, so disk trouble can never break the live
#   scan. The job dir mtime is NOT touched — the 48 h cleanup clock keeps
#   running from creation/last share, as before.
# - The threading locks (image_locks, csv_lock) are not persisted; they are
#   recreated on restore (image_locks starts empty — api_update_processed
#   re-creates per-image locks via setdefault).
# - A job whose status is still queued/running when it is restored had its
#   worker thread killed by the restart: it is resumed in a fresh _run_job
#   thread (append mode) over the uploads that still have no result. A job
#   that died after the last image but before the CSV write is finished in
#   place (CSV written, status done). A job with neither results nor
#   resumable uploads is marked error.
# ---------------------------------------------------------------------------


def _job_state_path(job_id):
    return os.path.join(JOBS_DIR, job_id, "job.json")


def _persist_job_state(job_id):
    """Snapshot the job's state to job.json (atomic, best-effort, never
    raises). The state is deep-copied under _jobs_lock so the dump sees a
    consistent picture; the file write happens outside the lock."""
    try:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is None:
                return
            state = {
                "version": 1,
                "status": job.get("status"),
                "step": job.get("step"),
                "current": job.get("current", 0),
                "total": job.get("total", 0),
                "progress": job.get("progress", 0.0),
                "error": job.get("error"),
                "csv_name": job.get("csv_name"),
                "show_price": job.get("show_price", True),
                "finish": job.get("finish", "nonfoil"),
                "filenames": copy.deepcopy(job.get("filenames") or []),
                "results": copy.deepcopy(job.get("results") or []),
                "rows": copy.deepcopy(job.get("rows") or []),
                "overrides": copy.deepcopy(job.get("overrides") or {}),
                "finishes": copy.deepcopy(job.get("finishes") or {}),
            }
        # Unique tmp name: the worker thread and a request thread can
        # persist the same job concurrently, and a shared tmp file would
        # interleave. A tmp left behind by a crash is harmless — restore
        # only reads job.json, and the TTL sweep deletes the whole dir.
        path = _job_state_path(job_id)
        tmp_path = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception as exc:  # noqa: BLE001 - best-effort by contract
        print(f"session: could not persist state for job {job_id}: {exc}")


def _restore_one(job_id):
    """(job or None, resumed, restored_now) for one job's job.json.
    Loads the state into _jobs (skipping it when it is already in memory)
    and recovers a batch the restart interrupted (see the section header).
    resumed=True only when a fresh _run_job thread was started; 
    restored_now=True only when a state file was actually loaded."""
    with _jobs_lock:
        if job_id in _jobs:
            return _jobs[job_id], False, False
    path = _job_state_path(job_id)
    if not os.path.isfile(path):
        return None, False, False
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, ValueError):
        print(f"session: skipping unreadable job.json in job {job_id}")
        return None, False, False
    if (not isinstance(state, dict) or state.get("version") != 1
            or not isinstance(state.get("status"), str)
            or not isinstance(state.get("filenames"), list)
            or not isinstance(state.get("results"), list)):
        print(f"session: skipping invalid job.json in job {job_id}")
        return None, False, False
    for key, expected in (("rows", list), ("overrides", dict), ("finishes", dict)):
        value = state.get(key)
        if value is None:
            state[key] = [] if expected is list else {}
        elif not isinstance(value, expected):
            print(f"session: skipping invalid job.json in job {job_id} (bad {key})")
            return None, False, False
    # JSON stringifies the int card-index keys of the per-image pick/finish
    # maps; restore the in-memory int-key invariant (picks/finishes are
    # looked up by int index everywhere).
    for key in ("overrides", "finishes"):
        normalized = {}
        for fname, per_card in state[key].items():
            if not isinstance(per_card, dict):
                continue
            norm_card = {}
            for idx, val in per_card.items():
                try:
                    norm_card[int(idx)] = val
                except (TypeError, ValueError):
                    norm_card[idx] = val
            normalized[fname] = norm_card
        state[key] = normalized
    state.setdefault("step", "")
    state.setdefault("current", 0)
    state.setdefault("total", 0)
    state.setdefault("progress", 0.0)
    state.setdefault("error", None)
    state.setdefault("csv_name", None)
    state.setdefault("show_price", True)
    state.setdefault("finish", "nonfoil")
    # Runtime-only objects, never persisted.
    state["image_locks"] = {}
    state["csv_lock"] = threading.Lock()
    with _jobs_lock:
        if job_id in _jobs:
            return _jobs[job_id], False, False
        _jobs[job_id] = state
    resumed = False
    if state["status"] in ("queued", "running"):
        upload_dir, output_dir = _job_dirs(job_id)
        processed = {r.get("filename") for r in state["results"]}
        remaining = [f for f in state["filenames"]
                     if f not in processed
                     and os.path.isfile(os.path.join(upload_dir, f))]
        if remaining:
            # The restart killed the worker mid-batch: finish the batch in a
            # fresh thread (append mode extends the persisted results/rows,
            # batch-relative progress for the remaining images only).
            threading.Thread(
                target=_run_job,
                args=(job_id, upload_dir, output_dir,
                      state.get("finish") or "nonfoil",
                      bool(state.get("show_price", True)), remaining),
                kwargs={"append": True},
                daemon=True,
            ).start()
            resumed = True
            print(f"sessions: resuming job {job_id} "
                  f"({len(remaining)} unprocessed image(s))")
        elif state["rows"]:
            # Died after the last image but before the CSV write: finish in
            # place.
            write_csv_export(
                [r for img_rows in state["rows"] for r in img_rows], output_dir)
            _set_job(job_id, status="done", step="Done", progress=100.0,
                     csv_name="cards_export.csv", error=None)
            _persist_job_state(job_id)
            _refresh_share_if_any(job_id)
        else:
            # No results and nothing resumable (uploads gone): mark it.
            _set_job(job_id, status="error",
                     error="The scan was interrupted (server restarted before "
                           "any image finished). Please scan again.",
                     step="Failed")
            _persist_job_state(job_id)
    return state, resumed, True


def _restore_job(job_id):
    """Restore one job from its job.json into memory (no-op when it is
    already in memory or has no state file). Returns the job dict or None.
    State-reading routes call this first so jobs survive app restarts."""
    job, _resumed, _restored_now = _restore_one(job_id)
    return job


def _restore_all_jobs():
    """Restore every job with a job.json under webapp_jobs/ (startup; tests
    use it to simulate a fresh app start). Returns (restored_count,
    resumed_count) — jobs already in memory are not counted."""
    restored = 0
    resumed = 0
    if os.path.isdir(JOBS_DIR):
        for entry in sorted(os.listdir(JOBS_DIR)):
            if not os.path.isdir(os.path.join(JOBS_DIR, entry)):
                continue
            _job, was_resumed, restored_now = _restore_one(entry)
            if restored_now:
                restored += 1
                if was_resumed:
                    resumed += 1
    return restored, resumed


# Per-card finish toggle (normal/foil/etched). Finishes are stored per image
# in job["finishes"] (filename -> {card_index: finish}) and are orthogonal to
# manual set picks (job["overrides"]): a pick replaces the name/edition, a
# finish re-prices and re-tags it. Both survive re-renders because every
# label is rebuilt from the full persisted set (see api_update_processed).
VALID_FINISHES = ("nonfoil", "foil", "etched")
FINISH_SUFFIXES = {"foil": " (Foil)", "etched": " (Etched Foil)"}


def _finish_suffix(finish):
    """Name suffix for a finish ("" for nonfoil)."""
    return FINISH_SUFFIXES.get(finish, "")


def _strip_finish_suffix(name):
    """Remove a trailing finish suffix (etched first: it contains foil's)."""
    for suffix in FINISH_SUFFIXES.values():
        if name.endswith(suffix):
            return name[: -len(suffix)].rstrip()
    return name


def _finish_from_name(name):
    """Finish implied by a card/label name's suffix (nonfoil if none)."""
    for finish, suffix in FINISH_SUFFIXES.items():
        if name.endswith(suffix):
            return finish
    return "nonfoil"


def _finish_price(src, finish):
    """Price of `src` for `finish` with the standard fallback to the nonfoil
    price (mirrors find_match's detection-time pricing). `src` is any dict
    with price_usd / price_usd_nonfoil / price_usd_foil / price_usd_etched.
    Returns None when no price is known."""
    base = src.get("price_usd_nonfoil")
    if base is None:
        base = src.get("price_usd")
    if finish == "foil":
        return src.get("price_usd_foil") or base
    if finish == "etched":
        return src.get("price_usd_etched") or base
    return base


def _pick_price_source(ov, card):
    """Price triple to use for a picked card's finish toggling. New-style
    picks carry picked_card (the full picked-edition card, including
    price_usd_nonfoil); old-style picks carry only the pick-time price_usd,
    which is the only price known for them, so it answers every finish."""
    picked = ov.get("picked_card") or {}
    nonfoil = picked.get("price_usd_nonfoil")
    foil = picked.get("price_usd_foil")
    etched = picked.get("price_usd_etched")
    if nonfoil is None and foil is None and etched is None:
        price = ov.get("price_usd", card.get("price_usd"))
        return {"price_usd_nonfoil": price, "price_usd_foil": price, "price_usd_etched": price}
    return {"price_usd_nonfoil": nonfoil, "price_usd_foil": foil, "price_usd_etched": etched}


def _result_payload(job_id, job, url_scheme, token=None):
    """Result payload for a done job, shared by the owner endpoint
    (/api/result/<job_id>, url_scheme="job") and the share snapshot
    (url_scheme="share", token=...). The "share" scheme rewrites every URL into
    the token-scoped /share/<token>/... namespace, never exposes the job_id,
    and never exposes the job_id.
    Both schemes apply the job's persisted manual set picks (job["overrides"])
    and per-card finish toggles (job["finishes"]) to the card dicts: the
    payload carries exactly what the owner last saw, picks included
    (set_picked + the picked name/price/set, plus the full picked-edition
    card's display fields when the client sent picked_card), and each matched
    card's effective finish (stored toggle, else the finish implied by the
    name suffix) with its finish-specific price. Without this, the owner
    page's full re-renders (append completion, failed-remove rollback) would
    drop previously made set picks and finish toggles.
    The caller must hold _jobs_lock while reading job state."""
    if url_scheme == "share":
        base = f"/share/{token}"

        def up(name):
            return f"{base}/uploads/{name}"

        def out(name):
            return f"{base}/output/{name}"

        def dl(name):  # download == same image; the client adds `download`
            return f"{base}/output/{name}"

        def csvu():
            return f"{base}/csv"
    else:
        def up(name):
            return f"/job/{job_id}/uploads/{name}"

        def out(name):
            return f"/job/{job_id}/output/{name}"

        def dl(name):
            return f"/api/download/image/{job_id}/{name}"

        def csvu():
            return f"/api/download/csv/{job_id}"

    # show_price: the job-level price-overlay flag (top level, so the owner
    # page can initialise its results-toolbar "Show price" toggle; the
    # read-only share page ignores it). Toggling via
    # /api/update-processed updates this flag, so full re-renders (append
    # completion, failed-remove rollback) restore the user's choice.
    payload = {"results": [], "csv_url": csvu() if job["csv_name"] else None,
               "show_price": bool(job.get("show_price", True))}
    for item in job["results"]:
        picks = (job.get("overrides") or {}).get(item["filename"], {})
        cards = []
        for i, card in enumerate(item["cards"]):
            ov = picks.get(i)
            if ov:
                # Manual set pick: restore the row as the owner last saw it.
                # picked_card (full picked-edition card sent by the client)
                # upgrades the display fields; older picks without it keep
                # the detected card's edition/type/links.
                picked = ov.get("picked_card") or {}
                card = dict(card)
                for key in ("edition", "card_type", "dual_face", "scryfall_uri",
                            "tcgplayer_purchase_uri", "image_url", "image_url_back"):
                    if picked.get(key) is not None:
                        card[key] = picked[key]
                card.update({
                    "name": ov.get("label_name") or card.get("name"),
                    "price_usd": ov.get("price_usd", card.get("price_usd")),
                    "set_code": ov.get("set_code") or card.get("set_code"),
                    "set_picked": True,  # view renders it as "set selected"
                })
                # Picked-edition price triple: finish toggles on a picked
                # card price from the picked edition, not the detected one.
                # Old-style picks without picked_card keep their pick-time
                # price for every finish.
                if card.get("scryfall_id"):
                    psrc = _pick_price_source(ov, card)
                    card["price_usd_nonfoil"] = psrc["price_usd_nonfoil"]
                    card["price_usd_foil"] = psrc["price_usd_foil"]
                    card["price_usd_etched"] = psrc["price_usd_etched"]
            cards.append({
                **card,
                "crop_url": out(card["crop_name"]) if card["crop_name"] else None,
            })
        # Per-card finish toggle (matched cards only), applied AFTER the
        # picks above: the stored finish wins, else the finish implied by the
        # name suffix (detection or pick time). The name is returned PLAIN
        # (suffix stripped) — the toggle conveys the finish in the list; the
        # suffix is re-applied only on the processed-image label (label_for
        # in api_update_processed) and the CSV row. Re-prices from the
        # card's price triple (picked editions already carry the
        # picked-edition triple).
        stored_fins = (job.get("finishes") or {}).get(item["filename"]) or {}
        for i, card in enumerate(cards):
            if not card.get("scryfall_id"):
                continue
            fin = stored_fins.get(i) or _finish_from_name(card.get("name") or "")
            card["name"] = _strip_finish_suffix(card.get("name") or "")
            card["finish"] = fin
            card["price_usd"] = _finish_price(card, fin)
        payload["results"].append({
            "filename": item["filename"],
            "original_url": up(item["original_name"]),
            "processed_url": out(item["processed_name"]) if item["processed_name"] else None,
            "processed_download_url": dl(item["processed_name"]) if item["processed_name"] else None,
            "cards": cards,
        })
    return payload


# Startup: index any existing share links from disk so they resolve after a
# restart (the cache is also updated on every create/refresh).
_indexed_share_count = _rebuild_share_index()
print(f"share: indexed {_indexed_share_count} shared link(s) from disk")


def _new_job_state(total_files, show_price=True):
    return {
        "status": "queued",  # queued -> running -> done -> error
        "step": "Waiting to start...",
        "current": 0,
        "total": total_files,
        "progress": 0.0,  # job-wide percentage 0..100 (per-card granularity)
        "error": None,
        "results": [],   # list of {filename, original, processed, cards:[...]}
        "csv_name": None,
        "show_price": show_price,
        # Manual set picks, persisted per image: filename -> {card_index: ov}.
        # The frontend fires /api/update-processed automatically (in the
        # background) on every pick, so picks must survive across requests:
        # each re-render rebuilds labels from the FULL persisted set, which
        # makes concurrent / quick successive picks converge (last write
        # wins per card). See api_update_processed.
        "overrides": {},
        # Per-card finish toggles, persisted per image:
        # filename -> {card_index: "nonfoil"|"foil"|"etched"}. Applied after
        # picks in _result_payload / api_update_processed (see helpers above).
        "finishes": {},
        "image_locks": {},   # filename -> threading.Lock (serializes re-renders)
        "csv_lock": threading.Lock(),
    }


def _set_job(job_id, **kwargs):
    with _jobs_lock:
        _jobs[job_id].update(kwargs)


def _run_job(job_id, upload_dir, output_dir, finish, show_price, filenames, append=False):
    """Process `filenames` (one batch) for job `job_id`.

    Batch-aware: results/rows are seeded from the job's existing state, so an
    append batch (plan 3.3, `append=True`) extends the previous batches
    instead of replacing them. A fresh job holds empty state, so a first
    batch behaves exactly as before. `total`/progress/step stay
    batch-relative: the UI shows one progress cycle per batch."""
    total = len(filenames)

    def image_progress(idx, frac):
        """Job-wide percentage for image `idx` at in-image fraction `frac`
        (capped at 99 so the bar never moves backwards; 100 = job done)."""
        return round(min(99.0, (idx - 1 + frac) / total * 100), 1)

    try:
        _set_job(job_id, status="running")
        # Seed from existing job state (under the lock): on an append these
        # carry the previous batches' results/rows forward; on a fresh job
        # they are empty, so behavior is unchanged.
        with _jobs_lock:
            _job = _jobs[job_id]
            results = list(_job["results"])
            # Per-image CSV rows; kept in job state so the Update button can
            # re-sync the CSV with manual set picks.
            rows_by_image = list(_job.get("rows") or [])
        all_rows = [r for img_rows in rows_by_image for r in img_rows]
        for idx, filename in enumerate(filenames, start=1):
            # process_image reports (message, in-image fraction); fold the
            # fraction into the job-wide percentage so the bar advances per
            # detected card, not per image.
            def progress_cb(msg, frac, _idx=idx):
                _set_job(job_id, step=msg, progress=image_progress(_idx, frac))

            _set_job(job_id, current=idx, progress=image_progress(idx, 0.0),
                     step=f"Processing {filename} ({idx}/{total})")
            src_path = os.path.join(upload_dir, filename)
            rows, info = process_image(
                src_path, output_dir, db=_card_db, finish=finish,
                show_price=show_price, progress_cb=progress_cb,
            )
            all_rows.extend(rows)
            rows_by_image.append(rows)

            cards = []
            for card in info["cards"]:
                cards.append({
                    "name": card["name"] or "Unmatched card",
                    "price_usd": card["price_usd"],
                    # Raw per-finish prices: the client re-prices the card
                    # locally on the finish toggle from these.
                    "price_usd_nonfoil": card.get("price_usd_nonfoil"),
                    "price_usd_foil": card.get("price_usd_foil"),
                    "price_usd_etched": card.get("price_usd_etched"),
                    "finish": card["finish"],
                    "scryfall_uri": card.get("scryfall_uri"),
                    "tcgplayer_purchase_uri": card.get("tcgplayer_purchase_uri"),
                    "crop_name": os.path.basename(card["crop_path"]) if card["crop_path"] else None,
                    "card_type": card.get("card_type"),
                    "type_line": card.get("type_line"),
                    "edition": card.get("edition"),
                    "set_code": card.get("set_code"),
                    "scryfall_id": card.get("scryfall_id"),
                    "set_status": card.get("set_status"),
                    "candidates": card.get("candidates"),
                    "symbol_margin": card.get("symbol_margin"),
                    "dual_face": card.get("dual_face"),
                    "matched_face": card.get("matched_face"),
                })

            results.append({
                "filename": filename,
                "original_name": filename,
                "processed_name": os.path.basename(info["debug_path"]) if info["debug_path"] else None,
                # This batch's toggle; per-item so a later re-render (plan
                # 3.6) of an old image keeps its own price-overlay choice.
                "show_price": show_price,
                "cards": cards,
            })
            with _jobs_lock:
                _jobs[job_id]["results"] = results
                _jobs[job_id]["rows"] = rows_by_image
            # Incremental persist: a restart mid-batch resumes from the last
            # finished image (see _restore_one).
            _persist_job_state(job_id)

        _set_job(job_id, step="Writing CSV export", progress=99.0)
        write_csv_export(all_rows, output_dir)
        # error=None: an append after a failed batch clears the stale error.
        _set_job(job_id, status="done", step="Done", csv_name="cards_export.csv",
                 progress=100.0, error=None)
        # Plan 3.5: if this job has a live share, re-snapshot the current
        # state (an append batch just finished). No-op for jobs without a
        # share; best-effort, never raises.
        _refresh_share_if_any(job_id)
        _persist_job_state(job_id)
    except Exception as exc:  # noqa: BLE001 - surface any error to the UI
        _set_job(job_id, status="error", error=str(exc), step="Failed")
        _persist_job_state(job_id)


def _job_dirs(job_id):
    base = os.path.join(JOBS_DIR, job_id)
    return os.path.join(base, "uploads"), os.path.join(base, "output")


def _safe_send(directory, filename):
    """Serve `filename` from `directory`, rejecting any path traversal."""
    if not filename:
        abort(404)
    full_path = os.path.abspath(os.path.join(directory, filename))
    if os.path.commonpath([full_path, os.path.abspath(directory)]) != os.path.abspath(directory):
        abort(403)
    if not os.path.isfile(full_path):
        abort(404)
    return send_file(full_path)


@app.route("/")
def index():
    return render_template(
        "index.html",
        db_loaded=_card_db is not None and bool(_card_db.db_entries),
        db_count=len(_card_db.db_entries) if _card_db else 0,
    )


@app.route("/search")
def search():
    return render_template(
        "search.html",
        db_loaded=_card_db is not None and bool(_card_db.db_entries),
        db_count=len(_card_db.db_entries) if _card_db else 0,
    )


@app.route("/api/search-card", methods=["POST"])
def api_search_card():
    card_name = request.form.get("card_name", "").strip()
    if not card_name:
        return jsonify({"error": "Card name is required"}), 400
    if not _card_db:
        return jsonify({"error": "Card database not loaded"}), 503
    
    card = None
    scryfall_id = request.form.get("scryfall_id", "").strip()
    if scryfall_id:
        # Exact edition picked from the typeahead suggestions
        card = _card_db.get_by_scryfall_id(scryfall_id)
    if not card:
        card = _card_db.search_by_name(card_name)
    if card:
        # Generate image URL. Image files in cache/scryfall_images/ are named after
        # the Scryfall card id (see generate_phash_json.py image_cache_path),
        # NOT after illustration_id (that id is shared by reprints, so it is
        # not a valid filename here).
        image_url = None
        image_url_back = None
        if card.get('scryfall_id'):
            image_url = f"/scryfall_images/{card['scryfall_id']}.jpg"
            if card.get('dual_face'):
                image_url_back = f"/scryfall_images/{card['scryfall_id']}_back.jpg"

        return jsonify({
            "success": True,
            "card": {
                "name": card['name'],
                "price_usd": card['price_usd'],
                # The search response's price_usd IS the picked edition's
                # nonfoil price; name it explicitly so clients can carry the
                # full price triple in their pick overrides.
                "price_usd_nonfoil": card['price_usd'],
                "price_usd_foil": card['price_usd_foil'],
                "price_usd_etched": card['price_usd_etched'],
                "card_type": card.get('card_type'),
                "type_line": card.get('type_line'),
                "mana_cost": card.get('mana_cost'),
                "edition": card.get('edition'),
                "set_code": card.get('set_code'),
                "dual_face": card.get('dual_face'),
                "image_url": image_url,
                "image_url_back": image_url_back,
                "scryfall_uri": card['scryfall_uri'],
                "tcgplayer_purchase_uri": card['tcgplayer_purchase_uri'],
            }
        })
    else:
        return jsonify({"success": False, "message": "Card not found"}), 404


@app.route("/api/suggest-cards")
def api_suggest_cards():
    """Typeahead: cards whose name starts with `q` (prefix, not substring)."""
    query = request.args.get("q", "").strip()
    if not _card_db:
        return jsonify({"error": "Card database not loaded"}), 503
    if not query:
        return jsonify({"suggestions": [], "truncated": False})
    return jsonify(_card_db.suggest_names(query, limit=15))


def _refresh_share_if_any(job_id):
    """Rewrite the on-disk share snapshot of a finished job with its CURRENT
    payload — same token, stable created_at, updated refreshed_at (plan 3.5).
    No-op when the job has no live share or is not done. Best-effort: any
    failure only prints a warning, so a missing/broken snapshot can never
    break job completion or image removal."""
    try:
        with _shares_lock:
            token = next((t for t, jid in _shares.items() if jid == job_id), None)
        if not token:
            return
        now = datetime.now(timezone.utc)
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is None or job["status"] != "done":
                return
            try:
                with open(_share_path(job_id), encoding="utf-8") as f:
                    existing = json.load(f)
            except (OSError, ValueError):
                existing = {}
            if existing.get("token") != token:
                return
            share = {
                "token": token,
                "created_at": existing.get("created_at") or _utc_iso(now),
                "refreshed_at": _utc_iso(now),
                "expires_hint": _utc_iso(now + timedelta(hours=JOB_TTL_HOURS)),
                "payload": _result_payload(job_id, job, "share", token),
            }
            tmp_path = _share_path(job_id) + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(share, f, ensure_ascii=False)
            os.replace(tmp_path, _share_path(job_id))
            os.utime(os.path.join(JOBS_DIR, job_id))
    except Exception as exc:  # noqa: BLE001 - best-effort by contract
        print(f"share: could not refresh snapshot for job {job_id}: {exc}")


def _save_job_uploads(upload_dir, files, used_names):
    """Validate, dedupe and save upload files into upload_dir.

    `used_names` is a set the caller can pre-seed; newly saved names are
    added to it. Returns the list of saved (secure) filenames."""
    saved_filenames = []
    for f in files:
        # Folder uploads report a relative path like "sub/dir/card.jpg" via
        # webkitRelativePath; keep just the leaf name, flattened, and made safe.
        leaf = os.path.basename(f.filename.replace("\\", "/"))
        name = secure_filename(leaf) or "image"
        ext = os.path.splitext(name)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        base, ext2 = os.path.splitext(name)
        candidate = name
        n = 1
        while candidate in used_names:
            candidate = f"{base}_{n}{ext2}"
            n += 1
        used_names.add(candidate)
        f.save(os.path.join(upload_dir, candidate))
        saved_filenames.append(candidate)
    return saved_filenames


@app.route("/api/process", methods=["POST"])
def api_process():
    files = request.files.getlist("images")
    files = [f for f in files if f and f.filename]
    if not files:
        return jsonify({"error": "No files uploaded."}), 400

    finish = request.form.get("finish", "nonfoil")
    if finish not in ("nonfoil", "foil", "etched"):
        finish = "nonfoil"
    show_price = request.form.get("show_price", "true").lower() in ("1", "true", "on", "yes")

    job_id = uuid.uuid4().hex
    upload_dir, output_dir = _job_dirs(job_id)
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    saved_filenames = _save_job_uploads(upload_dir, files, set())

    if not saved_filenames:
        shutil.rmtree(os.path.join(JOBS_DIR, job_id), ignore_errors=True)
        return jsonify({"error": "No valid image files found in upload."}), 400

    with _jobs_lock:
        _jobs[job_id] = _new_job_state(len(saved_filenames), show_price)
        # Plan 3.7: remember the job's source images (collision-safe names)
        # so a later /add batch can dedupe against them and /remove-image can
        # validate a filename.
        _jobs[job_id]["filenames"] = saved_filenames
        # This batch's finish, kept in the state so a restart can resume the
        # batch with the same finish (see _restore_one).
        _jobs[job_id]["finish"] = finish

    # The session exists from the Scan click: persist the state immediately
    # so the job (and its owner URL) survives a restart from this moment on.
    _persist_job_state(job_id)

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, upload_dir, output_dir, finish, show_price, saved_filenames),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/process/<job_id>/add", methods=["POST"])
def api_process_add(job_id):
    """Append images to an existing (finished) job. The new batch is processed
    in the background and its results/CSV rows extend the previous batches
    (plan 3.3); the job keeps its id, so a live share snapshot refreshes under
    the same token when the batch completes (plan 3.5)."""
    _restore_job(job_id)  # restart recovery: no-op while the job is in memory
    files = request.files.getlist("images")
    files = [f for f in files if f and f.filename]
    if not files:
        return jsonify({"error": "No files uploaded."}), 400

    finish = request.form.get("finish", "nonfoil")
    if finish not in ("nonfoil", "foil", "etched"):
        finish = "nonfoil"
    show_price = request.form.get("show_price", "true").lower() in ("1", "true", "on", "yes")

    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify({"error": "Unknown job"}), 404
        if job["status"] not in ("done", "error"):
            return jsonify({"error": "Job is still processing"}), 409
        job.setdefault("filenames", [])
        used_names = set(job["filenames"])

    upload_dir, output_dir = _job_dirs(job_id)
    saved = _save_job_uploads(upload_dir, files, used_names)
    if not saved:
        return jsonify({"error": "No valid image files found in upload."}), 400

    with _jobs_lock:
        job["filenames"].extend(saved)
        # Batch-relative status: total/current/progress describe the new batch;
        # the finished batches' rows stay in results/rows.
        job["total"] = len(saved)
        job["status"] = "running"
        job["error"] = None
        # Job-level default = the LATEST batch (plan 3.6); older images keep
        # their own per-item show_price on re-render.
        job["show_price"] = show_price
        # The new batch's finish (a restart mid-append resumes with it).
        job["finish"] = finish
    os.utime(os.path.join(JOBS_DIR, job_id))
    _persist_job_state(job_id)

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, upload_dir, output_dir, finish, show_price, saved),
        kwargs={"append": True},
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id, "appended": len(saved)})


@app.route("/api/remove-image/<job_id>", methods=["POST"])
def api_remove_image(job_id):
    """Remove one source image from a finished job (plan 3.4): deletes its
    upload, detection overlay and card crops, drops it from the results and
    the CSV export, and refreshes the share snapshot if one exists.
    Body JSON: {"filename": <source filename>}. Returns {success: true} plus
    the full /api/result payload."""
    _restore_job(job_id)  # restart recovery: no-op while the job is in memory
    body = request.get_json(silent=True) or {}
    filename = body.get("filename")

    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify({"error": "Unknown job"}), 404
        if job["status"] not in ("done", "error"):
            return jsonify({"error": "Job is still processing"}), 409
        known = set(job.get("filenames") or [])
        for r in job["results"]:
            known.add(r["filename"])
        if not filename or filename not in known:
            return jsonify({"error": "Unknown source image for this job"}), 404
        if filename in job["filenames"]:
            job["filenames"].remove(filename)
        removed_idx = None
        for i, r in enumerate(job["results"]):
            if r["filename"] == filename:
                removed_idx = i
                break
        if removed_idx is not None:
            job["results"].pop(removed_idx)
            rows = job.get("rows")
            if rows is not None and removed_idx < len(rows):
                rows.pop(removed_idx)
        # The removed image's per-image pick state and lock are no longer
        # needed.
        job["overrides"].pop(filename, None)
        job["finishes"].pop(filename, None)
        job["image_locks"].pop(filename, None)
        rows_for_csv = job.get("rows")

    # File cleanup outside the locks (I/O): the upload, the detection overlay
    # (any finish suffix), and every card crop of the removed image.
    upload_dir, output_dir = _job_dirs(job_id)
    stem = os.path.splitext(filename)[0]
    Path(upload_dir, filename).unlink(missing_ok=True)
    for p in Path(output_dir).glob(f"{stem}-detected*"):
        p.unlink(missing_ok=True)
    for p in Path(output_dir).glob(f"{stem}-card*.png"):
        p.unlink(missing_ok=True)

    # Re-sync the CSV (same pattern as api_update_processed).
    with job["csv_lock"]:
        if rows_for_csv is not None:
            write_csv_export([r for img_rows in rows_for_csv for r in img_rows], output_dir)

    # Live share? Refresh its snapshot (same token) so a finished share page
    # stops showing the removed image.
    _refresh_share_if_any(job_id)
    os.utime(os.path.join(JOBS_DIR, job_id))
    _persist_job_state(job_id)

    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return jsonify({"error": "Unknown job"}), 404
        payload = _result_payload(job_id, job, "job")
    return jsonify({"success": True, **payload})


@app.route("/api/status/<job_id>")
def api_status(job_id):
    _restore_job(job_id)  # restart recovery: no-op while the job is in memory
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify({"error": "Unknown job"}), 404
        return jsonify({
            "status": job["status"],
            "step": job["step"],
            "current": job["current"],
            "total": job["total"],
            "progress": job["progress"],
            "error": job["error"],
        })


@app.route("/api/result/<job_id>")
def api_result(job_id):
    _restore_job(job_id)  # restart recovery: no-op while the job is in memory
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify({"error": "Unknown job"}), 404
        if job["status"] != "done":
            return jsonify({"error": "Job not finished"}), 409
        payload = _result_payload(job_id, job, "job")
        return jsonify(payload)


@app.route("/api/update-processed/<job_id>", methods=["POST"])
def api_update_processed(job_id):
    """Re-render one source photo's processed ("detected & identified") image
    after the user manually picked a set for an unclear card.

    The frontend fires this automatically (in the background) on every set
    pick, so concurrent requests on the same image are normal. Picks are
    therefore persisted on the job (job["overrides"], last write wins per
    card) and every re-render uses the FULL persisted pick set under a
    per-image lock: whichever request renders last reflects every pick made
    so far, so quick successive picks (same or other images) can't lose each
    other. That image's CSV rows are re-synced under a per-job CSV lock.

    Per-card finish toggles work the same way: the request's valid finishes
    are persisted on the job (job["finishes"]) and every re-render rebuilds
    labels and CSV rows from the FULL persisted pick + finish set. Finishes
    only apply to matched cards (unmatched cards have no price to re-price);
    invalid entries are silently dropped, like invalid picks.

    The detection boxes are re-derived from the original upload
    (deterministic); if the re-detection disagrees with the stored card
    count -> 509. Body JSON:
        {"filename": <source filename>,
         "cards": [{"card_index": <0-based>, "label_name": <display name
                     incl. finish tag>, "set_code": <str>, "price_usd": <num>}],
         "finishes": [{"card_index": <0-based>,
                       "finish": "nonfoil" | "foil" | "etched"}],
         "show_price": <bool, optional>}  # price overlay on the exported
                                           # image (results-page "Show price"
                                           # toggle); persisted on the item
                                           # AND the job flag when present"""
    _restore_job(job_id)  # restart recovery: no-op while the job is in memory
    body = request.get_json(silent=True) or {}
    finishes_in = body.get("finishes")
    if finishes_in is None:
        finishes_in = []
    if not isinstance(finishes_in, list):
        return jsonify({"error": "finishes must be a list"}), 400
    # Optional price-overlay choice from the results-page "Show price"
    # toggle: a real boolean only (JSON true/false -> Python bool; anything
    # else, incl. strings and None, is a client bug -> 400 like bad
    # finishes).
    show_price_in = body.get("show_price")
    if "show_price" in body and show_price_in is not None \
            and not isinstance(show_price_in, bool):
        return jsonify({"error": "show_price must be a boolean"}), 400
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify({"error": "Unknown job"}), 404
        if job["status"] != "done":
            return jsonify({"error": "Job not finished"}), 409
        item = next((r for r in job["results"] if r["filename"] == body.get("filename")), None)
        if item is None:
            return jsonify({"error": "Unknown source image for this job"}), 404
        # Merge this request's valid picks into the job's persisted picks so
        # every re-render of this image carries the full pick set.
        image_picks = job["overrides"].setdefault(item["filename"], {})
        for ov in body.get("cards") or []:
            try:
                idx = int(ov.get("card_index"))
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(item["cards"]) and ov.get("label_name"):
                image_picks[idx] = ov
        # Same persistence for per-card finish toggles (matched cards only).
        image_fins = job.setdefault("finishes", {}).setdefault(item["filename"], {})
        for f in finishes_in:
            try:
                idx = int(f.get("card_index"))
            except (TypeError, ValueError):
                continue
            fin = f.get("finish")
            if (0 <= idx < len(item["cards"]) and fin in VALID_FINISHES
                    and item["cards"][idx].get("scryfall_id")):
                image_fins[idx] = fin
        img_lock = job["image_locks"].setdefault(item["filename"], threading.Lock())
        # Price overlay on the exported image. An explicit boolean in the
        # body (results-page "Show price" toggle) wins over everything and is
        # persisted on BOTH the item and the job flag: the item flag drives
        # every later re-render of this image, and the job flag is what
        # /api/result reports for the toggle's initial state.
        if isinstance(show_price_in, bool):
            item["show_price"] = show_price_in
            job["show_price"] = show_price_in
            show_price = show_price_in
        else:
            # Plan 3.6: per-item show_price (set when the image's batch ran)
            # wins; the job-level flag is only the default for items that
            # predate it.
            show_price = item.get("show_price", job.get("show_price", True))
        item = dict(item)
        item["cards"] = list(item["cards"])

    upload_dir, output_dir = _job_dirs(job_id)
    src_path = os.path.join(upload_dir, item["original_name"])
    if not os.path.isfile(src_path):
        return jsonify({"error": "Source image not found"}), 404
    image = cv2.imread(src_path)
    if image is None:
        return jsonify({"error": "Could not read source image"}), 500

    with img_lock:
        # Re-snapshot under the image lock: a concurrent request may have
        # persisted more picks/finishes between our merge and our turn to
        # render.
        with _jobs_lock:
            picks = dict(job["overrides"][item["filename"]])
            finishes = dict(job.get("finishes", {}).get(item["filename"], {}))

        boxes = sort_boxes(find_card_quads(image), image)
        if len(boxes) != len(item["cards"]):
            return jsonify({"error": "Card detection changed since the scan; re-scan this image"}), 509

        def label_for(i, card, ov):
            # Effective finish: stored toggle, else the finish implied by the
            # name suffix (detection or pick time). Drives both the finish
            # tag in the label and the finish-specific price.
            if ov:
                set_code = ov.get("set_code") or card.get("set_code")
                raw_name = ov["label_name"]
                price_src = _pick_price_source(ov, card)
            else:
                # Unmatched cards (no scryfall_id/set code) only get the green outline.
                if not card.get("scryfall_id") and not card.get("set_code"):
                    return None
                set_code = card.get("set_code")
                raw_name = card["name"]
                price_src = card
            # Effective finish: stored toggle, else the finish implied by
            # the name suffix BEFORE stripping (detection- or pick-time
            # finish, e.g. an old-style pick labelled "... (Foil)").
            fin = finishes.get(i) or _finish_from_name(raw_name)
            name = _strip_finish_suffix(raw_name) + _finish_suffix(fin)
            return name + (f" [{set_code}]" if set_code else ""), _finish_price(price_src, fin)

        card_labels = [label_for(i, card, picks.get(i)) for i, card in enumerate(item["cards"])]
        debug_image = render_debug_overlay(image, boxes, card_labels, show_price=show_price)

        if not item.get("processed_name"):
            return jsonify({"error": "No processed image for this source image"}), 404
        debug_path = os.path.join(output_dir, item["processed_name"])
        cv2.imwrite(debug_path, debug_image, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Keep the CSV export in sync with the on-image labels: re-sync this
        # image's rows from the full persisted pick + finish set (idempotent
        # per field) and rewrite cards_export.csv.
        if picks or finishes:
            with job["csv_lock"]:
                with _jobs_lock:
                    rows_by_image = job.get("rows")
                    img_idx = next((i for i, r in enumerate(job["results"])
                                    if r["filename"] == item["filename"]), None)
                    img_rows = rows_by_image[img_idx] if (rows_by_image and img_idx is not None
                                                          and img_idx < len(rows_by_image)) else None
                if img_rows is not None:
                    for idx in sorted(set(picks) | set(finishes)):
                        if idx >= len(img_rows):
                            continue
                        card = item["cards"][idx]
                        ov = picks.get(idx)
                        if card.get("scryfall_id"):
                            if ov:
                                raw_name = ov["label_name"]
                                price_src = _pick_price_source(ov, card)
                            else:
                                raw_name = card["name"]
                                price_src = card
                            # Same finish derivation as label_for: the
                            # stored toggle wins, else the un-stripped name's
                            # suffix (detection- or pick-time finish).
                            fin = finishes.get(idx) or _finish_from_name(raw_name)
                            row = list(img_rows[idx])
                            row[0] = _strip_finish_suffix(raw_name) + _finish_suffix(fin)
                            row[3] = _finish_price(price_src, fin)
                            row[4] = fin
                            if ov:
                                row[5] = (ov.get("set_code") or card.get("set_code")) or ""
                            img_rows[idx] = tuple(row)
                        elif ov:
                            # Unmatched pick: legacy behavior — label verbatim,
                            # pick-time price, picked set; finish column untouched.
                            row = list(img_rows[idx])
                            set_code = ov.get("set_code") or card.get("set_code")
                            row[0] = ov["label_name"]
                            row[3] = ov.get("price_usd", card.get("price_usd"))
                            row[5] = set_code or ""
                            img_rows[idx] = tuple(row)
                    if rows_by_image is not None:
                        flat_rows = [r for img_rows_ in rows_by_image for r in img_rows_]
                        write_csv_export(flat_rows, output_dir)

    _persist_job_state(job_id)
    return jsonify({"success": True, "processed_name": item["processed_name"]})


@app.route("/scryfall_images/<path:filename>")
def serve_scryfall_image(filename):
    images_dir = os.path.join(SCRIPT_DIR, "cache", "scryfall_images")
    return _safe_send(images_dir, filename)


@app.route("/job/<job_id>")
def session_page(job_id):
    """The unique owner URL of the session (job). The frontend (app.js) reads
    the injected job id (window.RESUMED_JOB_ID) to re-attach to the session's
    live state — progress, results, share bar — after a tab loss or an app
    restart (the state itself is restored from job.json by _restore_job). The
    id is the unguessable uuid4 hex of the job dir, so no separate token is
    needed (same trust model as the token-scoped share links: unguessable
    link == access, 48 h TTL). Unknown or already-cleaned-up jobs get the
    session_gone 404 page. The asset routes /job/<job_id>/uploads|output/…
    coexist — Flask matches the deeper rules first."""
    _restore_job(job_id)  # restart recovery: no-op while the job is in memory
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None or not os.path.isdir(os.path.join(JOBS_DIR, job_id)):
        return render_template("session_gone.html"), 404
    return render_template(
        "index.html",
        db_loaded=_card_db is not None and bool(_card_db.db_entries),
        db_count=len(_card_db.db_entries) if _card_db else 0,
        resumed_job_id=job_id,
    )


@app.route("/job/<job_id>/uploads/<path:filename>")
def serve_upload(job_id, filename):
    upload_dir, _ = _job_dirs(job_id)
    return _safe_send(upload_dir, filename)


@app.route("/job/<job_id>/output/<path:filename>")
def serve_output(job_id, filename):
    _, output_dir = _job_dirs(job_id)
    return _safe_send(output_dir, filename)


@app.route("/api/download/csv/<job_id>")
def download_csv(job_id):
    _, output_dir = _job_dirs(job_id)
    full_path = os.path.join(output_dir, "cards_export.csv")
    if not os.path.isfile(full_path):
        abort(404)
    return send_file(full_path, as_attachment=True, download_name="cards_export.csv")


@app.route("/api/download/image/<job_id>/<path:filename>")
def download_image(job_id, filename):
    _, output_dir = _job_dirs(job_id)
    full_path = os.path.abspath(os.path.join(output_dir, filename))
    if os.path.commonpath([full_path, os.path.abspath(output_dir)]) != os.path.abspath(output_dir):
        abort(403)
    if not os.path.isfile(full_path):
        abort(404)
    return send_file(full_path, as_attachment=True, download_name=os.path.basename(full_path))


@app.route("/api/share/<job_id>", methods=["POST"])
def api_share_create(job_id):
    """Create (or refresh) the job's read-only share link. The job must be
    done (409 while running, 404 for unknown jobs). An existing share.json is
    refreshed under the SAME token — re-sharing keeps the URL stable,
    re-snapshots the current in-memory state (manual set picks included), and
    extends the link's 48 h lifetime (os.utime below restarts the cleanup
    clock). Returns only the path; the frontend composes the absolute URL
    from location.origin, so the same code works on the test tunnel today and
    on the real domain later."""
    _restore_job(job_id)  # restart recovery: no-op while the job is in memory
    now = datetime.now(timezone.utc)
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify({"error": "Unknown job"}), 404
        if job["status"] != "done":
            return jsonify({"error": "Job not finished"}), 409
        try:
            with open(_share_path(job_id), encoding="utf-8") as f:
                existing = json.load(f)
        except (OSError, ValueError):
            existing = None
        token = existing.get("token") if existing else None
        if not token:
            token = secrets.token_urlsafe(16)
        expires_hint = _utc_iso(now + timedelta(hours=JOB_TTL_HOURS))
        share = {
            "token": token,
            "created_at": (existing.get("created_at") if existing else _utc_iso(now)),
            "refreshed_at": _utc_iso(now),
            "expires_hint": expires_hint,
            "payload": _result_payload(job_id, job, "share", token),
        }
        tmp_path = _share_path(job_id) + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(share, f, ensure_ascii=False)
        os.replace(tmp_path, _share_path(job_id))
        os.utime(os.path.join(JOBS_DIR, job_id))
    with _shares_lock:
        _shares[token] = job_id
    return jsonify({"share_path": f"/share/{token}", "expires_hint": expires_hint})


@app.route("/api/share/<job_id>", methods=["GET"])
def api_share_info(job_id):
    """Read-only share info for the session page: tells a resumed owner page
    whether the job already has a share link, so the frontend can re-show the
    share bar without re-clicking Share. Pure read — no re-snapshot, no token
    rotation, no os.utime (the 48 h clock is untouched). 404 for unknown jobs
    and for jobs without a share.json. Coexists with the POST route on the
    same rule (api_share_create) — Flask dispatches on method."""
    _restore_job(job_id)  # restart recovery: no-op while the job is in memory
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    try:
        with open(_share_path(job_id), encoding="utf-8") as f:
            share = json.load(f)
    except (OSError, ValueError):
        return jsonify({"error": "No share for this job"}), 404
    token = share.get("token")
    if not token:
        return jsonify({"error": "No share for this job"}), 404
    return jsonify({"share_path": f"/share/{token}",
                    "expires_hint": share.get("expires_hint")})


@app.route("/share/<token>")
def share_page(token):
    """Read-only share page. 404 on unknown/deleted tokens (no enumeration)."""
    _, share = _load_share(token)
    if share is None:
        abort(404)
    return render_template("share.html", token=token, shared_at=share.get("refreshed_at"))


@app.route("/api/share/<token>/result")
def share_result(token):
    """The stored snapshot payload plus shared_at (snapshot time) and
    expires_hint. 404 on unknown token."""
    _, share = _load_share(token)
    if share is None:
        return jsonify({"error": "Unknown link"}), 404
    payload = dict(share.get("payload") or {})
    payload["shared_at"] = share.get("refreshed_at")
    payload["expires_hint"] = share.get("expires_hint")
    return jsonify(payload)


@app.route("/share/<token>/uploads/<path:filename>")
def share_upload(token, filename):
    job_id, share = _load_share(token)
    if share is None:
        abort(404)
    upload_dir, _ = _job_dirs(job_id)
    return _safe_send(upload_dir, filename)


@app.route("/share/<token>/output/<path:filename>")
def share_output(token, filename):
    job_id, share = _load_share(token)
    if share is None:
        abort(404)
    _, output_dir = _job_dirs(job_id)
    return _safe_send(output_dir, filename)


@app.route("/share/<token>/csv")
def share_csv(token):
    job_id, share = _load_share(token)
    if share is None:
        abort(404)
    _, output_dir = _job_dirs(job_id)
    full_path = os.path.join(output_dir, "cards_export.csv")
    if not os.path.isfile(full_path):
        abort(404)
    return send_file(full_path, as_attachment=True, download_name="cards_export.csv")


@app.route("/share/<token>/ref/<filename>")
def share_ref(token, filename):
    """Scryfall reference images, token-scoped (the read-only renderer points
    its refBase here instead of /scryfall_images)."""
    job_id, share = _load_share(token)
    if share is None:
        abort(404)
    return _safe_send(os.path.join(SCRIPT_DIR, "cache", "scryfall_images"), filename)


# Startup: restore the job states from disk so sessions survive a restart
# (job.json is the source of truth, like share.json for the links). Placed
# at the end of the module because the resume paths use helpers defined
# above the routes (_run_job, _job_dirs, _refresh_share_if_any, ...).
_restored_job_count, _resumed_job_count = _restore_all_jobs()
print(f"sessions: restored {_restored_job_count} job(s) from disk "
      f"({_resumed_job_count} resumed)")


if __name__ == "__main__":
    debug = True
    # The debug reloader executes this module in two processes (parent =
    # reloader, child = the actual server, which sets WERKZEUG_RUN_MAIN=true).
    # Only the serving process runs the cleanup sweep. Note: app.debug is
    # False until app.run() flips it, so the guard uses the local `debug`
    # intent, not app.debug.
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not debug:
        threading.Thread(target=_cleanup_sweep_loop, daemon=True).start()
    app.run(debug=debug, host="127.0.0.1", port=5000)
