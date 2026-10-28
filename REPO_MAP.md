# Repo Map — magic_card_test

**What it is:** Detect Magic: The Gathering cards in photos (OpenCV), identify them
against a local Scryfall database via perceptual hashing (pHash), correct perspective
to upright 5:7 crops, and export results (name, set, price, finish) — via a Flask web
UI or a CLI. Python 3.10+, venv in `.venv/`. Setup and build order: `README.md`.

## Python modules

| File | Role |
|---|---|
| `app.py` | Flask app, single module — all `/api/` routes: `POST /api/process` (new job), `POST /api/process/<job_id>/add` (append a batch to a finished job), `POST /api/remove-image/<job_id>`, `GET /api/status/<job_id>` (progress 0–100 with per-card granularity), `GET /api/result/<job_id>` (owner payload), `POST /api/update-processed/<job_id>` (server-side re-render of one image after a set pick / finish toggle / price-overlay flip), `POST /api/search-card` (lookup by `scryfall_id`; also the edition-switching path — there is no dedicated route), `GET /api/suggest-cards` (name typeahead), CSV/image download routes. Pages: `/` and `GET /job/<job_id>` (owner session page — re-renders `index.html` with `window.RESUMED_JOB_ID` injected; unknown/expired → `session_gone.html`), `/search`, `GET /share/<token>` (read-only share page + token-scoped asset routes, no `job_id` in URLs). Jobs live in memory in `_jobs`, mirrored to `job.json`; share snapshots in `share.json`; the token index is rebuilt from disk at startup. Env: `SCRYSCAN_DISABLE_SET_DISAMBIGUATION` (any value → pHash-only mode), `SCRYSCAN_STAGED_BAND=0` (full-band-for-all instead of the staged protocol). `MAX_CONTENT_LENGTH = 300 MB`. |
| `extract_cards.py` | Core engine + CLI (`--foil`, `--etched` mark all cards in a run). Public API: `CardDatabase`, `process_image` (progress cb `(message, frac)`), `write_csv_export`, `find_card_quads`, `sort_boxes`, `render_debug_overlay` (the last two drive the web app's re-renders). Detection = contour/aspect/fill filters, then hardening: `correct_boxes` (grid row repair), `_complete_grid` (synthesizes grid holes, content-validated), `_split_subcard_boxes` + `_rescue_fusion_boxes` (recovery passes run LAST, after grid repair + hole synthesis, so grid logic gets first claim). Matching = vectorized pHash (uint64 bit planes + popcount LUT, front + back faces) with a multi-shift crop probe grid — pHash is hypersensitive to 1–2 px shifts, do not drop the shift search. `find_match` returns `matched_face`, `dual_face`, `scryfall_id`, `set_status`, `candidates`, `symbol_margin`, `type_line`, the stable price triple + effective `price_usd`; multi-set cards go through `set_disambiguation.disambiguate` (degrades gracefully to pHash-only; one status line printed at startup). Crops: 5:7, 1400 px long side, prices in USD. |
| `set_disambiguation.py` | Self-contained set-disambiguation layer (no DB access, no CLI). `disambiguate(...)`: staged protocol (default) = coarse band pass over ALL candidates → text tiebreak (only when the top-2 symbol+coarse margin < 0.10, applied to the top 18) → full band pass for the top 2; `staged_band=False` = full-band-for-all. Discriminators: per-set symbol template (zone in 488×680 reference space), the plst corner-logo template, and band/text template matching against each candidate's OWN reference image — no OCR. An identical-reference-text guard forces `ambiguous` for same-print near-ties (downgrade only; never fires for `NO_TEMPLATE_SETS`, currently just `plst` — a digital-only set whose reference image is its real twin's illustration). `make_photo()` = simulated-photo generator for offline tests. Known defect: gk2/prw2 (per-card guild-sigil sets) get corrupted symbol templates; the band tiebreak compensates. |
| `scripts/generate_phash_json.py` | Scryfall "Default Cards" dump (JSON array or JSONL, plain or gzip-compressed, first CLI arg; auto-detected) → `data/default-cards.json` with pHash. Downloads + caches every card image into `cache/scryfall_images/` — slow; aborting is safe, cached images are reused on re-runs (so a later run with a new dump only fetches what is missing). Dual-face: front in `phash`, back in `phash_back` (top-level `image_uris` is null in the dump; faces live in `card_faces[*]`; backs cache as `<id>_back.jpg`). Also writes `mana_cost` (dual-face → `card_faces[0]`) and `type_line`. |
| `scripts/build_scryfall_db.py` | `data/default-cards.json` → `data/scryfall_db.sqlite` (drops + recreates; skips `vanguard_avatar` layouts and rows without a type line). |
| `scripts/build_set_symbol_templates.py` | DB + cached images → `data/set_symbol_templates.pkl` (~6 min; skips `plst`, missing images; imports the production `set_disambiguation` module). |
| `scripts/build_mana_icons.py` / `scripts/build_set_icons.py` | Scryfall API (`/symbology`, `/sets`) → `resources/svg_icon.txt` / `resources/set_icon.txt` (base64, CRLF) → `static/js/mana_icons.js` / `static/js/set_icons.js`. SVGs cached in `cache/symbol_icons/` / `cache/set_icons/` (re-run fetches only what's missing; `--refresh` forces; `--offline` regenerates JS from the txt only). The txt is written only when EVERY icon resolved (a failed run keeps the old file, exits non-zero). Mana set includes a hand-authored `{F}` flip-arrow SVG (no API symbol for it). |

## Web UI

| File | Role |
|---|---|
| `templates/index.html` | Upload page (multi-image picker, progress bar, results section). Doubles as the session page for `GET /job/<job_id>` via the injected `window.RESUMED_JOB_ID`. |
| `templates/search.html` | Card search page: name typeahead (`/api/suggest-cards`) + detail panel (mana icons from `mana_icons.js`, set icons, dual-face images side by side). |
| `templates/session_gone.html` | 404 page for an unknown/expired session. |
| `templates/share.html` | Read-only share page (`window.SHARE = {token, sharedAt}` inline). |
| `static/js/app.js` | Owner page logic: upload/append, status polling, the automatic NON-BLOCKING re-render after picks (one `/api/update-processed` in flight per image, coalesced; UI stays interactive), share bar, append + remove image, New-Scan confirmation modal, session resume (`resumeSession`). |
| `static/js/results_view.js` | Shared results renderer `createResultsView(rootEl, data, opts)` used by BOTH owner and share pages: card rows, the per-card finish toggle (its client-side re-pricing `priceForFinish` is an exact mirror of the server's `_finish_price` — keep the two in sync), set-candidate chips with hover reference preview, dual-face flip, the global view toolbar (thumbnails / images / set-selection / owner-only price-overlay), grand total. |
| `static/js/share.js` | Read-only share viewer (fetches the stored snapshot; 404/failure → "expired" message). |
| `static/js/set_icons.js` / `static/js/mana_icons.js` | Generated, ship pre-generated — do NOT hand-edit; rebuild via the `build_set_icons.py` / `build_mana_icons.py` scripts. |
| `static/css/style.css` | All styling (dark theme, Cinzel/Inter). |
| `static/img/` | Favicons, `card-back.png`. |
| `resources/image-icon.svg` | Source for the "reference image" icon inlined in `results_view.js` — not loaded at runtime. |

## Data & generated files

Shipped (checked in): `data/plst_logo_template.npy` (static The List corner-logo, 13×14 uint8), `resources/svg_icon.txt` + `resources/set_icon.txt` (icon sources, rebuilt by the build scripts), `resources/image-icon.svg`, `docs/images/` (README screenshots — `scan-results.png` hero, `upload.png`, `upload-queued.png`, `price-labels.png`, `reference-preview.png` — and `logo.png`, the README banner; only referenced from `README.md`), `LICENSE` (AGPL-3.0, full canonical text).

Generated (gitignored, absent in a fresh checkout — build order in `README.md`):

| File | Notes |
|---|---|
| `data/default-cards.jsonl` → `data/default-cards.json` | ~600 MB raw dump → ~100 MB with pHash. |
| `cache/scryfall_images/` | ~118k jpgs, ~18 GB; also the reference-image pool behind the UI hover previews (missing → "No reference image in stock"). |
| `data/scryfall_db.sqlite` | ~82 MB, the primary lookup source. |
| `data/set_symbol_templates.pkl` | Format `(set_disambiguation.py mtime, {set_code: (gray, mask)})`; stale/missing → pHash-only with a warning. Rebuild whenever `set_disambiguation.py` changes. |
| `cache/symbol_icons/`, `cache/set_icons/` | Cached SVGs for the icon build scripts. All `cache/` folders keep their layout via committed `.gitkeep` files. |

Working dirs (gitignored, runtime): `input/` (CLI photos), `output/` (CLI crops + CSV), `webapp_jobs/<job_id>/{uploads,output,job.json,share.json}`.

## Job & share-link lifecycle

- A job is created on `POST /api/process`; a worker thread processes the images; the full state (`filenames`, `results`, `rows` = CSV rows, `overrides` = manual set picks incl. the full `picked_card`, `finishes` = per-card finish toggles) is mirrored to `job.json` atomically on every change (locks are NOT persisted). Appended batches EXTEND results/CSV instead of replacing them; each image keeps its own per-item `show_price` flag.
- Restart recovery: `_restore_all_jobs()` at startup resumes interrupted batches, completes a job that died before its CSV write, marks dead-end jobs `error`.
- TTL: `JOB_TTL_HOURS = 48` from the job DIR mtime; daemon sweep every 24 h (first at startup +60 s, serving process only — no double sweep under the dev reloader); re-sharing restarts the clock. After cleanup, downloads and share links stop working.
- Share: `POST /api/share/<job_id>` snapshots the finished job's CURRENT owner payload (manual picks included) into `share.json` — idempotent under the same token, atomic write; the snapshot is auto-refreshed in place whenever a shared job changes (`_refresh_share_if_any`), so a live share page tracks appended/removed images. Token index is disk-backed → links survive app restarts; unknown tokens 404 on every token route.

## Cross-file contracts

- DB `cards` columns: `scryfall_id, name, set_code, set_name, card_type, mana_cost, edition, phash, illustration_id, cardmarket_id, price_usd, price_usd_foil, price_usd_etched, scryfall_uri, tcgplayer_purchase_uri, phash_back, type_line` (indexes on `phash`, `name`, `(name, set_code)`).
- `cards_export.csv`: 7 columns `card_name,input_file,card_number,price_usd,finish,set,type_line`, written **UTF-8 WITH BOM** (`utf-8-sig`) so Windows Excel renders the em-dash type lines; readers use `utf-8-sig` to strip it. `set` = the selected (winning) set code.
- Finishes: valid values `nonfoil | foil | etched`; name suffixes `" (Foil)"` / `" (Etched Foil)"`. Effective finish = the stored toggle, else the finish implied by the name suffix. Foil/etched prices fall back to the nonfoil base when missing (0 counts as missing). Labels and CSV rows are re-derived from the effective finish, so a no-change re-render is byte-identical and reverting a toggle reproduces the original image bytes + CSV row. The card dicts carry the stable triple `price_usd_nonfoil / price_usd_foil / price_usd_etched` plus `price_usd` = the price FOR the effective finish.
- The DB is loaded once at Flask startup and shared read-only; `/api/search-card` opens the sqlite file per request, so a DB rebuild is picked up by a running app without restart.
- `scryfall_uri` / `tcgplayer_purchase_uri` null → plain-text name / no TCGplayer icon.

## Conventions & gotchas

- Query images are OpenCV **BGR with NO conversion** — pHash values were computed the same way.
- Dual-face (transform/dual_face) cards: `image_uris` is null for all of them in the bulk dump (images live in `card_faces[*]`); a match on either face is accepted; a back-face win reports the back name (after ` // `) while identity/price stay the dual card's.
- `card_type` is the *printed* type line; `type_line` is the raw Scryfall type line (`"Basic Land — Swamp"`); they differ for full-art cards and are copied independently, never reconciled.
- `_split_cells` preserves the parent quad's corner WINDING (the perspective warp maps corner i → destination corner i; reversed winding produces a mirrored crop and breaks pHash matching).
- All set-disambiguation zones/boxes are in 488×680 reference-image space (`cache/scryfall_images/<scryfall_id>.jpg`).
- The band tiebreak uses a wide scale/angle grid with a heavily padded search window because the detected quad sits INSIDE the true card edge — the warp can magnify past ~2.05× and clip past the photo edges.
