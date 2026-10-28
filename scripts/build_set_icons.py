#!/usr/bin/env python3
# Copyright (C) 2026 IgnorantGrimyYak
"""Fetch every MTG set icon from the Scryfall API and build
static/js/set_icons.js (the set symbol icons shown in the UI).

Two steps run in sequence, with resources/set_icon.txt as the on-disk
intermediate:

  1. Scryfall API  ->  resources/set_icon.txt
  2. resources/set_icon.txt  ->  static/js/set_icons.js

Download step data sources:

  GET https://api.scryfall.com/sets
      Every Scryfall set in a single List object (newest first); each entry
      carries `code`, `name` and `icon_svg_uri`. NOTE:
      GET https://api.scryfall.com/sets/all is NOT the list endpoint —
      "all" is the code of the Alliances set, so that URL returns just that
      one set (the docs page /docs/api/sets/all documents GET /sets).

  GET <icon_svg_uri>
      The set's monochrome SVG icon, served from the svgs.scryfall.io CDN
      (e.g. https://svgs.scryfall.io/sets/m19.svg).

Downloaded icons are cached in cache/set_icons/<CODE>.svg, so a re-run only
fetches sets added to Scryfall since the last run; --refresh forces every
icon to be re-downloaded. Icons are fetched with a small thread pool, with
retries and exponential backoff (a 429 Retry-After is honored).

Outputs:

  resources/set_icon.txt
      Blocks of two lines (separated by a blank line):

          <name> {<CODE>}:
          data:image/svg+xml;base64,<base64 of svg>

      Set codes are UPPERCASE (the file convention: data/scryfall_db.sqlite
      stores lowercase codes and the UI looks them up case-insensitively),
      entries in the API's release order (newest first). The file is
      replaced only when EVERY icon was resolved (cache or download) — a
      failed run leaves the old file in place and exits non-zero.

  static/js/set_icons.js
      One object: SET_ICONS  {setCode: svgMarkup}  e.g. 'FRA': '<svg ...>'
      The icons are stored as inline <svg> strings (not <img> data URLs)
      and recolored to fill="currentColor": an <img> document cannot
      inherit the page's text color, but an inlined <svg> can. The page
      inserts the string directly so the icon always matches the color of
      the edition text next to it; lookups in the page are done
      case-insensitively (see setIconFor() in templates/search.html).

Run from the project root (or anywhere; paths are resolved relative to
this file):

    python scripts/build_set_icons.py            # fetch (cached) + txt + JS
    python scripts/build_set_icons.py --refresh  # re-download every icon
    python scripts/build_set_icons.py --offline  # no network: only regenerate
                                                 # the JS from the existing txt
"""

import argparse
import base64
import concurrent.futures
import json
import re
import sys
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "resources" / "set_icon.txt"
OUT = ROOT / "static" / "js" / "set_icons.js"
CACHE_DIR = ROOT / "cache" / "set_icons"
SETS_URL = "https://api.scryfall.com/sets"
USER_AGENT = "magic-card-test (set-icon refresh script)"
WORKERS = 8
MAX_RETRIES = 5
RETRYABLE = (429, 500, 502, 503, 504)

_tls = threading.local()


def get_session():
    """One requests.Session per thread (a shared Session is not thread-safe)."""
    session = getattr(_tls, "session", None)
    if session is None:
        session = requests.Session()
        session.headers["User-Agent"] = USER_AGENT
        _tls.session = session
    return session


def get_text(url, label):
    """GET url with retries/backoff. Returns the body text, or None."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = get_session().get(url, timeout=30)
        except requests.RequestException as exc:
            print(f"  {label}: {exc} (attempt {attempt}/{MAX_RETRIES})", file=sys.stderr)
            time.sleep(min(2 ** attempt, 30))
            continue
        if resp.status_code == 200:
            return resp.text
        if resp.status_code in RETRYABLE:
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() \
                else min(2 ** attempt, 30)
            print(f"  {label}: HTTP {resp.status_code}, retrying in {wait:g}s",
                  file=sys.stderr)
            time.sleep(wait)
            continue
        print(f"  {label}: HTTP {resp.status_code} (not retrying)", file=sys.stderr)
        return None
    print(f"  {label}: giving up after {MAX_RETRIES} attempts", file=sys.stderr)
    return None


def one_line_svg(text):
    """Collapse the SVG to a single line (a data URL must not wrap)."""
    return re.sub(r"\s*\n\s*", "", text).strip()


def is_svg(svg):
    return svg.startswith("<svg") and svg.endswith("</svg>")


def download_set_icons(refresh):
    """Fetch the set list + every icon, then write SRC (resources/set_icon.txt)."""
    print(f"Fetching the set list from {SETS_URL} ...")
    listing = get_text(SETS_URL, "set list")
    if listing is None:
        sys.exit("Could not fetch the set list from the Scryfall API.")
    try:
        data = json.loads(listing)
    except ValueError:
        sys.exit("The set list is not valid JSON: %r" % listing[:200])
    if data.get("object") != "list" or data.get("has_more"):
        sys.exit("Unexpected set-list reply (expected one complete List object).")
    sets = data["data"]
    no_icon = [s["code"] for s in sets if not s.get("icon_svg_uri")]
    if no_icon:
        sys.exit("Sets without an icon_svg_uri: %s" % no_icon[:10])
    bad_name = [s["code"] for s in sets
                if not s.get("name") or "\n" in s["name"] or "}" in s["name"]]
    if bad_name:
        sys.exit("Sets with a name that would break the file format: %s"
                 % bad_name[:10])
    print(f"{len(sets)} sets")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def fetch(set_obj):
        code = set_obj["code"]
        cache = CACHE_DIR / (code.upper() + ".svg")
        if not refresh and cache.exists():
            try:
                cached = one_line_svg(cache.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                cached = ""
            if is_svg(cached):
                return set_obj, cached, True
        text = get_text(set_obj["icon_svg_uri"], code)
        svg = one_line_svg(text) if text is not None else ""
        if not is_svg(svg):
            return set_obj, None, False
        cache.write_text(svg, encoding="utf-8")
        return set_obj, svg, False

    by_code, failed, n_cached, n_fetched = {}, [], 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(fetch, s) for s in sets]
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            set_obj, svg, from_cache = fut.result()
            if svg is None:
                failed.append(set_obj["code"])
            else:
                by_code[set_obj["code"]] = (set_obj["name"], svg)
                if from_cache:
                    n_cached += 1
                else:
                    n_fetched += 1
            if i % 100 == 0 or i == len(sets):
                print(f"  icons: {i}/{len(sets)}")

    if failed:
        shown = ", ".join(sorted(failed)[:10])
        more = f" ... (+{len(failed) - 10} more)" if len(failed) > 10 else ""
        print(f"{len(failed)} icon(s) could not be downloaded: {shown}{more}",
              file=sys.stderr)
        sys.exit("%s left untouched - re-run once the failures are resolved."
                 % SRC.name)

    lines = []
    for s in sets:  # API order (newest set first)
        name, svg = by_code[s["code"]]
        data_url = ("data:image/svg+xml;base64,"
                    + base64.b64encode(svg.encode("utf-8")).decode("ascii"))
        lines.append(f"{name} {{{s['code'].upper()}}}:")
        lines.append(data_url)
        lines.append("")
    SRC.parent.mkdir(exist_ok=True)
    SRC.write_bytes(("\r\n".join(lines)).encode("utf-8"))
    print(f"Wrote {len(sets)} entries to {SRC.relative_to(ROOT)} "
          f"({n_fetched} downloaded, {n_cached} from {CACHE_DIR.name}/)")


# "Name {CODE}:" followed (possibly after whitespace) by the data URL. The
# code between the braces contains no braces itself (guarded at download time).
PAIR_RE = re.compile(
    r"^([^\r\n]+?)\s*\{([^{}\r\n]+)\}\s*:\s*\r?\n[ \t]*(data:image[^\r\n]*)",
    re.M,
)


def recolor(svg):
    """Turn the icon monochrome in the current text color.

    - any explicit color fill becomes fill="currentColor" (fill="none" is kept)
    - any explicit color stroke becomes stroke="currentColor"
    - the root <svg> gets fill="currentColor" so paths without a fill inherit it
    - fixed width/height attributes are dropped (viewBox keeps the aspect ratio)
    - class="set-icon" and aria-hidden="true" are added for page-side styling
    """
    def paint_repl(match):
        # match: <attr>=(quote)(value)(quote); keep "none"/"currentColor" values
        if match.group(3).strip().lower() in ("none", "currentcolor"):
            return match.group(0)
        return '%s=%scurrentColor%s' % (match.group(1), match.group(2), match.group(2))
    svg = re.sub(r'(fill|stroke)=("|\')([^"\']*)\2', paint_repl, svg)
    svg = re.sub(r'\s(?:width|height)\s*=\s*("[^"]*"|\'[^\']*\')', '', svg, count=2)

    def root_repl(match):
        tag = match.group(0)
        for name, attr in (("fill", 'fill="currentColor"'),
                           ("class", 'class="set-icon"'),
                           ("aria-hidden", 'aria-hidden="true"')):
            if not re.search(r'\s%s=' % name, tag):
                tag = tag[:-1] + " " + attr + ">"
        return tag
    return re.sub(r"^<svg\b[^>]*>", root_repl, svg, count=1)


def js_str(s):
    """Render a string as a single-quoted JS literal (inputs are simple ASCII)."""
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def build_js():
    """Generate static/js/set_icons.js from SRC (resources/set_icon.txt)."""
    if not SRC.exists():
        sys.exit("%s not found - run without --offline first." % SRC)
    text = SRC.read_text(encoding="utf-8")
    pairs = PAIR_RE.findall(text)
    if not pairs:
        sys.exit("No icon entries found in %s" % SRC)

    entries = {}
    for name, code, url in pairs:
        code = code.strip()
        if code in entries:
            sys.exit("Duplicate set code %s in %s" % (code, SRC))
        if not re.fullmatch(r"[A-Za-z0-9]+", code):
            sys.exit("Unexpected set code %r (for %s) in %s" % (code, name, SRC))
        if not url.startswith("data:image/svg+xml;base64,"):
            sys.exit("Not a base64 SVG data URL for %s in %s" % (code, SRC))
        svg = recolor(base64.b64decode(url.split(",", 1)[1]).decode("utf-8"))
        # The markup goes into a single-quoted JS literal; keep it simple.
        if any(ch in svg for ch in ("\n", "\r", "'", "\\")):
            sys.exit("Unexpected character in SVG for %s in %s" % (code, SRC))
        entries[code] = svg

    # Sanity check: every data URL line should have been consumed.
    n_lines = len(re.findall(r"^data:image[^\r\n]*", text, re.M))
    if n_lines != len(entries):
        sys.exit("Parsed %d entries but found %d data URLs" % (len(entries), n_lines))

    # Keep the source file's order (sets in release order).
    lines = [
        "// Set edition SVG icons for the search page.",
        "// Generated from resources/set_icon.txt by scripts/build_set_icons.py; do not edit by hand.",
        "// Keys are set codes (short set names), e.g. FRA, M19. The database stores",
        "// lowercase set codes, so look up case-insensitively (see setIconFor() in search.html).",
        "// Values are inline <svg> strings recolored to fill=\"currentColor\" (with class",
        "// \"set-icon\"), so the icon inherits the text color of the element it is inserted into.",
        "const SET_ICONS = {",
    ]
    for code, svg in entries.items():
        lines.append("    %s: %s," % (js_str(code), js_str(svg)))
    lines.append("};")

    OUT.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))
    print("Wrote %s (%d set icons)" % (OUT.relative_to(ROOT), len(entries)))


def main():
    parser = argparse.ArgumentParser(
        description="Fetch the set icons from the Scryfall API and build "
                    "static/js/set_icons.js.")
    parser.add_argument("--refresh", action="store_true",
                        help="re-download every icon, ignoring cache/set_icons/")
    parser.add_argument("--offline", action="store_true",
                        help="no network: only regenerate static/js/set_icons.js "
                             "from the existing resources/set_icon.txt")
    args = parser.parse_args()

    if args.offline:
        build_js()
    else:
        download_set_icons(args.refresh)
        build_js()


if __name__ == "__main__":
    main()
