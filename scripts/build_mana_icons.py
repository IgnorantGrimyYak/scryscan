#!/usr/bin/env python3
# Copyright (C) 2026 IgnorantGrimyYak
"""Fetch every Scryfall card-symbol icon from the Scryfall API and build
static/js/mana_icons.js (the mana cost icons shown in the UI).

Two steps run in sequence, with resources/svg_icon.txt as the on-disk
intermediate:

  1. Scryfall API  ->  resources/svg_icon.txt
  2. resources/svg_icon.txt  ->  static/js/mana_icons.js

Download step data sources:

  GET https://api.scryfall.com/symbology
      Every card symbol in a single List object (docs:
      https://scryfall.com/docs/api/card-symbols). Each entry carries
      `symbol` (e.g. {W/U}), `english` (e.g. "one white or blue mana")
      and `svg_uri`.

  GET <svg_uri>
      The symbol's SVG icon, served from the svgs.scryfall.io CDN
      (e.g. https://svgs.scryfall.io/card-symbols/WU.svg).

The API has no {F} (double-faced) symbol. The hand-authored {F} icon
(FALLBACK_SVGS below) is appended at the end of the file — it is the icon
the flip-arrows badge in results_view.js is based on.

Downloaded icons are cached in cache/symbol_icons/<NAME>.svg (NAME is the
file name of the symbol's svg_uri, e.g. WU.svg, HALF.svg), so a re-run only
fetches symbols added to Scryfall since the last run; --refresh forces
every icon to be re-downloaded. Icons are fetched with a small thread
pool, with retries and exponential backoff (a 429 Retry-After is honored).

Outputs:

  resources/svg_icon.txt
      Blocks of two lines (separated by a blank line):

          <english> {<SYMBOL>}:
          data:image/svg+xml;base64,<base64 of svg>

      Entries in the API's order, the fallback {F} last. The file is
      replaced only when EVERY icon was resolved (cache or download) — a
      failed run leaves the old file in place and exits non-zero.

  static/js/mana_icons.js
      Two objects (keys in the source file's order):
        MANA_ICONS        {token: dataUrl}  e.g. '{R}': 'data:image/svg+xml;...'
        MANA_ICON_TITLES  {token: english}  e.g. '{R}': 'one red mana'
      Tokens are the exact symbols used in a card's mana_cost (see
      renderManaCost() in templates/search.html).

Run from the project root (or anywhere; paths are resolved relative to
this file):

    python scripts/build_mana_icons.py            # fetch (cached) + txt + JS
    python scripts/build_mana_icons.py --refresh  # re-download every icon
    python scripts/build_mana_icons.py --offline  # no network: only regenerate
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
SRC = ROOT / "resources" / "svg_icon.txt"
OUT = ROOT / "static" / "js" / "mana_icons.js"
CACHE_DIR = ROOT / "cache" / "symbol_icons"
SYMBOLS_URL = "https://api.scryfall.com/symbology"
USER_AGENT = "magic-card-test (symbol-icon refresh script)"
WORKERS = 8
MAX_RETRIES = 5
RETRYABLE = (429, 500, 502, 503, 504)

# Symbols the Scryfall API does not serve: hand-authored icons appended at
# the end of the file. {F} is the double-faced icon the flip-arrows badge
# in results_view.js is based on.
FALLBACK_SVGS = {
    "{F}": ("double-faced {F}",
            '<svg focusable="false" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024"><path d="M884.3,357.6c116.8,117.7,151.7,277-362.2,320V496.4L243.2,763.8L522,1031.3V860.8C828.8,839.4,1244.9,604.5,884.3,357.6z"></path><path d="M557.8,288.2v138.4l230.8-213.4L557.8,0v142.8c-309.2,15.6-792.1,253.6-426.5,503.8C13.6,527.9,30,330.1,557.8,288.2z"></path></svg>'),
}

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


def download_symbol_icons(refresh):
    """Fetch the symbol list + every icon, then write SRC (resources/svg_icon.txt)."""
    print(f"Fetching the card symbols from {SYMBOLS_URL} ...")
    listing = get_text(SYMBOLS_URL, "symbol list")
    if listing is None:
        sys.exit("Could not fetch the card symbols from the Scryfall API.")
    try:
        data = json.loads(listing)
    except ValueError:
        sys.exit("The symbol list is not valid JSON: %r" % listing[:200])
    if data.get("object") != "list" or data.get("has_more"):
        sys.exit("Unexpected symbol-list reply (expected one complete List object).")
    syms = data["data"]
    no_svg = [s["symbol"] for s in syms if not s.get("svg_uri")]
    if no_svg:
        sys.exit("Symbols without an svg_uri: %s" % no_svg[:10])
    bad_symbol = [s["symbol"] for s in syms
                  if not re.fullmatch(r"\{[^{}]+\}", s.get("symbol", ""))]
    if bad_symbol:
        sys.exit("Malformed symbols in the reply: %s" % bad_symbol[:10])
    bad_english = [s["symbol"] for s in syms
                   if not s.get("english") or any(ch in s["english"]
                                                  for ch in ("\n", "{", "}"))]
    if bad_english:
        sys.exit("Symbols with an english description that would break the "
                 "file format: %s" % bad_english[:10])
    names = [Path(s["svg_uri"]).name for s in syms]
    if len(set(names)) != len(names):
        seen = {n for n in names if names.count(n) > 1}
        sys.exit("Duplicate svg_uri file names in the reply: %s" % sorted(seen))
    print(f"{len(syms)} symbols")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def fetch(sym):
        name = Path(sym["svg_uri"]).name
        cache = CACHE_DIR / name
        if not refresh and cache.exists():
            try:
                cached = one_line_svg(cache.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                cached = ""
            if is_svg(cached):
                return sym, cached, True
        text = get_text(sym["svg_uri"], sym["symbol"])
        svg = one_line_svg(text) if text is not None else ""
        if not is_svg(svg):
            return sym, None, False
        cache.write_text(svg, encoding="utf-8")
        return sym, svg, False

    by_symbol, failed, n_cached, n_fetched = {}, [], 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(fetch, s) for s in syms]
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            sym, svg, from_cache = fut.result()
            if svg is None:
                failed.append(sym["symbol"])
            else:
                by_symbol[sym["symbol"]] = (sym["english"], svg)
                if from_cache:
                    n_cached += 1
                else:
                    n_fetched += 1
            if i % 25 == 0 or i == len(syms):
                print(f"  icons: {i}/{len(syms)}")

    if failed:
        shown = ", ".join(sorted(failed)[:10])
        more = f" ... (+{len(failed) - 10} more)" if len(failed) > 10 else ""
        print(f"{len(failed)} icon(s) could not be downloaded: {shown}{more}",
              file=sys.stderr)
        sys.exit("%s left untouched - re-run once the failures are resolved."
                 % SRC.name)

    lines = []
    for s in syms:  # API order
        english, svg = by_symbol[s["symbol"]]
        data_url = ("data:image/svg+xml;base64,"
                    + base64.b64encode(svg.encode("utf-8")).decode("ascii"))
        lines.append(f"{english} {s['symbol']}:")
        lines.append(data_url)
        lines.append("")
    for token in sorted(FALLBACK_SVGS):  # hand-authored icons, last
        english, svg = FALLBACK_SVGS[token]
        data_url = ("data:image/svg+xml;base64,"
                    + base64.b64encode(svg.encode("utf-8")).decode("ascii"))
        lines.append(f"{english} {token}:")
        lines.append(data_url)
        lines.append("")
    SRC.parent.mkdir(exist_ok=True)
    SRC.write_bytes(("\r\n".join(lines)).encode("utf-8"))
    print(f"Wrote {len(syms) + len(FALLBACK_SVGS)} entries to {SRC.relative_to(ROOT)} "
          f"({n_fetched} downloaded, {n_cached} from {CACHE_DIR.name}/, "
          f"{len(FALLBACK_SVGS)} hand-authored)")


# "<english> {SYMBOL}:" followed (possibly after whitespace) by the data
# URL. The symbol between the braces contains no braces itself (guarded at
# download time); it may be non-ASCII (e.g. {∞}).
PAIR_RE = re.compile(
    r"^([^\r\n]+?)\s*(\{[^{}\r\n]+\})\s*:\s*\r?\n[ \t]*(data:image[^\r\n]*)",
    re.M,
)


def js_str(s):
    """Render a string as a single-quoted JS literal (may contain Unicode)."""
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def build_js():
    """Generate static/js/mana_icons.js from SRC (resources/svg_icon.txt)."""
    if not SRC.exists():
        sys.exit("%s not found - run without --offline first." % SRC)
    text = SRC.read_text(encoding="utf-8")
    pairs = PAIR_RE.findall(text)
    if not pairs:
        sys.exit("No icon entries found in %s" % SRC)

    entries = {}
    for english, token, url in pairs:
        if token in entries:
            sys.exit("Duplicate symbol %s in %s" % (token, SRC))
        if not url.startswith("data:image/svg+xml;base64,"):
            sys.exit("Not a base64 SVG data URL for %s in %s" % (token, SRC))
        svg = base64.b64decode(url.split(",", 1)[1]).decode("utf-8")
        if not is_svg(svg):
            sys.exit("Not an SVG for %s in %s" % (token, SRC))
        entries[token] = (english, url)

    # Sanity check: every data URL line should have been consumed.
    n_lines = len(re.findall(r"^data:image[^\r\n]*", text, re.M))
    if n_lines != len(entries):
        sys.exit("Parsed %d entries but found %d data URLs" % (len(entries), n_lines))

    lines = [
        "// Mana cost SVG icons for the search page.",
        "// Generated from resources/svg_icon.txt by scripts/build_mana_icons.py; do not edit by hand.",
        "// Keys are Scryfall card symbols, e.g. {2}, {W/U}, {X} (the exact tokens",
        "// used in a card's mana_cost). MANA_ICONS maps a symbol to its SVG data",
        "// URL; MANA_ICON_TITLES maps a symbol to its English description (tooltip).",
        "const MANA_ICONS = {",
    ]
    for token, (english, url) in entries.items():
        lines.append("    %s: %s," % (js_str(token), js_str(url)))
    lines.append("};")
    lines.append("")
    lines.append("const MANA_ICON_TITLES = {")
    for token, (english, url) in entries.items():
        lines.append("    %s: %s," % (js_str(token), js_str(english)))
    lines.append("};")

    OUT.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))
    print("Wrote %s (%d icons)" % (OUT.relative_to(ROOT), len(entries)))


def main():
    parser = argparse.ArgumentParser(
        description="Fetch the card-symbol icons from the Scryfall API and "
                    "build static/js/mana_icons.js.")
    parser.add_argument("--refresh", action="store_true",
                        help="re-download every icon, ignoring cache/symbol_icons/")
    parser.add_argument("--offline", action="store_true",
                        help="no network: only regenerate static/js/mana_icons.js "
                             "from the existing resources/svg_icon.txt")
    args = parser.parse_args()

    if args.offline:
        build_js()
    else:
        download_symbol_icons(args.refresh)
        build_js()


if __name__ == "__main__":
    main()
