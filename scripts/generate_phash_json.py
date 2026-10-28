# Copyright (C) 2026 IgnorantGrimyYak
import gzip
import hashlib
import json
import sys
from io import BytesIO
from pathlib import Path

import imagehash
import requests
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_JSONL = PROJECT_DIR / "data" / "default-cards.jsonl"
OUTPUT_JSON = PROJECT_DIR / "data" / "default-cards.json"
IMAGE_CACHE_DIR = PROJECT_DIR / "cache" / "scryfall_images"


def artwork_crop(image):
    """Extract the same central artwork region used by the detector."""
    height, width = image.height, image.width
    top, bottom = int(height * 0.10), int(height * 0.55)
    left, right = int(width * 0.08), int(width * 0.92)
    return image.crop((left, top, right, bottom))


def image_cache_path(card, image_url, suffix=""):
    """Return a stable cache path for a card image.

    ``suffix`` distinguishes faces of dual-sided cards: the front face uses the
    default ``<id>.jpg`` path, the back face uses ``<id>_back.jpg``.
    """
    card_id = card.get("id")
    cache_key = card_id or hashlib.sha256(image_url.encode("utf-8")).hexdigest()
    return IMAGE_CACHE_DIR / f"{cache_key}{suffix}.jpg"


def load_or_download_image(card, image_url, suffix=""):
    """Load a cached Scryfall image or download it once when absent."""
    cache_path = image_cache_path(card, image_url, suffix)
    if cache_path.exists():
        try:
            with Image.open(cache_path) as cached_image:
                return cached_image.convert("RGB")
        except Exception as error:
            print(f"Invalid cached image {cache_path}: {error}. Downloading again.")

    try:
        response = requests.get(
            image_url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        image.save(cache_path, format="JPEG", quality=95)
        return image
    except Exception as error:
        print(f"Error downloading image {image_url}: {error}")
        return None


def card_mana_cost(card):
    """Mana cost for a raw-dump card: top-level, else front face.

    In the bulk dump transform/dual_face cards have a null top-level
    ``mana_cost``; the cost lives in ``card_faces[0].mana_cost``.
    """
    cost = card.get("mana_cost")
    if cost:
        return cost
    faces = card.get("card_faces") or []
    if faces:
        return faces[0].get("mana_cost") or None
    return None


def build_entry(card, phash_front, phash_back=None):
    """Build the default-cards.json entry for one card.

    Dual-face cards (transform/dual_face) get the back-face hash in a
    ``phash_back`` field plus the ``layout``; other entries are unchanged.
    """
    entry = {
        "id": card.get("id"),
        "name": card.get("name"),
        "set": card.get("set"),
        "set_name": card.get("set_name"),
        "card_type": card.get("type_line"),
        "type_line": card.get("type_line"),
        "mana_cost": card_mana_cost(card),
        "edition": card.get("set_name"),
        "phash": phash_front,
        "illustration_id": card.get("illustration_id"),
        "cardmarket_id": card.get("cardmarket_id"),
        "scryfall_uri": card.get("scryfall_uri"),
        "tcgplayer_purchase_uri": (card.get("purchase_uris") or {}).get("tcgplayer"),
        "prices": {
            "usd": (card.get("prices") or {}).get("usd"),
            "usd_foil": (card.get("prices") or {}).get("usd_foil"),
            "usd_etched": (card.get("prices") or {}).get("usd_etched"),
        },
    }
    if phash_back is not None:
        entry["phash_back"] = phash_back
        entry["layout"] = card.get("layout")
    return entry


def open_raw_dump(raw_path):
    """Open a raw Scryfall dump for text reading.

    The bulk-data page serves a gzip-compressed JSON Lines download
    (``.jsonl.gz``); it is read straight through gzip without extracting it
    first. Plain ``.json`` / ``.jsonl`` files are read as-is.
    """
    if str(raw_path).endswith(".gz"):
        return gzip.open(raw_path, "rt", encoding="utf-8")
    return raw_path.open("r", encoding="utf-8")


def iter_raw_cards(raw_path):
    """Yield card objects from a raw Scryfall dump.

    Accepts the bulk-data file as downloaded (a single JSON array of card
    objects) or a JSON Lines file (one card object per line), each plain
    (``.json`` / ``.jsonl``) or gzip-compressed (``.json.gz`` / ``.jsonl.gz``).
    """
    with open_raw_dump(raw_path) as input_file:
        first_char = input_file.read(1)
        while first_char and first_char.isspace():
            first_char = input_file.read(1)
        if first_char not in ("{", "["):
            sys.exit(
                f"Error: {raw_path} does not look like a JSON dump (first "
                f"character {first_char!r}). Pass the file downloaded from "
                "Scryfall's bulk-data page as-is — it is a gzip-compressed "
                "JSON file, not a tar archive."
            )
        input_file.seek(0)
        if first_char == "[":
            for card in json.load(input_file):
                yield card
        else:
            for line in input_file:
                line = line.strip()
                if line:
                    yield json.loads(line)


def normalize_dump(raw_path):
    """Convert a raw dump (JSON array or JSON Lines, plain or gzip-compressed)
    to data/default-cards.jsonl.

    Raises ValueError when the input is the output file itself (converting
    it in place would truncate it before it is read).
    """
    if Path(raw_path).resolve() == INPUT_JSONL.resolve():
        raise ValueError(
            f"the input dump is the same file as the output ({INPUT_JSONL}); "
            "pass the downloaded bulk-data file instead"
        )
    with INPUT_JSONL.open("w", encoding="utf-8") as output_file:
        count = 0
        for card in iter_raw_cards(raw_path):
            output_file.write(json.dumps(card) + "\n")
            count += 1
    print(f"Wrote {count} cards to {INPUT_JSONL}")


def process_jsonl():
    """Generate the processed card JSON file with artwork perceptual hashes."""
    processed_cards = []

    with INPUT_JSONL.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                card = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"Invalid JSON on line {line_number}: {error}")
                continue

            faces = card.get("card_faces") or []
            is_dual = (
                card.get("layout") in ("transform", "dual_face") and len(faces) == 2
            )
            if is_dual:
                # Dual-face cards only expose per-face image URIs in the bulk dump.
                front_url = (faces[0].get("image_uris") or {}).get("normal")
                back_url = (faces[1].get("image_uris") or {}).get("normal")
            else:
                image_uris = card.get("image_uris") or (
                    faces[0].get("image_uris") if faces else None
                ) or {}
                front_url = image_uris.get("normal")
                back_url = None
            if not front_url:
                continue

            front_image = load_or_download_image(card, front_url)
            if front_image is None:
                continue

            try:
                phash = str(imagehash.phash(artwork_crop(front_image), hash_size=16))
            except Exception as error:
                print(f"Error calculating pHash for {card.get('name')}: {error}")
                continue

            phash_back = None
            if is_dual:
                if not back_url:
                    continue
                back_image = load_or_download_image(card, back_url, suffix="_back")
                if back_image is None:
                    continue
                try:
                    phash_back = str(
                        imagehash.phash(artwork_crop(back_image), hash_size=16)
                    )
                except Exception as error:
                    print(
                        f"Error calculating back-face pHash for {card.get('name')}: {error}"
                    )
                    continue

            processed_card = build_entry(card, phash, phash_back)

            processed_cards.append(processed_card)

            if line_number % 1000 == 0:
                print(f"Processed {line_number} lines; valid cards: {len(processed_cards)}")

    with OUTPUT_JSON.open("w", encoding="utf-8") as output_file:
        json.dump(processed_cards, output_file, indent=2)

    print(f"Finished: wrote {len(processed_cards)} cards with pHash to {OUTPUT_JSON}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raw_path = Path(sys.argv[1])
        if not raw_path.is_file():
            sys.exit(f"Error: raw dump not found: {raw_path}")
        print(f"Converting {raw_path} to JSON Lines ...")
        try:
            normalize_dump(raw_path)
        except ValueError as error:
            sys.exit(f"Error: {error}")
    elif not INPUT_JSONL.exists():
        sys.exit(
            f"Error: {INPUT_JSONL} not found.\n"
            "Download the Scryfall 'Default Cards' bulk data and pass the file\n"
            "as the first argument — the .jsonl.gz download can be used\n"
            "directly, e.g.:\n"
            "    python scripts\\generate_phash_json.py "
            "path\\to\\default-cards-all.jsonl.gz"
        )
    process_jsonl()
