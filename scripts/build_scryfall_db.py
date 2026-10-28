# Copyright (C) 2026 IgnorantGrimyYak
"""Build the SQLite card database from the processed Scryfall JSON file."""

import json
import sqlite3
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_JSON = PROJECT_DIR / "data" / "default-cards.json"
DATABASE_PATH = PROJECT_DIR / "data" / "scryfall_db.sqlite"

CREATE_TABLE_SQL = """
CREATE TABLE cards (
    scryfall_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    set_code TEXT NOT NULL,
    set_name TEXT NOT NULL,
    card_type TEXT,
    mana_cost TEXT,
    edition TEXT,
    phash TEXT,
    illustration_id TEXT,
    cardmarket_id INTEGER,
    price_usd REAL,
    price_usd_foil REAL,
    price_usd_etched REAL,
    scryfall_uri TEXT,
    tcgplayer_purchase_uri TEXT,
    phash_back TEXT,
    type_line TEXT
)
"""
INSERT_CARD_SQL = """
INSERT INTO cards (
    scryfall_id, name, set_code, set_name, card_type, mana_cost, edition, phash,
    illustration_id, cardmarket_id, price_usd, price_usd_foil,
    price_usd_etched, scryfall_uri, tcgplayer_purchase_uri, phash_back, type_line
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def price_as_float(value):
    """Convert a Scryfall price value to a float, or return None when absent."""
    return float(value) if value else None


def card_database_row(card):
    """Return the database values needed for one card."""
    prices = card.get("prices") or {}
    return (
        card.get("id"),
        card.get("name"),
        card.get("set"),
        card.get("set_name"),
        card.get("card_type"),
        card.get("mana_cost") or None,
        card.get("edition") or card.get("set_name"),
        card.get("phash"),
        card.get("illustration_id"),
        card.get("cardmarket_id"),
        price_as_float(prices.get("usd")),
        price_as_float(prices.get("usd_foil")),
        price_as_float(prices.get("usd_etched")),
        card.get("scryfall_uri"),
        card.get("tcgplayer_purchase_uri"),
        card.get("phash_back"),
        card.get("type_line"),
    )


def create_database():
    """Create the card database, replacing any existing database file."""
    with INPUT_JSON.open("r", encoding="utf-8") as input_file:
        cards = json.load(input_file)

    database_path = str(DATABASE_PATH)
    DATABASE_PATH.unlink(missing_ok=True)

    cards_to_insert = []
    for index, card in enumerate(cards, start=1):
        if card.get("layout") == "vanguard_avatar":
            continue

        database_row = card_database_row(card)
        if database_row[4]:
            cards_to_insert.append(database_row)

        if index % 5000 == 0:
            print(f"Processed {index} / {len(cards)} cards...")

    with sqlite3.connect(database_path) as connection:
        connection.execute(CREATE_TABLE_SQL)
        connection.execute("CREATE INDEX idx_phash ON cards (phash)")
        connection.execute("CREATE INDEX idx_name ON cards (name)")
        connection.execute("CREATE INDEX idx_name_set ON cards (name, set_code)")
        connection.executemany(INSERT_CARD_SQL, cards_to_insert)

    print(f"Created database '{DATABASE_PATH}' with {len(cards_to_insert)} cards.")


if __name__ == "__main__":
    create_database()