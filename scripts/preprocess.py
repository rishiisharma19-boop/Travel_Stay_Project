"""Clean the supplied Booking.com sample and link stays to canonical cities.

Run from the project root:
    python scripts/preprocess.py

Unresolved cities are deliberately retained with a NULL destination_id and
written to unmatched_cities.csv. We do not silently fuzzy-assign them.
"""
from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
CSV_PATH = RAW / "booking_com-travel_sample.csv"
CITIES_PATH = RAW / "cities.json"
DB_PATH = ROOT / "travel_stay.db"

# Explicit, reviewable aliases only. Extend this dictionary after validating
# the source records. Keys and values are normalized by normalize_city().
ALIASES = {
    "bangalore": "bengaluru",
    "bombay": "mumbai",
    "calcutta": "kolkata",
    "madras": "chennai",
    "allahabad": "prayagraj",
    "pondicherry": "puducherry",
    "gurgaon": "gurugram",
    "mysore": "mysuru",
}


def normalize_city(value: object) -> str:
    """Unicode-aware city normalization used for matching and grouping."""
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.casefold().strip()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA})


def numeric_rating(value: object) -> float:
    """Extract numeric star value from values such as '3 stars' or '2-star hotel'."""
    if pd.isna(value):
        return np.nan
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else np.nan


def parse_cleanliness(value: object) -> float:
    """Extract the cleanliness sub-rating from the encoded review breakdown."""
    if pd.isna(value):
        return np.nan
    match = re.search(r"Cleanliness\s*:\s*(\d+(?:\.\d+)?)", str(value), re.I)
    return float(match.group(1)) if match else np.nan


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists() or not CITIES_PATH.exists():
        raise FileNotFoundError(
            "Raw datasets not found. Put booking_com-travel_sample.csv and "
            "cities.json inside data/raw/."
        )

    raw = pd.read_csv(CSV_PATH, low_memory=False)
    with CITIES_PATH.open("r", encoding="utf-8") as f:
        city_records = json.load(f)

    cities = pd.DataFrame(city_records)
    required_city_cols = {"id", "name", "state"}
    if not required_city_cols.issubset(cities.columns):
        raise ValueError(f"cities.json must contain {sorted(required_city_cols)}")

    cities["city_normalized"] = cities["name"].map(normalize_city)
    cities["state_normalized"] = cities["state"].map(normalize_city)

    # Canonical lookup supports city + state where possible. City-only lookup
    # is accepted only when it resolves to exactly one canonical city.
    by_city_state: dict[tuple[str, str], dict] = {}
    by_city: dict[str, list[dict]] = {}
    for row in cities.to_dict("records"):
        by_city_state[(row["city_normalized"], row["state_normalized"])] = row
        by_city.setdefault(row["city_normalized"], []).append(row)

    for alias, canonical in ALIASES.items():
        alias_key, canonical_key = normalize_city(alias), normalize_city(canonical)
        if canonical_key in by_city:
            by_city[alias_key] = by_city[canonical_key]
            for state_key, row in list(by_city_state.items()):
                if state_key[0] == canonical_key:
                    by_city_state[(alias_key, state_key[1])] = row

    # Standardized, deliberately compact stay table. Original source columns
    # remain available in the raw CSV and are not overwritten.
    stays = pd.DataFrame()
    text_cols = ["city", "state", "country", "property_name", "property_type",
                 "hotel_brand", "hotel_description", "hotel_facilities", "room_type",
                 "address", "pageurl", "property_id", "uniq_id", "hotel_star_rating",
                 "site_review_count", "site_review_rating", "site_stay_review_rating"]
    for col in text_cols:
        stays[col] = clean_text(raw[col]) if col in raw.columns else pd.NA

    stays["city_normalized"] = stays["city"].map(normalize_city)
    stays["state_normalized"] = stays["state"].map(normalize_city)
    stays["hotel_stars"] = stays["hotel_star_rating"].map(numeric_rating)
    stays["review_rating"] = pd.to_numeric(stays["site_review_rating"], errors="coerce")
    stays["review_count"] = pd.to_numeric(stays["site_review_count"], errors="coerce")
    stays["cleanliness_rating"] = stays["site_stay_review_rating"].map(parse_cleanliness)
    stays["latitude"] = pd.to_numeric(raw.get("latitude"), errors="coerce")
    stays["longitude"] = pd.to_numeric(raw.get("longitude"), errors="coerce")
    stays["room_count"] = pd.to_numeric(raw.get("room_count"), errors="coerce")

    def resolve_city(row: pd.Series):
        city_key, state_key = row["city_normalized"], row["state_normalized"]
        if not city_key:
            return None, "missing_city"
        if state_key and (city_key, state_key) in by_city_state:
            return by_city_state[(city_key, state_key)]["id"], "exact_city_state_or_alias"
        candidates = by_city.get(city_key, [])
        if len(candidates) == 1:
            return candidates[0]["id"], "unique_city_or_alias"
        if len(candidates) > 1:
            return None, "ambiguous_city_state_required"
        return None, "unmatched_city"

    resolved = stays.apply(resolve_city, axis=1, result_type="expand")
    stays["destination_id"] = resolved[0]
    stays["mapping_status"] = resolved[1]
    stays["country"] = stays["country"].fillna("India")

    # Keep unresolved city rows for audit, but remove records without a usable
    # property name or city from the recommendation-ready output.
    before = len(stays)
    stays = stays[stays["property_name"].notna() & stays["city"].notna()].copy()
    after_required = len(stays)

    # Deduplicate by source unique ID where present, otherwise by name + city.
    stays["record_key"] = stays["uniq_id"].fillna(
        stays["property_name"].str.casefold() + "|" + stays["city_normalized"]
    )
    stays = stays.drop_duplicates(subset=["record_key"], keep="first")

    # Useful derived fields for later recommendation logic.
    stays["amenities_text"] = stays["hotel_facilities"].fillna("").str.casefold()
    stays["has_wifi"] = stays["amenities_text"].str.contains("wi ?fi", regex=True)
    stays["has_parking"] = stays["amenities_text"].str.contains("parking")
    stays["has_restaurant"] = stays["amenities_text"].str.contains("restaurant")
    stays["has_pool"] = stays["amenities_text"].str.contains("pool")
    stays["has_air_conditioning"] = stays["amenities_text"].str.contains("air conditioning")

    # Persist only the columns used by the MVP, while retaining source URL and
    # descriptive fields for result cards and later feature engineering.
    output_cols = [
        "record_key", "property_id", "uniq_id", "property_name", "city", "state",
        "country", "destination_id", "mapping_status", "property_type", "hotel_brand",
        "hotel_stars", "review_rating", "review_count", "cleanliness_rating",
        "room_type", "room_count", "hotel_description", "hotel_facilities",
        "amenities_text", "has_wifi", "has_parking", "has_restaurant", "has_pool",
        "has_air_conditioning", "latitude", "longitude", "address", "pageurl"
    ]
    stays[output_cols].to_csv(OUT / "stays_cleaned.csv", index=False)

    destination_output = cities[["id", "name", "state"]].rename(
        columns={"id": "destination_id", "name": "city"}
    )
    destination_output["country"] = "India"
    destination_output.to_csv(OUT / "destinations.csv", index=False)

    unmatched = stays[stays["destination_id"].isna()].copy()
    unmatched.groupby(["city", "state", "mapping_status"], dropna=False).size().reset_index(
        name="property_count"
    ).sort_values("property_count", ascending=False).to_csv(
        OUT / "unmatched_cities.csv", index=False
    )

    # SQLite MVP schema. destination_id is nullable for unresolved locations;
    # those rows remain available for later manual mapping.
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript("""
            DROP TABLE IF EXISTS stays;
            DROP TABLE IF EXISTS destinations;
            CREATE TABLE destinations (
                destination_id TEXT PRIMARY KEY,
                city TEXT NOT NULL,
                state TEXT,
                country TEXT NOT NULL DEFAULT 'India'
            );
            CREATE TABLE stays (
                stay_id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_key TEXT UNIQUE NOT NULL,
                property_id TEXT,
                property_name TEXT NOT NULL,
                destination_id TEXT,
                city TEXT NOT NULL,
                state TEXT,
                country TEXT,
                mapping_status TEXT NOT NULL,
                property_type TEXT,
                hotel_brand TEXT,
                hotel_stars REAL,
                review_rating REAL,
                review_count REAL,
                cleanliness_rating REAL,
                room_type TEXT,
                room_count REAL,
                hotel_description TEXT,
                hotel_facilities TEXT,
                has_wifi INTEGER,
                has_parking INTEGER,
                has_restaurant INTEGER,
                has_pool INTEGER,
                has_air_conditioning INTEGER,
                latitude REAL,
                longitude REAL,
                address TEXT,
                pageurl TEXT,
                FOREIGN KEY(destination_id) REFERENCES destinations(destination_id)
            );
            CREATE INDEX idx_stays_destination ON stays(destination_id);
            CREATE INDEX idx_stays_city ON stays(city);
            CREATE INDEX idx_stays_rating ON stays(review_rating);
        """)
        destination_output.to_sql("destinations", conn, if_exists="append", index=False)
        db_cols = [
            "record_key", "property_id", "property_name", "destination_id", "city", "state",
            "country", "mapping_status", "property_type", "hotel_brand", "hotel_stars",
            "review_rating", "review_count", "cleanliness_rating", "room_type", "room_count",
            "hotel_description", "hotel_facilities", "has_wifi", "has_parking", "has_restaurant",
            "has_pool", "has_air_conditioning", "latitude", "longitude", "address", "pageurl"
        ]
        db_stays = stays[db_cols].copy()
        for col in ["has_wifi", "has_parking", "has_restaurant", "has_pool", "has_air_conditioning"]:
            db_stays[col] = db_stays[col].astype(int)
        db_stays.to_sql("stays", conn, if_exists="append", index=False)

    report = {
        "source_file": CSV_PATH.name,
        "source_rows": int(len(raw)),
        "source_columns": int(len(raw.columns)),
        "source_missing_property_name": int(raw["property_name"].isna().sum()),
        "source_missing_property_id": int(raw["property_id"].isna().sum()),
        "source_missing_city": int(raw["city"].isna().sum()),
        "rows_removed_missing_name_or_city": int(before - after_required),
        "cleaned_stay_rows": int(len(stays)),
        "destination_rows": int(len(destination_output)),
        "mapped_stays": int(stays["destination_id"].notna().sum()),
        "unmapped_stays": int(stays["destination_id"].isna().sum()),
        "mapping_status_counts": {str(k): int(v) for k, v in stays["mapping_status"].value_counts(dropna=False).items()},
        "price_field_available": False,
        "notes": [
            "The supplied property sample has no reliable price-per-night field.",
            "Unresolved city mappings are retained and are not silently fuzzy-matched.",
            "The dataset crawl_date is 2016-09-01; records are historical sample data, not live inventory."
        ]
    }
    (OUT / "preprocessing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSQLite database created: {DB_PATH}")
    print(f"Cleaned CSV created: {OUT / 'stays_cleaned.csv'}")


if __name__ == "__main__":
    main()
