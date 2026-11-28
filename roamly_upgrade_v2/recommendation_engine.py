"""Roamly recommendation engine for the existing SQLite stays/destinations schema."""
from __future__ import annotations
import math, re, sqlite3, unicodedata
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "travel_stay.db"
AMENITY_COLUMNS = {
    "wifi": "has_wifi", "wi-fi": "has_wifi", "parking": "has_parking",
    "restaurant": "has_restaurant", "pool": "has_pool", "swimming pool": "has_pool",
    "air conditioning": "has_air_conditioning", "ac": "has_air_conditioning"
}

def norm(value):
    value = unicodedata.normalize("NFKD", str(value or ""))
    value = "".join(c for c in value if not unicodedata.combining(c)).casefold()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()

def haversine_km(lat1, lon1, lat2, lon2):
    if any(pd.isna(x) for x in (lat1, lon1, lat2, lon2)): return None
    r = 6371.0088
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dp, dl = math.radians(float(lat2)-float(lat1)), math.radians(float(lon2)-float(lon1))
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*r*math.asin(math.sqrt(a))

class TravelRecommendationEngine:
    def __init__(self, db_path=DB_PATH):
        self.db_path = Path(db_path)
        self.reload()

    def reload(self):
        with sqlite3.connect(self.db_path) as con:
            self.stays = pd.read_sql_query("SELECT * FROM stays", con)
            self.destinations = pd.read_sql_query("SELECT * FROM destinations", con)
            try: self.places = pd.read_sql_query("SELECT * FROM places", con)
            except Exception: self.places = pd.DataFrame()
        if self.stays.empty:
            self.vectorizer, self.matrix = None, None
            return
        text_cols = [c for c in ["property_name","property_type","hotel_brand","hotel_description","hotel_facilities","room_type","city"] if c in self.stays]
        self.stays["_text"] = self.stays[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=12000, ngram_range=(1,2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(self.stays["_text"].tolist())

    def _dedupe(self, df):
        if df.empty: return df
        d = df.copy()
        # Keep the best evidence row per normalized property/city/address. Include address
        # when present to avoid merging distinct branches with the same brand/name.
        def key(r):
            name, city, address = norm(r.get("property_name")), norm(r.get("city")), norm(r.get("address"))
            return "|".join((name, city))
        d["_property_key"] = d.apply(key, axis=1)
        d["_quality"] = pd.to_numeric(d.get("review_rating"), errors="coerce").fillna(0) * 100 + pd.to_numeric(d.get("review_count"), errors="coerce").fillna(0).clip(upper=10000) / 100
        d = d.sort_values("_quality", ascending=False).drop_duplicates("_property_key", keep="first")
        return d.drop(columns=["_property_key","_quality"], errors="ignore")

    def _preference_filter(self, df, property_type, preferences):
        d = df.copy()
        if property_type:
            term = norm(property_type)
            # Property type is inconsistently coded in source data; use a soft relevance
            # boost rather than incorrectly excluding records whose source code is opaque.
            d["_type_match"] = d["property_type"].fillna("").map(lambda x: 1.0 if term in norm(x) else 0.0)
        else: d["_type_match"] = 0.0
        for pref in preferences or []:
            col = AMENITY_COLUMNS.get(norm(pref))
            if col and col in d.columns:
                # Amenities are soft preferences because source tagging can be incomplete.
                d[f"_{col}"] = pd.to_numeric(d[col], errors="coerce").fillna(0).astype(float)
        return d

    def _score(self, df, query, preferences, property_type):
        if df.empty: return df
        d = df.copy()
        if self.vectorizer is not None and query.strip():
            qv = self.vectorizer.transform([query])
            sims = cosine_similarity(qv, self.vectorizer.transform(d["_text"].fillna("").astype(str))).ravel()
        else: sims = [0.0] * len(d)
        rating = pd.to_numeric(d.get("review_rating"), errors="coerce").fillna(0).clip(0,10) / 10
        reviews = pd.to_numeric(d.get("review_count"), errors="coerce").fillna(0).clip(0,100) / 100
        amenity_cols = [f"_{AMENITY_COLUMNS[norm(p)]}" for p in (preferences or []) if norm(p) in AMENITY_COLUMNS and f"_{AMENITY_COLUMNS[norm(p)]}" in d]
        amenity = d[amenity_cols].mean(axis=1) if amenity_cols else 0.0
        type_match = d["_type_match"] if "_type_match" in d else 0.0
        d["recommendation_score"] = (0.55*pd.Series(sims,index=d.index) + 0.20*rating + 0.10*reviews + 0.10*amenity + 0.05*type_match).clip(0,1)
        return d.sort_values("recommendation_score", ascending=False)

    def _target_place(self, city, purpose=None, attraction=None):
        if self.places.empty: return None
        p = self.places.copy()
        p = p[p["city"].map(norm) == norm(city)]
        if p.empty: return None
        query = norm(attraction or "")
        if query:
            exact = p[p["name"].map(norm).str.contains(re.escape(query), na=False)]
            if not exact.empty: return exact.iloc[0].to_dict()
        if norm(city) == "ujjain":
            # Ujjain's default anchor is Mahakaleshwar unless another landmark was selected.
            temple = p[p["name"].map(norm).str.contains("mahakaleshwar", na=False)]
            if not temple.empty: return temple.iloc[0].to_dict()
        if norm(city) == "goa" and (not attraction or "beach" in norm(attraction)):
            beaches = p[p["category"].map(norm).str.contains("beach", na=False)]
            if not beaches.empty: return beaches.iloc[0].to_dict()
        if purpose and norm(purpose) in {"temple darshan","religious visit","pilgrimage"}:
            temple = p[p["category"].map(norm).str.contains("religious|temple|place of worship", na=False)]
            if not temple.empty: return temple.iloc[0].to_dict()
        return None

    def recommend_stays(self, city, property_type=None, preferences=None, purpose=None, attraction=None, radius_km=1, limit=10):
        d = self.stays[self.stays["city"].map(norm) == norm(city)].copy()
        if d.empty: return []
        # City-aware guardrails: never let irrelevant landscape preferences
        # influence a destination where that experience is not appropriate.
        effective_preferences = list(preferences or [])
        if norm(city) in {"ujjain", "goa", "varanasi", "haridwar", "ayodhya", "mathura", "vrindavan"}:
            effective_preferences = [p for p in effective_preferences if norm(p) not in {"mountain view", "mountain views"}]
        if norm(city) not in {"goa", "puri", "varkala", "kovalam", "gokarna", "alleppey"}:
            effective_preferences = [p for p in effective_preferences if norm(p) not in {"beach access", "near beach", "beachfront"}]
        d = self._dedupe(d)
        d = self._preference_filter(d, property_type, effective_preferences)
        query = " ".join([str(purpose or ""), str(attraction or ""), str(property_type or ""), " ".join(effective_preferences)])
        d = self._score(d, query, effective_preferences, property_type)
        target = self._target_place(city, purpose, attraction)
        if target:
            d["distance_km"] = d.apply(lambda r: haversine_km(r.get("latitude"),r.get("longitude"),target.get("latitude"),target.get("longitude")),axis=1)
            radius = max(float(radius_km or 1), 0.1)
            # Distance is a transparent ranking factor; hard radius filter only when requested.
            d["distance_score"] = d["distance_km"].apply(lambda x: max(0.0, 1.0-(x/radius)) if x is not None else 0.0)
            d["recommendation_score"] = (0.60*d["recommendation_score"] + 0.40*d["distance_score"]).clip(0,1)
            # Prefer stays inside the selected radius. If the dataset has none,
            # return the nearest geocoded alternatives instead of a blank screen.
            inside = d[d["distance_km"].notna() & (d["distance_km"] <= radius)]
            if not inside.empty:
                d = inside
            d = d.sort_values(["distance_km","recommendation_score"], ascending=[True,False], na_position="last")
            for c in ["_text","_type_match"]: d.drop(columns=[c],errors="ignore",inplace=True)
            d.attrs["target_place"] = target
        else:
            d = d.sort_values("recommendation_score",ascending=False)
            for c in ["_text","_type_match"]: d.drop(columns=[c],errors="ignore",inplace=True)
        d = d.head(limit).copy()
        d = d.where(pd.notna(d), None)
        return d.to_dict("records")

    def recommend_destinations(self, property_type=None, preferences=None, purpose=None, limit=10):
        if self.stays.empty: return []
        d = self._dedupe(self.stays)
        d = self._preference_filter(d, property_type, preferences or [])
        d = self._score(d, " ".join([str(purpose or ""),str(property_type or "")," ".join(preferences or [])]), preferences or [], property_type)
        grouped = d.groupby("destination_id",dropna=True).agg(match_score=("recommendation_score","mean"),matching_stays=("property_name","nunique")).reset_index()
        out = grouped.merge(self.destinations,on="destination_id",how="left").sort_values("match_score",ascending=False).head(limit)
        return out.where(pd.notna(out),None).to_dict("records")

    def recommend_complete_trip(self, property_type=None, preferences=None, purpose=None, limit=5):
        destinations = self.recommend_destinations(property_type,preferences,purpose,limit)
        results=[]
        for dest in destinations:
            stays=self.recommend_stays(dest.get("city"),property_type,preferences,purpose=purpose,limit=3)
            results.append({"destination":dest,"recommended_stays":stays})
        return results
