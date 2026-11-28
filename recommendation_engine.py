"""Roamly recommendation engine with explainable weighted-fit scoring."""
from __future__ import annotations
import math, re, sqlite3, unicodedata
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "travel_stay.db"
AMENITY_COLUMNS = {
    "wifi": "has_wifi", "wi fi": "has_wifi", "wi-fi": "has_wifi", "parking": "has_parking",
    "restaurant": "has_restaurant", "pool": "has_pool", "swimming pool": "has_pool",
    "air conditioning": "has_air_conditioning", "ac": "has_air_conditioning"
}
# Weights sum to 100. A factor is only counted when the request asks for it and
# the individual property has enough data to evaluate it.
WEIGHTS = {"location":30, "preference":25, "property_type":15, "rating":15, "amenities":10, "review_confidence":5}

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
        text_cols = [c for c in ["property_name","property_type","hotel_brand","hotel_description","hotel_facilities","room_type","city","address"] if c in self.stays]
        self.stays["_text"] = self.stays[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=12000, ngram_range=(1,2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(self.stays["_text"].tolist())

    def _dedupe(self, df):
        if df.empty: return df
        d=df.copy()
        d["_property_key"] = d.apply(lambda r: "|".join((norm(r.get("property_name")),norm(r.get("city")))),axis=1)
        rating=pd.to_numeric(d.get("review_rating"),errors="coerce").fillna(0)
        reviews=pd.to_numeric(d.get("review_count"),errors="coerce").fillna(0).clip(upper=10000)
        d["_quality"]=rating*100+reviews/100
        # Prefer the row with the strongest verified property metadata and image.
        d["_quality"] += d.get("image_url",pd.Series(index=d.index,dtype=object)).fillna("").astype(str).str.startswith(("http://","https://")).astype(int)*1000
        return d.sort_values("_quality",ascending=False).drop_duplicates("_property_key").drop(columns=["_property_key","_quality"],errors="ignore")

    def _target_place(self, city, purpose=None, attraction=None):
        if self.places.empty: return None
        p=self.places[self.places["city"].map(norm)==norm(city)].copy()
        if p.empty:return None
        query=norm(attraction or "")
        if query and query not in {"no specific attraction","no specific area"}:
            exact=p[p["name"].map(norm).str.contains(re.escape(query),na=False)]
            if not exact.empty:return exact.iloc[0].to_dict()
            # match a key phrase in the user's selected attraction
            words=[w for w in query.split() if len(w)>3]
            if words:
                scored=p["name"].map(lambda n:sum(w in norm(n) for w in words))
                if scored.max()>0:return p.loc[scored.idxmax()].to_dict()
        if norm(city)=="ujjain":
            temple=p[p["name"].map(norm).str.contains("mahakaleshwar",na=False)]
            if not temple.empty:return temple.iloc[0].to_dict()
        if norm(city)=="goa" and (not attraction or "beach" in norm(attraction)):
            beaches=p[p["category"].map(norm).str.contains("beach",na=False)]
            if not beaches.empty:return beaches.iloc[0].to_dict()
        if purpose and norm(purpose) in {"temple darshan","religious visit","pilgrimage","mahakaleshwar darshan","other temple visits"}:
            temple=p[p["category"].map(norm).str.contains("temple|religious|worship",na=False)]
            if not temple.empty:return temple.iloc[0].to_dict()
        return None

    def _score(self, df, query, preferences, property_type, target=None, radius_km=1):
        if df.empty:return df
        d=df.copy()
        if self.vectorizer is not None and query.strip():
            qv=self.vectorizer.transform([query]); sims=cosine_similarity(qv,self.vectorizer.transform(d["_text"].fillna("").astype(str))).ravel()
        else:sims=[None]*len(d)
        pref_amenities=[(p,AMENITY_COLUMNS[norm(p)]) for p in (preferences or []) if norm(p) in AMENITY_COLUMNS]
        other_prefs=[p for p in (preferences or []) if norm(p) not in AMENITY_COLUMNS]
        scores=[]; coverages=[]; breakdowns=[]
        for pos,(_,row) in enumerate(d.iterrows()):
            factors={}
            # Proximity: only available when both property and target coordinates exist.
            if target and all(pd.notna(target.get(k)) for k in ("latitude","longitude")) and pd.notna(row.get("latitude")) and pd.notna(row.get("longitude")):
                dist=haversine_km(row.get("latitude"),row.get("longitude"),target.get("latitude"),target.get("longitude"))
                if dist is not None:
                    factors["location"]=(max(0.0,1.0-dist/max(float(radius_km or 1),0.5)),WEIGHTS["location"])
            # Text relevance represents purpose, attraction and non-amenity preferences.
            if query.strip() and str(row.get("_text") or "").strip():
                factors["preference"]=(float(sims[pos] or 0),WEIGHTS["preference"])
            if property_type and pd.notna(row.get("property_type")) and str(row.get("property_type")).strip():
                wanted=norm(property_type); actual=norm(row.get("property_type"))
                factors["property_type"]=(1.0 if wanted in actual or actual in wanted else 0.0,WEIGHTS["property_type"])
            rating=pd.to_numeric(pd.Series([row.get("review_rating")]),errors="coerce").iloc[0]
            if pd.notna(rating): factors["rating"]=(max(0,min(10,float(rating)))/10,WEIGHTS["rating"])
            # Evaluate only amenity fields actually populated for this property.
            known=[]
            for pref,col in pref_amenities:
                value=row.get(col)
                if pd.notna(value): known.append(1.0 if str(value).strip().lower() in {"1","1.0","true","yes"} else 0.0)
            if known:factors["amenities"]=(sum(known)/len(known),WEIGHTS["amenities"])
            reviews=pd.to_numeric(pd.Series([row.get("review_count")]),errors="coerce").iloc[0]
            if pd.notna(reviews):
                # Log curve rewards a meaningful review base without letting volume dominate.
                confidence=min(1.0,math.log1p(max(0,float(reviews)))/math.log1p(500))
                factors["review_confidence"]=(confidence,WEIGHTS["review_confidence"])
            denom=sum(w for _,w in factors.values())
            score=sum(v*w for v,w in factors.values())/denom if denom else 0.0
            coverage=denom/100.0
            scores.append(score);coverages.append(coverage)
            breakdowns.append({k:round(v,3) for k,(v,_) in factors.items()})
        d["recommendation_score"]=scores
        d["score_coverage"]=coverages
        d["score_breakdown"]=breakdowns
        return d.sort_values("recommendation_score",ascending=False)

    def recommend_stays(self, city, property_type=None, preferences=None, purpose=None, attraction=None, radius_km=1, limit=10):
        d=self.stays[self.stays["city"].map(norm)==norm(city)].copy()
        if d.empty:return []
        effective=list(preferences or [])
        if norm(city) in {"ujjain","goa","varanasi","haridwar","ayodhya","mathura","vrindavan"}:
            effective=[p for p in effective if norm(p) not in {"mountain view","mountain views"}]
        if norm(city) not in {"goa","puri","varkala","kovalam","gokarna","alleppey"}:
            effective=[p for p in effective if norm(p) not in {"beach access","near beach","beachfront"}]
        d=self._dedupe(d)
        target=self._target_place(city,purpose,attraction)
        query=" ".join([str(purpose or ""),str(attraction or ""),str(property_type or "")," ".join(effective)])
        d=self._score(d,query,effective,property_type,target,radius_km)
        if target:
            d["distance_km"]=d.apply(lambda r:haversine_km(r.get("latitude"),r.get("longitude"),target.get("latitude"),target.get("longitude")),axis=1)
            radius=max(float(radius_km or 1),0.5)
            inside=d[d["distance_km"].notna() & (d["distance_km"]<=radius)]
            if not inside.empty:d=inside
            d=d.sort_values(["recommendation_score","distance_km"],ascending=[False,True],na_position="last")
            d.attrs["target_place"]=target
        else:d=d.sort_values("recommendation_score",ascending=False)
        d=d.head(limit).copy().drop(columns=["_text"],errors="ignore")
        d=d.where(pd.notna(d),None)
        return d.to_dict("records")

    def recommend_destinations(self, property_type=None, preferences=None, purpose=None, limit=10):
        if self.stays.empty:return []
        d=self._dedupe(self.stays)
        d["_text"]=d.get("_text","")
        d=self._score(d," ".join([str(purpose or ""),str(property_type or "")," ".join(preferences or [])]),preferences or [],property_type)
        grouped=d.groupby("destination_id",dropna=True).agg(match_score=("recommendation_score","mean"),matching_stays=("property_name","nunique"),score_coverage=("score_coverage","mean")).reset_index()
        out=grouped.merge(self.destinations,on="destination_id",how="left").sort_values("match_score",ascending=False).head(limit)
        return out.where(pd.notna(out),None).to_dict("records")

    def recommend_complete_trip(self, property_type=None, preferences=None, purpose=None, limit=5):
        destinations=self.recommend_destinations(property_type,preferences,purpose,limit); results=[]
        for dest in destinations:
            stays=self.recommend_stays(dest.get("city"),property_type,preferences,purpose=purpose,limit=3)
            results.append({"destination":dest,"recommended_stays":stays})
        return results
