"""Import new accommodation CSV rows into the existing stays table safely.

Usage: python scripts/import_stays.py data/raw/new_stays.csv
Accepted common column names: property_name/name, city, state, country, latitude/lat,
longitude/lon, property_type/type, hotel_description/description, hotel_facilities/amenities.
"""
from __future__ import annotations
import argparse, hashlib, re, sqlite3, sys, unicodedata
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; DB=ROOT/"travel_stay.db"

def norm(x):
    x=unicodedata.normalize("NFKD",str(x or "")); x="".join(c for c in x if not unicodedata.combining(c)).casefold()
    return re.sub(r"[^a-z0-9]+"," ",x).strip()
def val(row,*keys):
    for k in keys:
        if k in row and pd.notna(row[k]) and str(row[k]).strip(): return row[k]
    return None
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("csv"); args=ap.parse_args()
    src=Path(args.csv)
    if not src.is_absolute(): src=ROOT/src
    frame=pd.read_csv(src,low_memory=False)
    if not any(k in frame.columns for k in ("property_name","name")) or "city" not in frame.columns:
        raise SystemExit("CSV needs at least property_name (or name) and city columns.")
    with sqlite3.connect(DB) as c:
        c.execute("PRAGMA foreign_keys=ON")
        destinations={norm(city):did for did,city in c.execute("SELECT destination_id,city FROM destinations")}
        existing=set()
        for name,city,address in c.execute("SELECT property_name,city,address FROM stays"):
            existing.add((norm(name),norm(city),norm(address)[:55] if address else ""))
        added=dupes=unmapped=0
        for _,r in frame.iterrows():
            row=r.to_dict(); name=val(row,"property_name","name"); city=val(row,"city")
            if not name or not city: continue
            state=val(row,"state"); country=val(row,"country") or "India"; address=val(row,"address")
            key=(norm(name),norm(city),norm(address)[:55] if address else "")
            if key in existing: dupes+=1; continue
            did=destinations.get(norm(city)); status="csv_exact_city" if did else "unmatched_city"
            if not did: unmapped+=1
            raw_id=val(row,"property_id","id","osm_id")
            record_key="import:"+hashlib.sha1((norm(name)+"|"+norm(city)+"|"+norm(address)).encode()).hexdigest()
            def num(*ks):
                x=val(row,*ks)
                try: return float(x) if x is not None else None
                except (ValueError,TypeError): return None
            facilities=val(row,"hotel_facilities","amenities","facilities")
            text=str(facilities or "").casefold()
            flags=[int(bool(re.search(p,text))) for p in (r"wi\s*fi",r"parking",r"restaurant",r"pool",r"air conditioning|a/c")]
            c.execute("""INSERT OR IGNORE INTO stays(record_key,property_id,property_name,destination_id,city,state,country,mapping_status,
              property_type,hotel_brand,hotel_stars,review_rating,review_count,cleanliness_rating,room_type,room_count,hotel_description,
              hotel_facilities,has_wifi,has_parking,has_restaurant,has_pool,has_air_conditioning,latitude,longitude,address,pageurl)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
              (record_key,str(raw_id) if raw_id is not None else None,str(name),did,str(city),state,country,status,
               val(row,"property_type","type"),val(row,"hotel_brand","brand"),num("hotel_stars","stars"),num("review_rating","rating"),
               num("review_count","reviews"),num("cleanliness_rating"),val(row,"room_type"),num("room_count"),
               val(row,"hotel_description","description"),facilities,*flags,num("latitude","lat"),num("longitude","lon","lng"),address,val(row,"pageurl","url","website")))
            if c.execute("SELECT changes()").fetchone()[0]: added+=1; existing.add(key)
            else: dupes+=1
        c.commit()
    print(f"Import complete: {added} added, {dupes} duplicates skipped, {unmapped} city names need mapping review.")
if __name__=="__main__": main()
