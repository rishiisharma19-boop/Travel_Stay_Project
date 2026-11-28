"""Cached, city-centred OSM import helpers. Public Overpass is used conservatively."""
from __future__ import annotations
import json, re, sqlite3, time
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]
DB=ROOT/"travel_stay.db"
CACHE=ROOT/"data"/"osm_cache"
OVERPASS="https://overpass-api.de/api/interpreter"
HEADERS={"User-Agent":"RoamlyTravelAssistant/1.0 (educational project; contact: project-maintainer)"}

CITY_CENTRES={
 "ujjain":(23.1828,75.7772), "varanasi":(25.3176,82.9739), "haridwar":(29.9457,78.1642),
 "rishikesh":(30.0869,78.2676), "ayodhya":(26.7922,82.1998), "mathura":(27.4924,77.6737),
 "vrindavan":(27.5810,77.7000), "jaipur":(26.9124,75.7873), "agra":(27.1767,78.0081),
 "delhi":(28.6139,77.2090), "new delhi":(28.6139,77.2090), "greater noida":(28.4744,77.5040),
 "manali":(32.2396,77.1887), "shimla":(31.1048,77.1734), "amritsar":(31.6340,74.8723),
 "puri":(19.8135,85.8312), "tirupati":(13.6288,79.4192), "dwarka":(22.2442,68.9685),
 "ooty":(11.4102,76.6950), "goa":(15.4909,73.8278),
}

def init_tables(db=DB):
    with sqlite3.connect(db) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS places (
          place_id INTEGER PRIMARY KEY AUTOINCREMENT, osm_type TEXT NOT NULL, osm_id TEXT NOT NULL,
          name TEXT NOT NULL, category TEXT, city TEXT NOT NULL, state TEXT, country TEXT DEFAULT 'India',
          latitude REAL, longitude REAL, tags_json TEXT, source TEXT DEFAULT 'OpenStreetMap',
          updated_at TEXT DEFAULT CURRENT_TIMESTAMP, UNIQUE(osm_type,osm_id))""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_places_city ON places(city)")
        # Store only a property-specific image URL when OSM explicitly supplies one.
        try:
            cols={r[1] for r in c.execute("PRAGMA table_info(stays)")}
            if "image_url" not in cols: c.execute("ALTER TABLE stays ADD COLUMN image_url TEXT")
        except sqlite3.OperationalError:
            pass
        c.execute("""CREATE TABLE IF NOT EXISTS stay_place_distances (
          stay_id INTEGER NOT NULL, place_id INTEGER NOT NULL, distance_km REAL NOT NULL,
          calculated_at TEXT DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(stay_id,place_id),
          FOREIGN KEY(stay_id) REFERENCES stays(stay_id), FOREIGN KEY(place_id) REFERENCES places(place_id))""")

def get_city_centre(city):
    key=re.sub(r"[^a-z0-9 ]","",city.casefold()).strip()
    if key in CITY_CENTRES: return CITY_CENTRES[key]
    # One user-triggered lookup, cached permanently. This is not used for autocomplete.
    CACHE.mkdir(parents=True,exist_ok=True)
    geo_file=CACHE/"geocoded_city_centres.json"
    cached=json.loads(geo_file.read_text(encoding="utf-8")) if geo_file.exists() else {}
    if key in cached: return tuple(cached[key])
    response=requests.get("https://nominatim.openstreetmap.org/search",params={"city":city,"country":"India","format":"jsonv2","limit":1},headers=HEADERS,timeout=20)
    response.raise_for_status(); found=response.json()
    if not found: raise ValueError(f"Could not find a city centre for {city!r}. Add verified coordinates to CITY_CENTRES.")
    point=(float(found[0]["lat"]),float(found[0]["lon"]))
    cached[key]=point; geo_file.write_text(json.dumps(cached),encoding="utf-8")
    return point

def _cache_file(city): return CACHE/(re.sub(r"[^a-z0-9]+","_",city.casefold()).strip("_")+".json")

def fetch_city_osm(city, radius_m=8000, refresh=False, db=DB):
    """Fetch a compact POI + accommodation subset around a configured city centre."""
    init_tables(db); CACHE.mkdir(parents=True,exist_ok=True)
    city_key = re.sub(r"[^a-z0-9 ]", "", city.casefold()).strip()
    if radius_m == 8000 and city_key == "goa": radius_m = 30000
    if radius_m == 8000 and city_key == "ooty": radius_m = 12000
    path=_cache_file(city)
    if path.exists() and not refresh:
        payload=json.loads(path.read_text(encoding="utf-8"))
    else:
        lat,lon=get_city_centre(city)
        centres=[(lat,lon)]
        # Goa is a state, so cover both North Goa (Panaji) and South Goa (Margao).
        if city_key == "goa": centres=[(15.4909,73.8278),(15.2832,73.9862)]
        blocks=[]
        for clat,clon in centres:
            blocks.append(f''' node["amenity"="place_of_worship"](around:{radius_m},{clat},{clon});
 way["amenity"="place_of_worship"](around:{radius_m},{clat},{clon});
 relation["amenity"="place_of_worship"](around:{radius_m},{clat},{clon});
 node["tourism"~"attraction|museum|viewpoint|zoo|theme_park|hotel|hostel|guest_house|apartment|motel|chalet|camp_site"](around:{radius_m},{clat},{clon});
 way["tourism"~"attraction|museum|viewpoint|zoo|theme_park|hotel|hostel|guest_house|apartment|motel|chalet|camp_site"](around:{radius_m},{clat},{clon});
 relation["tourism"~"attraction|museum|viewpoint|zoo|theme_park|hotel|hostel|guest_house|apartment|motel|chalet|camp_site"](around:{radius_m},{clat},{clon});
 node["natural"="beach"](around:{radius_m},{clat},{clon}); way["natural"="beach"](around:{radius_m},{clat},{clon});
 node["leisure"="park"](around:{radius_m},{clat},{clon}); way["leisure"="park"](around:{radius_m},{clat},{clon});''')
        query="[out:json][timeout:90];(\n"+"\n".join(blocks)+"\n);out center tags;"
        response=requests.post(OVERPASS,data={"data":query},headers=HEADERS,timeout=60)
        response.raise_for_status(); payload=response.json()
        path.write_text(json.dumps(payload,ensure_ascii=False),encoding="utf-8")
    rows=[]
    for el in payload.get("elements",[]):
        tags=el.get("tags",{}); name=tags.get("name")
        if not name: continue
        lat=el.get("lat",(el.get("center") or {}).get("lat")); lon=el.get("lon",(el.get("center") or {}).get("lon"))
        if lat is None or lon is None: continue
        if tags.get("amenity")=="place_of_worship": category="temple" if tags.get("religion")=="hindu" else "place of worship"
        else: category=tags.get("tourism") or tags.get("historic") or tags.get("natural") or tags.get("amenity") or "attraction"
        rows.append((el.get("type"),str(el.get("id")),name,category,city,None,"India",float(lat),float(lon),json.dumps(tags,ensure_ascii=False)))
    with sqlite3.connect(db) as c:
        c.execute("PRAGMA foreign_keys=ON")
        state_name = "Goa" if city.casefold() == "goa" else ("Tamil Nadu" if city.casefold() == "ooty" else None)
        c.execute("INSERT INTO destinations(city,state,country) SELECT ?,?,'India' WHERE NOT EXISTS (SELECT 1 FROM destinations WHERE lower(city)=lower(?) AND lower(country)='india')", (city,state_name,city))
        c.executemany("""INSERT INTO places(osm_type,osm_id,name,category,city,state,country,latitude,longitude,tags_json)
          VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(osm_type,osm_id) DO UPDATE SET name=excluded.name,category=excluded.category,
          latitude=excluded.latitude,longitude=excluded.longitude,tags_json=excluded.tags_json,updated_at=CURRENT_TIMESTAMP""",rows)
        destinations={re.sub(r"[^a-z0-9]+"," ",str(n).casefold()).strip():did for did,n in c.execute("SELECT destination_id,city FROM destinations")}
        accommodation_types={"hotel","hostel","guest_house","apartment","motel","chalet","camp_site"}
        imported=0
        for typ,oid,name,category,city_name,state,country,lat,lon,tags_json in rows:
            tags=json.loads(tags_json)
            prop_type=tags.get("tourism")
            if prop_type not in accommodation_types: continue
            city_key=re.sub(r"[^a-z0-9]+"," ",city.casefold()).strip()
            did=destinations.get(city_key)
            record_key=f"osm:{typ}:{oid}"
            c.execute("""INSERT OR IGNORE INTO stays(record_key,property_id,property_name,destination_id,city,state,country,mapping_status,
              property_type,hotel_brand,hotel_stars,review_rating,review_count,cleanliness_rating,room_type,room_count,hotel_description,
              hotel_facilities,has_wifi,has_parking,has_restaurant,has_pool,has_air_conditioning,latitude,longitude,address,pageurl,image_url)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
              (record_key,f"osm:{typ}:{oid}",name,did,city,state_name,"India","osm_imported",prop_type,tags.get("brand"),
               None,None,None,None,None,None,tags.get("description"),", ".join(k for k in ["internet_access" if tags.get("internet_access") else None,"parking" if tags.get("parking") else None] if k),
               int(tags.get("internet_access") in ("yes","wlan")),int(tags.get("parking") in ("yes","private")),int(tags.get("restaurant") in ("yes","true")),0,int(tags.get("air_conditioning")=="yes"),lat,lon,tags.get("addr:full"),f"https://www.openstreetmap.org/{typ}/{oid}",tags.get("image")))
            imported += c.execute("SELECT changes()").fetchone()[0]
    return {"city":city,"records_received":len(rows),"osm_accommodations_added":imported,"cache_file":str(path)}
