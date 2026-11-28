from __future__ import annotations
import json, math, sqlite3
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from recommendation_engine import TravelRecommendationEngine, norm, haversine_km
from services.osm_service import init_tables, fetch_city_osm

ROOT=Path(__file__).resolve().parent
DB_PATH=ROOT/"travel_stay.db"
app=Flask(__name__)
app.config["JSON_SORT_KEYS"]=False
engine=None

def db(): return sqlite3.connect(DB_PATH)
def seed_popular_destinations():
    """Add searchable destination entries without deleting the user's existing data."""
    seeds = [
        ("Goa", "Goa", "India"),
        ("Ooty", "Tamil Nadu", "India"),
    ]
    with db() as con:
        for city, state, country in seeds:
            con.execute("""INSERT INTO destinations(city,state,country)
                SELECT ?,?,? WHERE NOT EXISTS
                (SELECT 1 FROM destinations WHERE lower(city)=lower(?) AND lower(country)=lower(?))""",
                (city, state, country, city, country))
        # The source sample already contains Ooty properties with NULL destination_id.
        did = con.execute("SELECT destination_id FROM destinations WHERE lower(city)='ooty' AND lower(country)='india' LIMIT 1").fetchone()
        if did:
            con.execute("UPDATE stays SET destination_id=? WHERE lower(city)='ooty' AND destination_id IS NULL", (did[0],))

def refresh_engine():
    global engine
    engine=TravelRecommendationEngine(DB_PATH)

def clean(value):
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if hasattr(value,"item"):
        try: value=value.item()
        except Exception: pass
    if isinstance(value,float) and (math.isnan(value) or math.isinf(value)): return None
    return value

@app.before_request
def ensure_schema():
    init_tables(DB_PATH)
    seed_popular_destinations()
    if engine is None: refresh_engine()

@app.get("/")
def home(): return render_template("index.html")

@app.get("/api/health")
def health():
    with db() as c:
        stays=c.execute("SELECT COUNT(*) FROM stays").fetchone()[0]
        destinations=c.execute("SELECT COUNT(*) FROM destinations").fetchone()[0]
        places=c.execute("SELECT COUNT(*) FROM places").fetchone()[0]
    return jsonify({"status":"ok","stays":stays,"destinations":destinations,"places":places})

@app.get("/api/destinations")
def destinations():
    q=request.args.get("q","").strip()
    with db() as c:
        rows=c.execute("SELECT destination_id,city,state,country FROM destinations WHERE city LIKE ? ORDER BY city LIMIT 15",(f"%{q}%",)).fetchall()
    return jsonify([{"destination_id":r[0],"city":r[1],"state":r[2],"country":r[3]} for r in rows])

@app.get("/api/questions")
def questions():
    city=request.args.get("city","").strip(); mode=request.args.get("mode","destination_to_stay")
    purpose=request.args.get("purpose","").strip()
    c=norm(city)
    if not city or mode=="stay_to_destination": return jsonify({"questions":[]})
    if c=="ujjain":
        qs=[{"id":"purpose","label":"What brings you to Ujjain?","type":"select","options":["Mahakaleshwar darshan","Other temple visits","Heritage and sightseeing","Family trip","Business","Relaxation"],"default":"Mahakaleshwar darshan"}]
        if norm(purpose) in {"", "mahakaleshwar darshan", "other temple visits"}:
            opts=["Mahakaleshwar Temple","Harsiddhi Mata Temple","Kal Bhairav Temple","Ram Ghat"]
            default="Mahakaleshwar Temple" if norm(purpose)!="other temple visits" else None
            qs.append({"id":"attraction","label":"Which temple or sacred place should be nearby?","type":"select","options":opts,"default":default})
            qs.append({"id":"radius_km","label":"Maximum distance from that place","type":"select","options":[{"label":"Within 500 metres","value":0.5},{"label":"Within 1 km","value":1},{"label":"Within 3 km","value":3},{"label":"Flexible, nearest first","value":10}],"default":1})
        else:
            qs.append({"id":"attraction","label":"Which area or landmark matters?","type":"select","options":["Ram Ghat","Ujjain city centre","No specific landmark"]})
        return jsonify({"questions":qs,"context":"Ujjain is treated as a temple-focused destination. Mountain-view and beach preferences are hidden because they are not relevant to this city."})
    if c=="goa":
        qs=[{"id":"purpose","label":"What kind of Goa trip are you planning?","type":"select","options":["Beach holiday","Nightlife and cafes","Family vacation","Relaxation","Sightseeing","Workation"],"default":"Beach holiday"}]
        qs.append({"id":"attraction","label":"Which part of Goa would you prefer?","type":"select","options":["North Goa beaches","South Goa beaches","Panaji and Old Goa","No specific area"]})
        qs.append({"id":"radius_km","label":"How close should your stay be to the beach or area?","type":"select","options":[{"label":"Within 1 km","value":1},{"label":"Within 3 km","value":3},{"label":"Within 5 km","value":5},{"label":"Flexible","value":20}],"default":5})
        return jsonify({"questions":qs,"context":"Goa recommendations prioritise beach and coastal areas. Goa is a large state, so imported map coverage depends on the selected search area."})
    if c=="ooty":
        qs=[{"id":"purpose","label":"What would you like to experience in Ooty?","type":"select","options":["Nature and mountain views","Tea gardens","Family sightseeing","Relaxing resort stay","Adventure and outdoors"],"default":"Nature and mountain views"}]
        qs.append({"id":"attraction","label":"Which nearby place or area interests you?","type":"select","options":["Ooty Lake","Botanical Garden","Doddabetta Peak","Tea gardens","No specific attraction"]})
        qs.append({"id":"radius_km","label":"Preferred distance from this place","type":"select","options":[{"label":"Within 1 km","value":1},{"label":"Within 3 km","value":3},{"label":"Within 5 km","value":5},{"label":"Flexible","value":10}],"default":3})
        return jsonify({"questions":qs,"context":"Ooty is treated as a hill-station and nature destination, so nature, viewpoints and tea-garden preferences are prioritised."})
    qs=[{"id":"purpose","label":"What is the main purpose of this trip?","type":"select","options":["Sightseeing","Nature and outdoors","Family trip","Business","Relaxation","Religious visit"]}]
    if norm(purpose) in {"religious visit","temple darshan"}:
        label="Which temple or pilgrimage site should be nearby?"
    elif norm(purpose) in {"nature and outdoors","nature outdoors"}:
        label="Which nature area, lake or viewpoint should be nearby?"
    elif norm(purpose)=="business": label="Which business district, venue or station should be nearby?"
    else: label="Any attraction or area you want to stay near?"
    qs.extend([{"id":"attraction","label":label,"type":"text","placeholder":"Enter a place or leave blank"},{"id":"radius_km","label":"Preferred distance (km)","type":"select","options":[{"label":"Within 1 km","value":1},{"label":"Within 3 km","value":3},{"label":"Within 5 km","value":5},{"label":"Flexible","value":10}],"default":3}])
    return jsonify({"questions":qs,"context":"Questions adapt to your destination and trip purpose. Distances are straight-line estimates."})

@app.post("/api/recommend")
def recommend():
    data=request.get_json(silent=True) or {}
    mode=data.get("mode","destination_to_stay")
    city=(data.get("city") or "").strip()
    property_type=data.get("property_type") or None
    preferences=data.get("preferences") or []
    purpose=data.get("purpose") or None
    attraction=data.get("attraction") or None
    radius=float(data.get("radius_km") or 1)
    if mode=="destination_to_stay":
        if not city: return jsonify({"status":"error","message":"Choose a destination."}),400
        results=engine.recommend_stays(city,property_type,preferences,purpose,attraction,radius,10)
        target=engine._target_place(city,purpose,attraction)
        return jsonify(clean({"status":"success","mode":mode,"count":len(results),"results":results,"target_place":target,"radius_km":radius if target else None,"message":"Distances are straight-line estimates, not walking or driving routes." if target else None}))
    if mode=="stay_to_destination":
        results=engine.recommend_destinations(property_type,preferences,purpose,10)
    elif mode=="complete_trip":
        results=engine.recommend_complete_trip(property_type,preferences,purpose,5)
    else: return jsonify({"status":"error","message":"Unknown recommendation mode."}),400
    return jsonify(clean({"status":"success","mode":mode,"count":len(results),"results":results}))

@app.post("/api/osm/import-city")
def osm_import_city():
    data=request.get_json(silent=True) or {}; city=(data.get("city") or "").strip()
    if not city: return jsonify({"status":"error","message":"City is required."}),400
    try:
        result=fetch_city_osm(city, radius_m=int(data.get("radius_m",8000)), refresh=bool(data.get("refresh",False)),db=DB_PATH)
        refresh_engine()
        return jsonify({"status":"success",**result})
    except Exception as e: return jsonify({"status":"error","message":str(e)}),502

@app.get("/api/places")
def places():
    city=request.args.get("city","").strip()
    with db() as c:
        rows=c.execute("SELECT place_id,name,category,latitude,longitude FROM places WHERE city LIKE ? ORDER BY category,name LIMIT 300",(f"%{city}%",)).fetchall()
    return jsonify([{"place_id":r[0],"name":r[1],"category":r[2],"latitude":r[3],"longitude":r[4]} for r in rows])

if __name__=="__main__":
    init_tables(DB_PATH); seed_popular_destinations(); refresh_engine()
    app.run(host="127.0.0.1",port=5000,debug=True)
