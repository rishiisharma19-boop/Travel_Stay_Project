# Roamly upgrade pack (Flask + SQLite)

This is a drop-in upgrade for the Flask version of Roamly. It preserves the existing `travel_stay.db` and the current `destinations` / `stays` tables. New tables (`places`, `stay_place_distances`) are created automatically.

## Files

- `app.py`: Flask routes for the UI, recommendations, adaptive questions, POIs and city OSM import.
- `recommendation_engine.py`: explainable weighted fit score with per-property missing-data adjustment, distance-aware ranking and property deduplication.
- `services/osm_service.py`: cached city-centred Overpass importer, one-time city geocoding fallback, OSM place and accommodation ingestion.
- `scripts/import_stays.py`: safe CSV importer for additional accommodation rows.
- `templates/index.html`, `static/js/app.js`, `static/css/style.css`: adaptive Roamly frontend with Leaflet map.

## Integration into your existing Windows project

1. Stop Flask (`Ctrl+C`). Make a backup first:

   ```powershell
   Copy-Item .\travel_stay.db .\travel_stay_backup_before_roamly_upgrade.db
   Copy-Item .\app.py .\app_before_roamly_upgrade.py
   ```

2. Extract this ZIP into your existing project root `C:\Users\pawan\Desktop\travel_stay_assistant`, choosing **overwrite/merge**. It replaces `app.py`, adds `recommendation_engine.py` and `services`, and replaces `templates/index.html`, `static/js/app.js`, and `static/css/style.css`. Keep your existing `travel_stay.db`, `data/`, `scripts/preprocess.py`, `.venv`, and raw CSVs.

3. Install the added dependency in your existing venv:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

4. Start Flask:

   ```powershell
   .\.venv\Scripts\python.exe app.py
   ```

5. Open `http://127.0.0.1:5000`. Check `http://127.0.0.1:5000/api/health`.

## OSM city import

Choose a destination in Roamly and press **Refresh OpenStreetMap data for this city**. The server fetches a compact radius of mapped POIs and accommodation tags around the city centre and caches the response in `data/osm_cache/`. Ujjain's Mahakaleshwar proximity is the default for temple-oriented searches; select another attraction or distance when needed. If no property lies inside the chosen radius, Roamly returns the nearest geocoded alternatives rather than silently returning nothing.

For cities not in the built-in coordinate list, the importer makes one user-triggered Nominatim city lookup and caches it. It is not used for autocomplete. Do not automate repeated lookups or use the public Nominatim service as a general high-volume geocoder. Public OSM services are shared; use a suitable provider or self-hosted service for sustained production traffic. Display `© OpenStreetMap contributors` (already included in the map UI) and review ODbL obligations before redistributing a derived database.

OSM coverage varies. The imported coordinates and place tags are community-maintained and may be incomplete. Distances are straight-line (Haversine), not road/walking distance. The map tiles are fetched from the standard OSM tile service and must not be bulk-downloaded or prefetched.

## Add more stays from CSV

Put a file such as `data/raw/new_stays.csv` in the project. Minimum columns: `property_name,city`. Recommended columns:

`property_name,city,state,country,property_type,hotel_stars,review_rating,review_count,hotel_description,hotel_facilities,latitude,longitude,address,pageurl,image_url`

Then run:

```powershell
.\.venv\Scripts\python.exe scripts/import_stays.py data/raw/new_stays.csv
```

The importer maps city names only to existing canonical destination names (normalized exact match). Unmatched cities are still imported with `destination_id=NULL` and `mapping_status=unmatched_city`, so they won't appear in destination-linked recommendations until you map them. It skips normalized duplicate property-name + city combinations. Restart Flask after importing so the in-memory TF-IDF model reloads.

Do not append rows blindly to the cleaned CSV or rerun `preprocess.py` just to add hotels: preprocessing rebuilds the database from the original source dataset. Use this importer for incremental additions.

## Important data limitations

- The original Booking.com sample is historical (2016), not current inventory.
- The source has no reliable nightly price column. Roamly does not calculate or promise prices.
- No live room availability or booking confirmation is provided.
- OSM hotel records are supplementary, not a complete accommodation catalogue; they may lack ratings, facilities or booking links.
- `property_type` in the original dataset can be source-coded, so filtering is a soft ranking preference rather than a strict exclusion.


## Destination expansion (Goa and Ooty)
On startup, Roamly adds Goa and Ooty to the destination autocomplete without deleting existing records. Ooty already had historical sample properties with a missing destination mapping; startup now repairs that mapping. Use the OpenStreetMap import button after selecting Goa or Ooty to add mapped accommodation and nearby attraction records. Goa import searches around both Panaji (North Goa) and Margao (South Goa) with a 30 km radius per centre; it is still not a complete verified statewide inventory. Ooty import uses a 12 km radius. OpenStreetMap coverage varies and is not a verified hotel catalogue.

## Booking links
For historical sample properties, Roamly opens a Booking.com search prefilled with the exact property name and city rather than trusting old 2016 session/availability URLs. OSM-only properties open their OpenStreetMap object page unless a specific booking URL is available in OSM tags.


## Context-aware preferences
The Ujjain form hides mountain-view preferences and defaults to temple-focused questions. Goa offers beach-area choices, and Ooty offers hill-station/nature choices. The backend also ignores mountain-view and beach preferences when they are incompatible with the selected city, even if an older browser submits them.


## Property photographs and AI-generated fallback images

Roamly uses a property-specific photograph only when `image_url` contains a direct HTTP(S) image URL. The database migration adds the nullable `stays.image_url` column automatically on startup; existing hotel records are not modified. CSV imports accept `image_url` (or `image`) and OSM accommodation imports use an explicit OSM `image` tag only when it is a URL. If no property-specific image is available, the frontend uses a locally bundled AI-generated contextual illustration for Goa, Ooty, Ujjain, or a general hotel room. Every fallback is labelled **AI illustration** and is not represented as a photo of that property.

Use only image URLs you have permission to use. A hotel image can be added to a CSV row via the `image_url` column. The URL should point directly to an image file and should be served over HTTPS where possible.

## Weighted fit score and missing-data adjustment

The fit score uses these maximum weights: location/proximity 30, request and preference relevance 25, property type 15, guest rating 15, requested amenities 10, and review-count confidence 5. Factors are included only when relevant to the request and supported by that property's data. For each property, available weights are re-normalised to calculate its displayed match percentage; missing fields therefore do not automatically count as zero. The card also displays **Data coverage**, the share of the 100-point model supported by available evidence, so a high score with sparse data is visibly distinguishable from a high score backed by complete data. This is a fit index, not a probability or guarantee.


## Image selection in this update

For each recommended property, Roamly uses a property-specific `image_url` only when one is present as an HTTP(S) URL. When no property photo is supplied, the frontend assigns a shuffled destination-contextual AI-generated illustrative image from a local pool. The fallback assignment avoids repeating the same image within one result list (up to ten results per city). These images are illustrative accommodation concepts, not photographs of the named property. If you add verified property images to `image_url`, those take priority.


## Optional semantic ML upgrade (Roamly v5)

The base application remains compatible with TF-IDF. To enable semantic matching, install the optional model dependency inside the project virtual environment:

```powershell
.\\.venv\\Scripts\\python.exe -m pip install -r requirements-ml.txt
```

On the first recommendation request, Roamly downloads `sentence-transformers/all-MiniLM-L6-v2` and caches it locally. The first run needs an internet connection and may take several minutes. Subsequent runs use the cached model. If the package or model is unavailable, recommendations automatically fall back to TF-IDF so the application remains usable.

When the semantic model is available, preference similarity is blended as 75% sentence-embedding similarity and 25% TF-IDF similarity. The existing weighted fit factors remain: location 30, preferences 25, property type 15, rating 15, amenities 10 and review confidence 5. Factors without sufficient property data are excluded and the available weights are proportionally normalised.

### Interactive fit-score explanation

Hover over a property's fit score to reveal the information icon, then click it. The dialog shows each available factor's normalised score, original weight, points contributed to the final score, the data coverage, and which factor weight was unavailable. Keyboard focus also reveals the information icon; Escape is not currently required to close it, use the close button or backdrop.

The fit score is a ranking aid, not a probability of satisfaction. Data coverage indicates how much of the original factor weight could be evaluated.
