ROAMLY FRONTEND INSTALLATION

Copy these folders into your existing travel_stay_assistant project:
- templates/index.html
- static/css/style.css
- static/js/app.js

Keep your existing app.py, recommendation_engine.py and travel_stay.db.
Run from the project root:
  .\.venv\Scripts\python.exe app.py

Open http://127.0.0.1:5000

This frontend uses your existing endpoints:
  GET  /api/destinations?q=
  POST /api/recommend

Expected mode values:
  destination_to_stay
  stay_to_destination
  complete_trip

The data note in the UI clarifies that the accommodation dataset is historical,
not live prices, availability, or confirmed bookings.
