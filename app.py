from pathlib import Path
import json
import sqlite3

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "travel_stay.db"
REPORT_PATH = ROOT / "data" / "processed" / "preprocessing_report.json"

st.set_page_config(page_title="Travel & Stay Assistant", page_icon="🧭", layout="wide")
st.title("Travel & Stay Decision Assistant")
st.caption("MVP · Dataset readiness and stay catalogue")

if not DB_PATH.exists() or not REPORT_PATH.exists():
    st.error("Preprocessed data not found. Run `python scripts/preprocess.py` from the project root first.")
    st.stop()

report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
with sqlite3.connect(DB_PATH) as conn:
    stays = pd.read_sql_query("SELECT * FROM stays", conn)
    destinations = pd.read_sql_query("SELECT * FROM destinations", conn)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Source properties", f"{report['source_rows']:,}")
c2.metric("Usable stay records", f"{report['cleaned_stay_rows']:,}")
c3.metric("Mapped stays", f"{report['mapped_stays']:,}")
c4.metric("Canonical cities", f"{report['destination_rows']:,}")

st.info("This screen validates the data pipeline. Recommendation modes and the conversational questionnaire are the next implementation stage.")

tab1, tab2, tab3 = st.tabs(["Data quality", "Explore stays", "City mapping"])
with tab1:
    st.subheader("Preprocessing summary")
    status = pd.DataFrame(
        [(k, v) for k, v in report["mapping_status_counts"].items()],
        columns=["Mapping status", "Stay records"],
    )
    left, right = st.columns(2)
    with left:
        st.write("**City mapping status**")
        st.bar_chart(status.set_index("Mapping status"))
    with right:
        st.write("**Data limitations**")
        st.write("- No reliable nightly price field")
        st.write("- Historical crawl date: 2016-09-01")
        st.write("- Unresolved city links are retained for review")
        st.write("- Property type is stored as a source code")
    st.subheader("Preprocessing report")
    st.json(report)

with tab2:
    st.subheader("Stay catalogue")
    city_options = ["All"] + sorted(stays["city"].dropna().astype(str).unique().tolist())
    selected_city = st.selectbox("Filter by source city", city_options)
    min_rating = st.slider("Minimum guest review rating", 0.0, 10.0, 0.0, 0.5)
    view = stays.copy()
    if selected_city != "All":
        view = view[view["city"] == selected_city]
    view = view[view["review_rating"].fillna(0) >= min_rating]
    st.caption(f"Showing {len(view):,} records")
    columns = ["property_name", "city", "state", "hotel_stars", "review_rating", "review_count", "room_type", "mapping_status"]
    st.dataframe(view[columns].sort_values("review_rating", ascending=False), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Canonical city mapping")
    mapped = stays[stays["destination_id"].notna()]
    unmapped = stays[stays["destination_id"].isna()]
    st.write(f"Mapped: **{len(mapped):,}** · Unmapped/ambiguous: **{len(unmapped):,}**")
    city_summary = (stays.groupby(["city", "state", "mapping_status"], dropna=False)
                    .size().reset_index(name="stay_count")
                    .sort_values("stay_count", ascending=False))
    st.dataframe(city_summary, use_container_width=True, hide_index=True)
    st.download_button("Download mapping review", unmapped[["property_name", "city", "state", "mapping_status"]].to_csv(index=False), "unmapped_stays.csv", "text/csv")
