
import sqlite3
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TravelRecommendationEngine:

    def __init__(self, database_path="travel_stay.db"):

        self.database_path = database_path

        self.stays = None
        self.destinations = None

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2)
        )

        self.tfidf_matrix = None

        self.load_data()
        self.train_model()

    # ---------------------------------------
    # 1. LOAD DATA FROM SQLITE
    # ---------------------------------------

    def load_data(self):

        conn = sqlite3.connect(self.database_path)

        self.stays = pd.read_sql_query(
            "SELECT * FROM stays",
            conn
        )

        self.destinations = pd.read_sql_query(
            "SELECT * FROM destinations",
            conn
        )

        conn.close()

        # Convert numeric columns safely
        numeric_columns = [
            "hotel_stars",
            "review_rating",
            "review_count",
            "cleanliness_rating",
            "latitude",
            "longitude"
        ]

        for col in numeric_columns:
            self.stays[col] = pd.to_numeric(
                self.stays[col],
                errors="coerce"
            )

        # Missing text values
        text_columns = [
            "property_name",
            "property_type",
            "hotel_description",
            "hotel_facilities",
            "city",
            "state"
        ]

        for col in text_columns:
            self.stays[col] = (
                self.stays[col].fillna("").astype(str)
            )

        print("Database loaded successfully.")
        print("Stays:", len(self.stays))
        print("Destinations:", len(self.destinations))

    # ---------------------------------------
    # 2. TRAIN TF-IDF MODEL
    # ---------------------------------------

    def train_model(self):

        self.stays["combined_features"] = (
            self.stays["property_name"] + " " +
            self.stays["property_type"] + " " +
            self.stays["hotel_description"] + " " +
            self.stays["hotel_facilities"]
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.stays["combined_features"]
        )

        print("TF-IDF model trained successfully.")

    # ---------------------------------------
    # 3. BUILD USER PREFERENCE QUERY
    # ---------------------------------------

    def build_query(
        self,
        property_type=None,
        preferences=None
    ):

        query_parts = []

        if property_type:
            query_parts.append(property_type)

        if preferences:
            query_parts.extend(preferences)

        return " ".join(query_parts)

    # ---------------------------------------
    # 4. RANK STAYS
    # ---------------------------------------

    def rank_stays(self, candidates, query):

        if candidates.empty:
            return candidates

        query_vector = self.vectorizer.transform([query])

        indices = candidates.index.to_numpy()

        similarities = cosine_similarity(
            self.tfidf_matrix[indices],
            query_vector
        ).flatten()

        candidates = candidates.copy()

        candidates["similarity_score"] = similarities

        # Normalize rating to a 0-1 scale
        ratings = candidates["review_rating"].fillna(0)

        candidates["rating_score"] = (
            ratings.clip(0, 10) / 10
        )

        # Review confidence: avoid overvaluing
        # properties with very few reviews
        review_counts = candidates["review_count"].fillna(0)

        candidates["confidence_score"] = (
            review_counts / (review_counts + 20)
        )

        # Final weighted score
        candidates["recommendation_score"] = (
            0.65 * candidates["similarity_score"] +
            0.25 * candidates["rating_score"] +
            0.10 * candidates["confidence_score"]
        )

        return candidates.sort_values(
            "recommendation_score",
            ascending=False
        )

    # ---------------------------------------
    # 5. MODE 1:
    # DESTINATION TO STAY
    # ---------------------------------------

    def recommend_stays(
        self,
        city,
        property_type=None,
        preferences=None,
        limit=10
    ):

        candidates = self.stays[
            self.stays["city"].str.lower()
            == city.strip().lower()
        ].copy()

        if candidates.empty:
            return []

        # Exclude records without a mapped destination
        candidates = candidates[
            candidates["destination_id"].notna()
        ]

        query = self.build_query(
            property_type,
            preferences
        )

        if not query.strip():
            query = "hotel accommodation comfortable stay"

        ranked = self.rank_stays(candidates, query)

        return ranked.head(limit).to_dict("records")

    # ---------------------------------------git
    # 6. MODE 2:
    # STAY PREFERENCE TO DESTINATION
    # ---------------------------------------

    def recommend_destinations(
        self,
        property_type=None,
        preferences=None,
        limit=10
    ):

        query = self.build_query(
            property_type,
            preferences
        )

        if not query.strip():
            query = "hotel accommodation comfortable stay"

        # Rank all available stays
        candidates = self.stays[
            self.stays["destination_id"].notna()
        ].copy()

        ranked = self.rank_stays(candidates, query)

        # Aggregate the best matching stays by destination
        destination_scores = (
            ranked.groupby("destination_id")
            .agg(
                match_score=("recommendation_score", "mean"),
                matching_stays=("stay_id", "count"),
                city=("city", "first"),
                state=("state", "first"),
                country=("country", "first")
            )
            .reset_index()
        )

        destination_scores = destination_scores.merge(
            self.destinations,
            on="destination_id",
            how="left",
            suffixes=("", "_destination")
        )

        destination_scores = destination_scores.sort_values(
            "match_score",
            ascending=False
        )

        return destination_scores.head(limit).to_dict("records")

    # ---------------------------------------
    # 7. MODE 3:
    # COMPLETE TRIP
    # ---------------------------------------

    def recommend_complete_trip(
        self,
        property_type=None,
        preferences=None,
        limit=5
    ):

        destinations = self.recommend_destinations(
            property_type=property_type,
            preferences=preferences,
            limit=limit
        )

        results = []

        for destination in destinations:

            city = destination.get("city")

            stays = self.recommend_stays(
                city=city,
                property_type=property_type,
                preferences=preferences,
                limit=3
            )

            results.append({
                "destination": destination,
                "recommended_stays": stays
            })

        return results


# ---------------------------------------
# TEST THE ENGINE
# ---------------------------------------

if __name__ == "__main__":

    engine = TravelRecommendationEngine()

    print("\nMODE 1: Destination to Stay")

    stays = engine.recommend_stays(
        city="pauri",
        preferences=["mountain view", "wifi", "parking"]
    )

    for stay in stays[:3]:
        print(
            stay["property_name"],
            stay["recommendation_score"]
        )

    print("\nMODE 2: Stay Preference to Destination")

    destinations = engine.recommend_destinations(
        property_type="resort",
        preferences=["mountain view", "swimming pool"]
    )

    for destination in destinations[:3]:
        print(destination)

    print("\nMODE 3: Complete Trip")

    trips = engine.recommend_complete_trip(
        property_type="hotel",
        preferences=["wifi", "parking"]
    )

    for trip in trips[:2]:
        print(trip["destination"])
        print([
            stay["property_name"]
            for stay in trip["recommended_stays"]
        ])
