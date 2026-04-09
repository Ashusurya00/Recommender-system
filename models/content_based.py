"""
Content-Based Filtering Module
================================
Uses TF-IDF on movie genres + title tokens to build item profiles.
Recommends items whose profiles are most similar to the user's
watched items (weighted by rating).
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class ContentBasedFilter:
    """
    Content-Based recommender using TF-IDF item profiles.

    Parameters
    ----------
    movies_df  : pd.DataFrame  — item metadata (item_id, title, genres, year)
    ratings_df : pd.DataFrame  — (user_id, item_id, rating)
    """

    def __init__(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
        self.movies_df  = movies_df.set_index("item_id") if "item_id" in movies_df.columns else movies_df
        self.ratings_df = ratings_df.copy()
        self._tfidf_matrix = None
        self._vectorizer    = None
        self._item_ids      = None
        self._build_profiles()

    # ------------------------------------------------------------------

    def _build_profiles(self):
        """
        Construct a 'soup' string per item and fit TF-IDF.
        Soup = genres (repeated for emphasis) + title tokens.
        """
        df = self.movies_df.copy()
        df["soup"] = (
            df["genres"].fillna("").str.replace("|", " ", regex=False)
            + " "
            + df["genres"].fillna("").str.replace("|", " ", regex=False)   # doubled weight
            + " "
            + df["title"].fillna("").str.lower()
        )

        self._vectorizer   = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        self._tfidf_matrix = self._vectorizer.fit_transform(df["soup"])
        self._item_ids     = list(df.index)
        logger.info("Content profiles built for %d items.", len(self._item_ids))

    # ------------------------------------------------------------------

    def recommend(self, user_id: int, top_k: int = 10) -> pd.DataFrame:
        """
        Build a weighted user profile from rated items, then rank unseen items.

        Returns a DataFrame [item_id, score] sorted descending.
        """
        user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]
        if user_ratings.empty:
            return pd.DataFrame(columns=["item_id", "score"])

        already_rated = set(user_ratings["item_id"])

        # Build user profile = weighted average of item TF-IDF vectors
        profile_vecs = []
        for _, row in user_ratings.iterrows():
            iid = row["item_id"]
            if iid not in self._item_ids:
                continue
            idx  = self._item_ids.index(iid)
            vec  = self._tfidf_matrix[idx].toarray().flatten()
            profile_vecs.append(vec * row["rating"])

        if not profile_vecs:
            return pd.DataFrame(columns=["item_id", "score"])

        user_profile = np.mean(profile_vecs, axis=0).reshape(1, -1)

        # Score all items via cosine similarity
        scores = cosine_similarity(user_profile, self._tfidf_matrix).flatten()

        recs = []
        for idx, score in enumerate(scores):
            iid = self._item_ids[idx]
            if iid not in already_rated:
                recs.append({"item_id": iid, "score": float(score)})

        return (
            pd.DataFrame(recs)
            .sort_values("score", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )

    def get_similar_items(self, item_id: int, top_k: int = 10) -> pd.DataFrame:
        """Return the top_k content-similar items for a given item."""
        if item_id not in self._item_ids:
            return pd.DataFrame(columns=["item_id", "similarity"])
        idx      = self._item_ids.index(item_id)
        item_vec = self._tfidf_matrix[idx]
        sims     = cosine_similarity(item_vec, self._tfidf_matrix).flatten()
        top_idx  = np.argsort(sims)[::-1][1 : top_k + 1]
        return pd.DataFrame({
            "item_id":    [self._item_ids[i] for i in top_idx],
            "similarity": [round(float(sims[i]), 4) for i in top_idx],
        })

    def explain(self, item_id: int, top_n: int = 5) -> list[str]:
        """Return the top TF-IDF feature names that define this item's profile."""
        if item_id not in self._item_ids:
            return []
        idx      = self._item_ids.index(item_id)
        vec      = self._tfidf_matrix[idx].toarray().flatten()
        top_idx  = np.argsort(vec)[::-1][:top_n]
        features = self._vectorizer.get_feature_names_out()
        return [features[i] for i in top_idx if vec[i] > 0]

    def cold_start_recommend(self, genre_preferences: list[str], top_k: int = 10) -> pd.DataFrame:
        """
        Handle cold-start: user provides genre preferences (strings).
        Returns top_k items matching those genres.
        """
        query = " ".join(genre_preferences) * 3   # emphasise
        query_vec = self._vectorizer.transform([query])
        sims      = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        top_idx   = np.argsort(sims)[::-1][:top_k]
        return pd.DataFrame({
            "item_id": [self._item_ids[i] for i in top_idx],
            "score":   [round(float(sims[i]), 4) for i in top_idx],
        })
