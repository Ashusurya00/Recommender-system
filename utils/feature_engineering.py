"""
Feature Engineering Utilities
===============================
Builds rich user profiles and item profiles from raw ratings + metadata.
Provides helpers for:
  - User profile vectors (genre preferences, activity stats)
  - Item profile vectors (popularity, avg rating, genre one-hot)
  - Cosine / Euclidean distance helpers
  - Ranking logic with relevance weighting
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from typing import Optional


GENRE_COLS = [
    "action", "adventure", "animation", "children", "comedy", "crime",
    "documentary", "drama", "fantasy", "film_noir", "horror", "musical",
    "mystery", "romance", "sci_fi", "thriller", "war", "western",
]


class UserProfileBuilder:
    """
    Constructs a rich feature vector for each user.

    Features per user
    -----------------
    - Mean rating given
    - Rating std deviation (taste consistency)
    - Number of ratings (activity level)
    - Weighted genre preference scores (avg rating × frequency per genre)
    - Favourite decade (derived from movie release years)
    """

    def __init__(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        self.ratings = ratings_df.copy()
        self.movies  = movies_df.copy()
        self._profiles: Optional[pd.DataFrame] = None
        self._build()

    def _build(self):
        r = self.ratings
        m = self.movies

        # ── Basic stats ──────────────────────────────────────────────
        stats = r.groupby("user_id")["rating"].agg(
            mean_rating="mean",
            std_rating="std",
            n_ratings="count",
        ).fillna(0)

        # ── Genre preferences ────────────────────────────────────────
        available_genres = [g for g in GENRE_COLS if g in m.columns]

        merged = r.merge(
            m[["item_id"] + available_genres], on="item_id", how="left"
        )

        genre_prefs = {}
        for g in available_genres:
            # Weighted genre score: avg rating on movies that belong to genre g
            mask = merged[g] == 1
            genre_prefs[f"pref_{g}"] = (
                merged[mask].groupby("user_id")["rating"].mean().fillna(0)
            )

        genre_df = pd.DataFrame(genre_prefs).fillna(0)

        # ── Combine ──────────────────────────────────────────────────
        profile = stats.join(genre_df, how="left").fillna(0)

        # Normalise numeric columns
        scaler = MinMaxScaler()
        num_cols = ["mean_rating", "std_rating", "n_ratings"]
        profile[num_cols] = scaler.fit_transform(profile[num_cols])

        self._profiles = profile

    def get_profile(self, user_id: int) -> pd.Series:
        if user_id not in self._profiles.index:
            return pd.Series(dtype=float)
        return self._profiles.loc[user_id]

    def get_all_profiles(self) -> pd.DataFrame:
        return self._profiles.copy()

    def find_similar_users(self, user_id: int, top_k: int = 10) -> pd.DataFrame:
        """Find similar users by Euclidean distance on profile vectors."""
        if user_id not in self._profiles.index:
            return pd.DataFrame(columns=["user_id", "distance"])

        target_vec = self._profiles.loc[[user_id]].values
        all_vecs   = self._profiles.values
        dists      = euclidean_distances(target_vec, all_vecs).flatten()

        idx_sorted = np.argsort(dists)
        result = []
        for i in idx_sorted:
            uid = self._profiles.index[i]
            if uid != user_id:
                result.append({"user_id": int(uid), "distance": round(float(dists[i]), 4)})
            if len(result) >= top_k:
                break

        return pd.DataFrame(result)

    def get_genre_preferences(self, user_id: int) -> dict:
        """Return genre → preference score mapping for a user."""
        profile = self.get_profile(user_id)
        if profile.empty:
            return {}
        return {
            col.replace("pref_", ""): round(float(val), 3)
            for col, val in profile.items()
            if col.startswith("pref_") and val > 0
        }


class ItemProfileBuilder:
    """
    Constructs item-level features.

    Features per item
    -----------------
    - Avg rating received
    - Popularity (# ratings)
    - Rating std (controversy)
    - Genre one-hot vector (from movies_df)
    - Normalised popularity rank
    """

    def __init__(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        self.ratings = ratings_df.copy()
        self.movies  = movies_df.copy()
        self._profiles: Optional[pd.DataFrame] = None
        self._build()

    def _build(self):
        r = self.ratings
        m = self.movies

        stats = r.groupby("item_id")["rating"].agg(
            avg_rating="mean",
            n_ratings="count",
            rating_std="std",
        ).fillna(0)

        # Popularity rank (0 = least popular, 1 = most popular)
        stats["popularity_rank"] = (
            stats["n_ratings"].rank(pct=True)
        )

        # Merge genre one-hot
        available_genres = [g for g in GENRE_COLS if g in m.columns]
        genre_part = m.set_index("item_id")[available_genres].fillna(0)

        self._profiles = stats.join(genre_part, how="left").fillna(0)

    def get_profile(self, item_id: int) -> pd.Series:
        if item_id not in self._profiles.index:
            return pd.Series(dtype=float)
        return self._profiles.loc[item_id]

    def get_all_profiles(self) -> pd.DataFrame:
        return self._profiles.copy()

    def rank_items(
        self,
        item_ids: list[int],
        user_genre_prefs: dict,
        boost_popular: float = 0.2,
    ) -> pd.DataFrame:
        """
        Rank a candidate list of items using:
          1. Genre preference alignment (cosine similarity)
          2. Popularity boost
          3. Avg rating signal

        Returns DataFrame [item_id, relevance_score] sorted descending.
        """
        if not item_ids:
            return pd.DataFrame(columns=["item_id", "relevance_score"])

        available_genres = [g for g in GENRE_COLS if g in self._profiles.columns]

        # Build user genre preference vector
        user_vec = np.array([user_genre_prefs.get(g, 0.0) for g in available_genres])
        user_norm = np.linalg.norm(user_vec)
        if user_norm > 0:
            user_vec = user_vec / user_norm

        rows = []
        for iid in item_ids:
            if iid not in self._profiles.index:
                continue
            prof = self._profiles.loc[iid]

            # Genre alignment
            item_genre_vec = np.array([float(prof.get(g, 0)) for g in available_genres])
            item_norm = np.linalg.norm(item_genre_vec)
            if item_norm > 0:
                item_genre_vec = item_genre_vec / item_norm
            genre_sim = float(np.dot(user_vec, item_genre_vec))

            # Popularity & rating signals
            popularity  = float(prof.get("popularity_rank", 0))
            avg_rating  = float(prof.get("avg_rating", 3.0)) / 5.0

            relevance = (
                0.6  * genre_sim
                + boost_popular * popularity
                + 0.2  * avg_rating
            )
            rows.append({"item_id": iid, "relevance_score": round(relevance, 4)})

        return (
            pd.DataFrame(rows)
            .sort_values("relevance_score", ascending=False)
            .reset_index(drop=True)
        )


# ── Standalone distance / similarity helpers ─────────────────────────────

def cosine_sim_vectors(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two 1-D numpy arrays."""
    norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    return float(np.dot(vec_a, vec_b) / norm) if norm > 0 else 0.0


def pearson_sim(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Pearson correlation coefficient between two vectors."""
    if np.std(vec_a) == 0 or np.std(vec_b) == 0:
        return 0.0
    return float(np.corrcoef(vec_a, vec_b)[0, 1])


def jaccard_genres(genres_a: str, genres_b: str, sep: str = "|") -> float:
    """Jaccard similarity between two genre strings."""
    set_a = set(genres_a.split(sep))
    set_b = set(genres_b.split(sep))
    inter = set_a & set_b
    union = set_a | set_b
    return len(inter) / len(union) if union else 0.0
