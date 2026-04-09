"""
Collaborative Filtering Module
================================
Implements:
  - User-User CF (cosine similarity)
  - Item-Item CF (cosine similarity)
  - SVD via scikit-learn TruncatedSVD (matrix factorization)
  - Surprise SVD wrapper (rating-prediction)
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
import logging

logger = logging.getLogger(__name__)


class CollaborativeFilter:
    """
    Provides User-User and Item-Item collaborative filtering
    plus SVD-based matrix factorization recommendations.

    Parameters
    ----------
    ratings_df : pd.DataFrame
        Must contain columns [user_id, item_id, rating].
    n_factors : int
        Number of latent factors for SVD. Default 50.
    """

    def __init__(self, ratings_df: pd.DataFrame, n_factors: int = 50):
        self.ratings_df = ratings_df.copy()
        self.n_factors   = n_factors

        # Build user-item matrix (fill NaN with 0 for similarity)
        self.ui_matrix = (
            ratings_df.pivot_table(index="user_id", columns="item_id", values="rating")
            .fillna(0)
        )
        self.users = list(self.ui_matrix.index)
        self.items = list(self.ui_matrix.columns)

        self._user_sim = None
        self._item_sim = None
        self._svd_model = None
        self._surprise_svd = None

        self._build_similarities()
        self._build_svd()

    # ------------------------------------------------------------------
    # Internal build methods
    # ------------------------------------------------------------------

    def _build_similarities(self):
        """Pre-compute user-user and item-item cosine similarity matrices."""
        mat = self.ui_matrix.values
        # Normalize rows for cosine similarity
        user_norm = normalize(mat, norm="l2")
        item_norm = normalize(mat.T, norm="l2")
        self._user_sim = pd.DataFrame(
            cosine_similarity(user_norm),
            index=self.ui_matrix.index,
            columns=self.ui_matrix.index,
        )
        self._item_sim = pd.DataFrame(
            cosine_similarity(item_norm),
            index=self.ui_matrix.columns,
            columns=self.ui_matrix.columns,
        )
        logger.info("Similarity matrices built.")

    def _build_svd(self):
        """Fit TruncatedSVD on the user-item matrix."""
        mat = self.ui_matrix.values
        svd = TruncatedSVD(n_components=self.n_factors, random_state=42)
        self._user_factors = svd.fit_transform(mat)
        self._item_factors = svd.components_.T          # shape: (n_items, n_factors)
        self._svd_model = svd
        self._explained_var = svd.explained_variance_ratio_.sum()
        logger.info("SVD built. Explained variance: %.3f", self._explained_var)

    def train_surprise_svd(self):
        """Train Surprise SVD for rating prediction and cross-validation."""
        reader  = Reader(rating_scale=(1, 5))
        data    = Dataset.load_from_df(
            self.ratings_df[["user_id", "item_id", "rating"]], reader
        )
        algo    = SVD(n_factors=self.n_factors, n_epochs=20, random_state=42)
        results = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=3, verbose=False)
        self._surprise_svd = algo
        # Final fit on full data
        trainset = data.build_full_trainset()
        algo.fit(trainset)
        return {
            "rmse": float(np.mean(results["test_rmse"])),
            "mae":  float(np.mean(results["test_mae"])),
        }

    # ------------------------------------------------------------------
    # Recommendation methods
    # ------------------------------------------------------------------

    def recommend_user_user(self, user_id: int, top_k: int = 10) -> pd.DataFrame:
        """
        User-User CF: find similar users, aggregate their ratings.

        Returns top_k items the target user hasn't rated yet.
        """
        if user_id not in self._user_sim.index:
            return pd.DataFrame(columns=["item_id", "score"])

        sim_scores = self._user_sim.loc[user_id].drop(user_id).sort_values(ascending=False)
        top_users  = sim_scores.head(20)

        already_rated = set(
            self.ratings_df[self.ratings_df["user_id"] == user_id]["item_id"]
        )

        weighted_scores = {}
        sim_sum         = {}
        for peer, sim in top_users.items():
            peer_ratings = self.ui_matrix.loc[peer]
            for item_id, rating in peer_ratings.items():
                if item_id in already_rated or rating == 0:
                    continue
                weighted_scores[item_id] = weighted_scores.get(item_id, 0) + sim * rating
                sim_sum[item_id]         = sim_sum.get(item_id, 0) + abs(sim)

        scores = {
            iid: weighted_scores[iid] / sim_sum[iid]
            for iid in weighted_scores if sim_sum[iid] > 0
        }
        result = (
            pd.DataFrame(list(scores.items()), columns=["item_id", "score"])
            .sort_values("score", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )
        return result

    def recommend_item_item(self, user_id: int, top_k: int = 10) -> pd.DataFrame:
        """
        Item-Item CF: for each item the user rated highly, find similar items.

        Returns top_k candidate items scored by weighted similarity.
        """
        user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]
        if user_ratings.empty:
            return pd.DataFrame(columns=["item_id", "score"])

        already_rated = set(user_ratings["item_id"])
        scores        = {}

        for _, row in user_ratings.iterrows():
            seed_item = row["item_id"]
            if seed_item not in self._item_sim.index:
                continue
            sims = self._item_sim.loc[seed_item].drop(seed_item)
            for candidate, sim in sims.items():
                if candidate in already_rated:
                    continue
                scores[candidate] = scores.get(candidate, 0) + sim * (row["rating"] / 5.0)

        result = (
            pd.DataFrame(list(scores.items()), columns=["item_id", "score"])
            .sort_values("score", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )
        return result

    def recommend_svd(self, user_id: int, top_k: int = 10) -> pd.DataFrame:
        """
        SVD (matrix factorization): reconstruct user preferences in latent space.

        Returns top_k items not yet rated by the user.
        """
        if user_id not in self.ui_matrix.index:
            return pd.DataFrame(columns=["item_id", "score"])

        user_idx     = self.users.index(user_id)
        user_vec     = self._user_factors[user_idx]          # (n_factors,)
        scores_all   = self._item_factors @ user_vec          # (n_items,)
        already_rated = set(
            self.ratings_df[self.ratings_df["user_id"] == user_id]["item_id"]
        )

        recs = []
        for idx, score in enumerate(scores_all):
            item_id = self.items[idx]
            if item_id not in already_rated:
                recs.append({"item_id": item_id, "score": float(score)})

        result = (
            pd.DataFrame(recs)
            .sort_values("score", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )
        return result

    def predict_rating(self, user_id: int, item_id: int) -> float:
        """Predict rating using Surprise SVD (if trained)."""
        if self._surprise_svd is None:
            raise RuntimeError("Call train_surprise_svd() first.")
        pred = self._surprise_svd.predict(user_id, item_id)
        return round(pred.est, 2)

    def get_similar_items(self, item_id: int, top_k: int = 10) -> pd.DataFrame:
        """Return top_k most similar items by item-item cosine similarity."""
        if item_id not in self._item_sim.index:
            return pd.DataFrame(columns=["item_id", "similarity"])
        sims = (
            self._item_sim.loc[item_id]
            .drop(item_id)
            .sort_values(ascending=False)
            .head(top_k)
            .reset_index()
        )
        sims.columns = ["item_id", "similarity"]
        return sims

    def get_similar_users(self, user_id: int, top_k: int = 10) -> pd.DataFrame:
        """Return top_k most similar users."""
        if user_id not in self._user_sim.index:
            return pd.DataFrame(columns=["user_id", "similarity"])
        sims = (
            self._user_sim.loc[user_id]
            .drop(user_id)
            .sort_values(ascending=False)
            .head(top_k)
            .reset_index()
        )
        sims.columns = ["user_id", "similarity"]
        return sims

    @property
    def explained_variance(self) -> float:
        return self._explained_var
