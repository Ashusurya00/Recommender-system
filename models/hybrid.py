"""
Hybrid Recommendation System
==============================
Combines Collaborative Filtering and Content-Based Filtering
using a weighted ensemble.

Handles cold-start via content-based fallback.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class HybridRecommender:
    """
    Weighted hybrid of CF (SVD or User-User) and Content-Based filtering.

    Parameters
    ----------
    cf_model  : CollaborativeFilter instance
    cb_model  : ContentBasedFilter instance
    movies_df : pd.DataFrame with item metadata
    cf_weight : float  weight assigned to CF scores (default 0.6)
    cb_weight : float  weight assigned to CB scores (default 0.4)
    """

    def __init__(self, cf_model, cb_model, movies_df: pd.DataFrame,
                 cf_weight: float = 0.6, cb_weight: float = 0.4):
        self.cf       = cf_model
        self.cb       = cb_model
        self.movies   = movies_df.set_index("item_id") if "item_id" in movies_df.columns else movies_df
        self.cf_w     = cf_weight
        self.cb_w     = cb_weight

    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(df: pd.DataFrame, col: str = "score") -> pd.DataFrame:
        """Min-max normalise scores to [0, 1]."""
        if df.empty:
            return df
        mn, mx = df[col].min(), df[col].max()
        df = df.copy()
        df[col] = (df[col] - mn) / (mx - mn + 1e-9)
        return df

    # ------------------------------------------------------------------

    def recommend(
        self,
        user_id:     int,
        top_k:       int = 10,
        method:      str = "svd",       # 'svd' | 'user_user' | 'item_item'
        new_user:    bool = False,
        genre_prefs: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Generate hybrid recommendations for a user.

        Parameters
        ----------
        user_id    : target user
        top_k      : number of recommendations
        method     : CF sub-method to use
        new_user   : True → cold-start path (content-only via genre_prefs)
        genre_prefs: genre list for cold-start
        """
        # ── Cold-start ───────────────────────────────────────────────
        if new_user or genre_prefs:
            prefs = genre_prefs or ["drama", "comedy"]
            recs  = self.cb.cold_start_recommend(prefs, top_k=top_k)
            return self._enrich(recs)

        # ── CF branch ────────────────────────────────────────────────
        if method == "user_user":
            cf_recs = self.cf.recommend_user_user(user_id, top_k=top_k * 3)
        elif method == "item_item":
            cf_recs = self.cf.recommend_item_item(user_id, top_k=top_k * 3)
        else:
            cf_recs = self.cf.recommend_svd(user_id, top_k=top_k * 3)

        # ── CB branch ────────────────────────────────────────────────
        cb_recs = self.cb.recommend(user_id, top_k=top_k * 3)

        # ── Merge & weight ───────────────────────────────────────────
        cf_recs = self._normalise(cf_recs)
        cb_recs = self._normalise(cb_recs)

        cf_recs = cf_recs.rename(columns={"score": "cf_score"})
        cb_recs = cb_recs.rename(columns={"score": "cb_score"})

        merged = pd.merge(cf_recs, cb_recs, on="item_id", how="outer").fillna(0)
        merged["hybrid_score"] = (
            self.cf_w * merged["cf_score"] + self.cb_w * merged["cb_score"]
        )

        result = (
            merged[["item_id", "hybrid_score", "cf_score", "cb_score"]]
            .sort_values("hybrid_score", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )
        return self._enrich(result, score_col="hybrid_score")

    # ------------------------------------------------------------------

    def _enrich(self, recs: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
        """Join movie metadata onto recommendation results."""
        if recs.empty:
            return recs
        recs = recs.copy()
        for col in ["title", "genres", "year"]:
            if col in self.movies.columns:
                recs[col] = recs["item_id"].map(self.movies[col])
        return recs

    def explain_recommendation(self, user_id: int, item_id: int) -> dict:
        """
        Produce a plain-language explanation of why item was recommended.

        Returns a dict with:
          - similar_users : list of similar user IDs
          - key_features  : top TF-IDF features of the item
          - cf_score      : collaborative filtering score
          - cb_score      : content-based score
        """
        sim_users   = self.cf.get_similar_users(user_id, top_k=5)
        key_features = self.cb.explain(item_id, top_n=5)
        movie_info  = self.movies.loc[item_id] if item_id in self.movies.index else {}

        return {
            "item_id":       item_id,
            "title":         movie_info.get("title", "Unknown"),
            "genres":        movie_info.get("genres", ""),
            "similar_users": sim_users["user_id"].tolist(),
            "key_features":  key_features,
        }
