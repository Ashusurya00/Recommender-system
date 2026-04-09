"""
Evaluation Module
==================
Implements Precision@K, Recall@K, NDCG@K, RMSE, MAE.
Compares all models trained on a proper train split.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

logger = logging.getLogger(__name__)


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    top_k = recommended[:k]
    return sum(1 for r in top_k if r in relevant) / k if k else 0.0


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    return sum(1 for r in top_k if r in relevant) / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    top_k = recommended[:k]
    dcg   = sum((1 / np.log2(i + 2)) for i, r in enumerate(top_k) if r in relevant)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae_score(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


class Evaluator:
    """
    Evaluates recommendation models using a train/test split.

    Each model factory is called with train_df so it only sees
    80% of ratings, then evaluated against the held-out 20%.
    """

    def __init__(self, ratings_df: pd.DataFrame, threshold: float = 3.0, k: int = 10):
        self.ratings   = ratings_df
        self.threshold = threshold
        self.k         = k
        self.train_df, self.test_df = train_test_split(
            ratings_df, test_size=0.2, random_state=42
        )

    def evaluate_ranking_from_train(
        self,
        recommend_fn_factory,
        model_name: str,
        n_users: int = 80,
    ) -> dict:
        """
        Factory receives train_df and returns a recommend_fn(uid, top_k)->DataFrame.
        We evaluate on test_df held-out items.
        """
        logger.info("Training %s …", model_name)
        recommend_fn = recommend_fn_factory(self.train_df)

        sample_users = (
            self.test_df["user_id"]
            .value_counts()
            .head(n_users)
            .index.tolist()
        )

        precisions, recalls, ndcgs = [], [], []
        for uid in sample_users:
            relevant = set(
                self.test_df[
                    (self.test_df["user_id"] == uid)
                    & (self.test_df["rating"] >= self.threshold)
                ]["item_id"]
            )
            if not relevant:
                continue
            try:
                recs    = recommend_fn(uid, self.k)
                rec_ids = recs["item_id"].tolist() if not recs.empty else []
            except Exception as exc:
                logger.debug("Rec error for user %s: %s", uid, exc)
                continue

            precisions.append(precision_at_k(rec_ids, relevant, self.k))
            recalls.append(recall_at_k(rec_ids, relevant, self.k))
            ndcgs.append(ndcg_at_k(rec_ids, relevant, self.k))

        return {
            "model":          model_name,
            f"P@{self.k}":    round(float(np.mean(precisions)), 4) if precisions else 0.0,
            f"R@{self.k}":    round(float(np.mean(recalls)),    4) if recalls    else 0.0,
            f"NDCG@{self.k}": round(float(np.mean(ndcgs)),      4) if ndcgs      else 0.0,
            "n_users_eval":   len(precisions),
        }

    def compare_models_from_train(self, model_factories: dict, n_users: int = 80) -> pd.DataFrame:
        rows = []
        for name, factory in model_factories.items():
            row = self.evaluate_ranking_from_train(factory, name, n_users)
            rows.append(row)
        return pd.DataFrame(rows).set_index("model")
