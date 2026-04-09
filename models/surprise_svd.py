"""
Surprise SVD — Training & Cross-Validation
============================================
Trains the Surprise SVD model with proper cross-validation
and saves the RMSE / MAE results.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate, GridSearchCV

from data.data_loader import MovieLensLoader


def train_and_evaluate(loader: MovieLensLoader):
    reader  = Reader(rating_scale=(1, 5))
    data    = Dataset.load_from_df(
        loader.ratings[["user_id", "item_id", "rating"]], reader
    )

    print("=" * 55)
    print("  Surprise SVD — Cross-Validation (3-fold)")
    print("=" * 55)

    # ── Default SVD ───────────────────────────────────────────────────
    algo  = SVD(n_factors=50, n_epochs=20, random_state=42)
    cv    = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=3, verbose=False)
    rmse  = np.mean(cv["test_rmse"])
    mae   = np.mean(cv["test_mae"])

    print(f"\n  SVD (50 factors, 20 epochs)")
    print(f"    RMSE : {rmse:.4f}")
    print(f"    MAE  : {mae:.4f}")

    # ── Grid search over n_factors ────────────────────────────────────
    print("\n  Grid search over n_factors (20, 50, 100) …")
    param_grid = {"n_factors": [20, 50, 100], "n_epochs": [20]}
    gs = GridSearchCV(SVD, param_grid, measures=["rmse"], cv=3)
    gs.fit(data)

    best_params = gs.best_params["rmse"]
    best_rmse   = gs.best_score["rmse"]
    print(f"    Best params : {best_params}")
    print(f"    Best RMSE   : {best_rmse:.4f}")

    # ── Fit final model on all data ───────────────────────────────────
    best_algo = SVD(**best_params, random_state=42)
    trainset  = data.build_full_trainset()
    best_algo.fit(trainset)

    # ── Sample predictions ────────────────────────────────────────────
    print("\n  Sample rating predictions:")
    for uid, iid in [(1,1),(1,50),(100,200),(500,300)]:
        pred = best_algo.predict(uid, iid)
        print(f"    User {uid:4d} → Item {iid:4d} : predicted {pred.est:.2f}")

    return {
        "default_rmse": round(rmse, 4),
        "default_mae":  round(mae, 4),
        "best_rmse":    round(best_rmse, 4),
        "best_params":  best_params,
        "model":        best_algo,
    }


if __name__ == "__main__":
    loader  = MovieLensLoader()
    results = train_and_evaluate(loader)
    print("\n✅ Surprise SVD training complete.")
    print(f"   Best RMSE: {results['best_rmse']}")
