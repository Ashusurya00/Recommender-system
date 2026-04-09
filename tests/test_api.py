"""
API Integration Tests
======================
Smoke-tests every endpoint without a running server
by calling model functions directly.
Run: python tests/test_api.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warnings
warnings.filterwarnings("ignore")

from data.data_loader import MovieLensLoader
from models.collaborative_filter import CollaborativeFilter
from models.content_based import ContentBasedFilter
from models.hybrid import HybridRecommender
from evaluation.metrics import Evaluator, precision_at_k, recall_at_k, ndcg_at_k
from utils.feature_engineering import UserProfileBuilder, ItemProfileBuilder

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"


def assert_ok(condition: bool, label: str):
    if condition:
        print(f"  {PASS} {label}")
    else:
        print(f"  {FAIL} {label}")
        raise AssertionError(label)


def run_tests():
    print("\n" + "=" * 55)
    print("  CineMatch — Integration Test Suite")
    print("=" * 55)

    # ── Data Layer ────────────────────────────────────────────────────
    print("\n[1] Data Layer")
    loader = MovieLensLoader()
    assert_ok(len(loader.ratings) > 1000,   "Ratings loaded (>1000 rows)")
    assert_ok(len(loader.movies)  > 100,    "Movies loaded (>100 rows)")
    assert_ok(len(loader.users)   > 100,    "Users loaded (>100 rows)")
    summary = loader.eda_summary()
    assert_ok("n_users"   in summary,       "EDA summary has n_users")
    assert_ok("sparsity"  in summary,       "EDA summary has sparsity")
    assert_ok(0 < summary["sparsity"] < 1,  "Sparsity is between 0 and 1")
    mat = loader.get_user_item_matrix()
    assert_ok(mat.shape[0] > 0,             "User-item matrix has rows")

    # ── Feature Engineering ───────────────────────────────────────────
    print("\n[2] Feature Engineering")
    upb = UserProfileBuilder(loader.ratings, loader.movies)
    p1  = upb.get_profile(1)
    assert_ok(len(p1) > 0,                  "User profile built for user 1")
    prefs = upb.get_genre_preferences(1)
    assert_ok(isinstance(prefs, dict),      "Genre preferences returned as dict")
    sim_users = upb.find_similar_users(1, top_k=5)
    assert_ok(len(sim_users) == 5,          "find_similar_users returns 5 rows")

    ipb = ItemProfileBuilder(loader.ratings, loader.movies)
    ip1 = ipb.get_profile(1)
    assert_ok("avg_rating" in ip1.index,    "Item profile has avg_rating")
    ranked = ipb.rank_items([1,2,3,4,5], prefs)
    assert_ok(len(ranked) > 0,              "rank_items returns results")

    # ── Collaborative Filter ──────────────────────────────────────────
    print("\n[3] Collaborative Filtering")
    cf = CollaborativeFilter(loader.ratings, n_factors=50)
    assert_ok(cf.explained_variance > 0,    "SVD explained variance > 0")

    recs_uu = cf.recommend_user_user(1, top_k=10)
    assert_ok(len(recs_uu) > 0,             "User-User CF returns recs")
    assert_ok("item_id" in recs_uu.columns, "User-User recs has item_id col")
    assert_ok("score"   in recs_uu.columns, "User-User recs has score col")

    recs_ii = cf.recommend_item_item(1, top_k=10)
    assert_ok(len(recs_ii) > 0,             "Item-Item CF returns recs")

    recs_svd = cf.recommend_svd(1, top_k=10)
    assert_ok(len(recs_svd) > 0,            "SVD CF returns recs")

    # Ensure no already-rated items appear in CF recs
    rated = set(loader.ratings[loader.ratings["user_id"] == 1]["item_id"])
    assert_ok(
        not any(iid in rated for iid in recs_svd["item_id"]),
        "SVD recs do not include already-rated items"
    )

    sim_items = cf.get_similar_items(1, top_k=5)
    assert_ok(len(sim_items) == 5,          "get_similar_items returns 5 rows")
    sim_usr   = cf.get_similar_users(1, top_k=5)
    assert_ok(len(sim_usr)   == 5,          "get_similar_users returns 5 rows")

    # ── Content-Based Filter ──────────────────────────────────────────
    print("\n[4] Content-Based Filtering")
    cb = ContentBasedFilter(loader.movies, loader.ratings)

    recs_cb = cb.recommend(1, top_k=10)
    assert_ok(len(recs_cb) > 0,             "CB returns recs")

    cold = cb.cold_start_recommend(["action", "comedy"], top_k=10)
    assert_ok(len(cold) > 0,                "Cold-start recs returned")
    assert_ok("item_id" in cold.columns,    "Cold-start recs has item_id")

    features = cb.explain(1, top_n=5)
    assert_ok(isinstance(features, list),   "explain() returns a list")
    assert_ok(len(features) > 0,            "explain() returns features")

    sim_cb = cb.get_similar_items(1, top_k=5)
    assert_ok(len(sim_cb) > 0,              "CB get_similar_items works")

    # ── Hybrid Recommender ────────────────────────────────────────────
    print("\n[5] Hybrid Recommender")
    h = HybridRecommender(cf, cb, loader.movies)

    for method in ("svd", "user_user", "item_item"):
        recs_h = h.recommend(1, top_k=10, method=method)
        assert_ok(len(recs_h) > 0,          f"Hybrid ({method}) returns recs")

    recs_cold = h.recommend(1, new_user=True, genre_prefs=["drama", "comedy"])
    assert_ok(len(recs_cold) > 0,           "Hybrid cold-start returns recs")

    exp = h.explain_recommendation(1, 50)
    assert_ok("similar_users"  in exp,      "Explanation has similar_users")
    assert_ok("key_features"   in exp,      "Explanation has key_features")
    assert_ok("title"          in exp,      "Explanation has title")

    # ── Evaluation Metrics ────────────────────────────────────────────
    print("\n[6] Evaluation Metrics")
    p = precision_at_k([1, 2, 3, 4, 5], {2, 4, 6}, k=5)
    assert_ok(abs(p - 0.4) < 1e-6,         "precision_at_k correct (0.4)")

    r = recall_at_k([1, 2, 3, 4, 5], {2, 4, 6}, k=5)
    assert_ok(abs(r - 2/3) < 1e-6,         "recall_at_k correct (0.667)")

    n = ndcg_at_k([1, 2, 3], {2}, k=3)
    assert_ok(n > 0,                        "ndcg_at_k > 0")

    ev = Evaluator(loader.ratings, threshold=3.0, k=10)
    assert_ok(len(ev.train_df) > len(ev.test_df), "Train larger than test")
    row = ev.evaluate_ranking_from_train(
        lambda df: (lambda uid, k: cf.recommend_svd(uid, k)),
        "CF-SVD-test",
        n_users=20,
    )
    assert_ok("P@10"     in row,            "Eval result has P@10")
    assert_ok("NDCG@10"  in row,            "Eval result has NDCG@10")
    assert_ok(row["n_users_eval"] > 0,      "At least 1 user evaluated")

    print("\n" + "=" * 55)
    print("  ✅ All tests passed!")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    run_tests()
