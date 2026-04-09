"""
FastAPI Backend
================
Endpoints:
  GET  /recommend/{user_id}              — hybrid recommendations
  GET  /recommend/{user_id}/cf           — collaborative filtering
  GET  /recommend/{user_id}/cb           — content-based
  GET  /similar/items/{item_id}          — similar items
  GET  /similar/users/{user_id}          — similar users
  POST /cold-start                       — new user recommendations
  GET  /explain/{user_id}/{item_id}      — explainability
  GET  /movies/{item_id}                 — movie details
  GET  /eda/summary                      — EDA statistics
  GET  /health                           — health check
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging
import pandas as pd

from data.data_loader import MovieLensLoader
from models.collaborative_filter import CollaborativeFilter
from models.content_based import ContentBasedFilter
from models.hybrid import HybridRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="🎬 Movie Recommendation API",
    description="Production-ready recommendation system using CF + Content-Based + Hybrid approaches.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state (loaded once at startup) ────────────────────────────────
loader   = None
cf_model = None
cb_model = None
hybrid   = None


@app.on_event("startup")
async def startup():
    global loader, cf_model, cb_model, hybrid
    logger.info("Loading data …")
    loader   = MovieLensLoader()
    cf_model = CollaborativeFilter(loader.ratings, n_factors=50)
    cb_model = ContentBasedFilter(loader.movies, loader.ratings)
    hybrid   = HybridRecommender(cf_model, cb_model, loader.movies)
    logger.info("Models ready.")


# ── Schemas ──────────────────────────────────────────────────────────────
class ColdStartRequest(BaseModel):
    genres: list[str]
    top_k:  int = 10


# ── Helpers ──────────────────────────────────────────────────────────────
def _enrich_recs(recs: pd.DataFrame) -> list[dict]:
    """Add movie metadata to recommendation rows."""
    movies = loader.movies.set_index("item_id") if "item_id" in loader.movies.columns else loader.movies
    rows   = []
    for _, row in recs.iterrows():
        iid  = int(row["item_id"])
        meta = movies.loc[iid] if iid in movies.index else {}
        rows.append({
            "item_id":  iid,
            "title":    str(meta.get("title", "Unknown")),
            "genres":   str(meta.get("genres", "")),
            "year":     int(meta.get("year", 0)) if pd.notna(meta.get("year", None)) else None,
            "score":    round(float(row.get("score") or row.get("hybrid_score", 0)), 4),
        })
    return rows


# ── Routes ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": hybrid is not None}


@app.get("/eda/summary")
def eda_summary():
    return loader.eda_summary()


@app.get("/movies/{item_id}")
def movie_detail(item_id: int):
    movies = loader.movies.set_index("item_id") if "item_id" in loader.movies.columns else loader.movies
    if item_id not in movies.index:
        raise HTTPException(404, f"Movie {item_id} not found")
    m = movies.loc[item_id]
    return {
        "item_id": item_id,
        "title":   str(m.get("title")),
        "genres":  str(m.get("genres")),
        "year":    int(m.get("year")) if pd.notna(m.get("year")) else None,
    }


@app.get("/recommend/{user_id}")
def recommend_hybrid(
    user_id: int,
    top_k:   int  = Query(10, ge=1, le=50),
    method:  str  = Query("svd", regex="^(svd|user_user|item_item)$"),
):
    recs = hybrid.recommend(user_id, top_k=top_k, method=method)
    if recs.empty:
        raise HTTPException(404, f"No recommendations for user {user_id}")
    return {"user_id": user_id, "method": f"hybrid-{method}", "recommendations": _enrich_recs(recs)}


@app.get("/recommend/{user_id}/cf")
def recommend_cf(
    user_id: int,
    top_k:   int = Query(10, ge=1, le=50),
    method:  str = Query("svd", regex="^(svd|user_user|item_item)$"),
):
    if method == "user_user":
        recs = cf_model.recommend_user_user(user_id, top_k)
    elif method == "item_item":
        recs = cf_model.recommend_item_item(user_id, top_k)
    else:
        recs = cf_model.recommend_svd(user_id, top_k)
    if recs.empty:
        raise HTTPException(404, "No CF recommendations")
    return {"user_id": user_id, "method": f"cf-{method}", "recommendations": _enrich_recs(recs)}


@app.get("/recommend/{user_id}/cb")
def recommend_cb(user_id: int, top_k: int = Query(10, ge=1, le=50)):
    recs = cb_model.recommend(user_id, top_k)
    if recs.empty:
        raise HTTPException(404, "No CB recommendations")
    return {"user_id": user_id, "method": "content-based", "recommendations": _enrich_recs(recs)}


@app.post("/cold-start")
def cold_start(req: ColdStartRequest):
    recs = cb_model.cold_start_recommend(req.genres, top_k=req.top_k)
    return {"method": "cold-start-content-based", "recommendations": _enrich_recs(recs)}


@app.get("/similar/items/{item_id}")
def similar_items(item_id: int, top_k: int = Query(10, ge=1, le=30)):
    recs = cf_model.get_similar_items(item_id, top_k)
    movies = loader.movies.set_index("item_id") if "item_id" in loader.movies.columns else loader.movies
    rows  = []
    for _, row in recs.iterrows():
        iid  = int(row["item_id"])
        meta = movies.loc[iid] if iid in movies.index else {}
        rows.append({
            "item_id":    iid,
            "title":      str(meta.get("title", "Unknown")),
            "genres":     str(meta.get("genres", "")),
            "similarity": round(float(row["similarity"]), 4),
        })
    return {"item_id": item_id, "similar_items": rows}


@app.get("/similar/users/{user_id}")
def similar_users(user_id: int, top_k: int = Query(10, ge=1, le=30)):
    recs = cf_model.get_similar_users(user_id, top_k)
    return {"user_id": user_id, "similar_users": recs.to_dict(orient="records")}


@app.get("/explain/{user_id}/{item_id}")
def explain(user_id: int, item_id: int):
    return hybrid.explain_recommendation(user_id, item_id)
