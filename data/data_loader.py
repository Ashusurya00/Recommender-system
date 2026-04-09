"""
Data Loader Module
==================
Downloads and preprocesses the MovieLens 100K dataset.
Performs EDA, handles missing values, duplicates, and outliers.
"""

import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MOVIELENS_ZIP = DATA_DIR / "ml-100k.zip"
MOVIELENS_DIR = DATA_DIR / "ml-100k"


class MovieLensLoader:
    """
    Handles downloading, loading, and preprocessing of the MovieLens 100K dataset.
    
    Attributes:
        ratings (pd.DataFrame): User-movie rating interactions.
        movies (pd.DataFrame): Movie metadata including genres and title.
        users (pd.DataFrame): User demographic information.
    """

    RATING_COLS = ["user_id", "item_id", "rating", "timestamp"]
    USER_COLS   = ["user_id", "age", "gender", "occupation", "zip_code"]
    ITEM_COLS   = [
        "item_id", "title", "release_date", "video_release_date", "imdb_url",
        "unknown", "action", "adventure", "animation", "children", "comedy",
        "crime", "documentary", "drama", "fantasy", "film_noir", "horror",
        "musical", "mystery", "romance", "sci_fi", "thriller", "war", "western",
    ]
    GENRE_COLS = [
        "action", "adventure", "animation", "children", "comedy", "crime",
        "documentary", "drama", "fantasy", "film_noir", "horror", "musical",
        "mystery", "romance", "sci_fi", "thriller", "war", "western",
    ]

    def __init__(self):
        self.ratings = None
        self.movies  = None
        self.users   = None
        self._ensure_data()
        self._load()
        self._preprocess()

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------

    def _ensure_data(self):
        if MOVIELENS_DIR.exists():
            return
        try:
            logger.info("Downloading MovieLens 100K …")
            urllib.request.urlretrieve(MOVIELENS_URL, MOVIELENS_ZIP)
            with zipfile.ZipFile(MOVIELENS_ZIP, "r") as zf:
                zf.extractall(DATA_DIR)
            logger.info("Download complete.")
        except Exception as e:
            logger.warning("Download failed (%s). Generating synthetic data …", e)
            import subprocess, sys
            gen = DATA_DIR / "generate_synthetic.py"
            if gen.exists():
                subprocess.run([sys.executable, str(gen)], check=True)

    # ------------------------------------------------------------------
    # Raw loading
    # ------------------------------------------------------------------

    def _load(self):
        self.ratings = pd.read_csv(
            MOVIELENS_DIR / "u.data",
            sep="\t", names=self.RATING_COLS,
        )
        self.movies = pd.read_csv(
            MOVIELENS_DIR / "u.item",
            sep="|", names=self.ITEM_COLS, encoding="latin-1",
        )
        self.users = pd.read_csv(
            MOVIELENS_DIR / "u.user",
            sep="|", names=self.USER_COLS,
        )

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _preprocess(self):
        # ── Ratings ──────────────────────────────────────────────────
        self.ratings.drop_duplicates(inplace=True)
        self.ratings["rating"] = self.ratings["rating"].clip(1, 5)
        self.ratings["timestamp"] = pd.to_datetime(self.ratings["timestamp"], unit="s")

        # ── Movies ───────────────────────────────────────────────────
        self.movies["release_date"] = pd.to_datetime(
            self.movies["release_date"], errors="coerce"
        )
        self.movies["year"] = self.movies["release_date"].dt.year
        self.movies.drop(
            columns=["video_release_date", "imdb_url", "unknown"], inplace=True, errors="ignore"
        )
        # Genres as a human-readable list
        self.movies["genres"] = self.movies[self.GENRE_COLS].apply(
            lambda row: "|".join(col for col in self.GENRE_COLS if row[col] == 1), axis=1
        )
        self.movies["genres"] = self.movies["genres"].replace("", "unknown")

        # ── Users ────────────────────────────────────────────────────
        self.users["age_group"] = pd.cut(
            self.users["age"],
            bins=[0, 18, 25, 35, 50, 100],
            labels=["<18", "18-25", "25-35", "35-50", "50+"],
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_ratings(self) -> pd.DataFrame:
        return self.ratings.copy()

    def get_movies(self) -> pd.DataFrame:
        return self.movies.copy()

    def get_users(self) -> pd.DataFrame:
        return self.users.copy()

    def get_merged(self) -> pd.DataFrame:
        """Full join of ratings + movie metadata + user demographics."""
        df = (
            self.ratings
            .merge(self.movies[["item_id", "title", "genres", "year"]], on="item_id")
            .merge(self.users[["user_id", "age", "gender", "occupation", "age_group"]], on="user_id")
        )
        return df

    def get_user_item_matrix(self) -> pd.DataFrame:
        """Pivot table of users × movies (NaN where unrated)."""
        return self.ratings.pivot_table(
            index="user_id", columns="item_id", values="rating"
        )

    def eda_summary(self) -> dict:
        """Returns key EDA statistics as a plain dict."""
        r = self.ratings
        return {
            "n_users":        r["user_id"].nunique(),
            "n_items":        r["item_id"].nunique(),
            "n_ratings":      len(r),
            "rating_mean":    round(r["rating"].mean(), 3),
            "rating_std":     round(r["rating"].std(), 3),
            "sparsity":       round(
                1 - len(r) / (r["user_id"].nunique() * r["item_id"].nunique()), 4
            ),
            "min_rating":     int(r["rating"].min()),
            "max_rating":     int(r["rating"].max()),
            "ratings_per_user_mean": round(r.groupby("user_id").size().mean(), 2),
            "ratings_per_item_mean": round(r.groupby("item_id").size().mean(), 2),
        }
