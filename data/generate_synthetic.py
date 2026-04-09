"""Generate synthetic MovieLens-like data for offline testing."""
import numpy as np
import pandas as pd
from pathlib import Path
import os

np.random.seed(42)

DATA_DIR = Path(__file__).parent
ML_DIR   = DATA_DIR / "ml-100k"
ML_DIR.mkdir(exist_ok=True)

GENRES = ["action","adventure","animation","children","comedy","crime",
          "documentary","drama","fantasy","film_noir","horror","musical",
          "mystery","romance","sci_fi","thriller","war","western"]

N_USERS, N_ITEMS, N_RATINGS = 943, 1682, 100000

# ── Users ──────────────────────────────────────────────────────────────
occupations = ["student","engineer","doctor","artist","teacher","scientist","writer","none"]
users = pd.DataFrame({
    "user_id":    range(1, N_USERS+1),
    "age":        np.random.randint(15, 70, N_USERS),
    "gender":     np.random.choice(["M","F"], N_USERS),
    "occupation": np.random.choice(occupations, N_USERS),
    "zip_code":   [f"{np.random.randint(10000,99999):05d}" for _ in range(N_USERS)],
})
users.to_csv(ML_DIR/"u.user", sep="|", index=False, header=False)

# ── Movies ─────────────────────────────────────────────────────────────
titles = [
    "Star Wars (1977)","Fargo (1996)","Toy Story (1995)","GoodFellas (1990)",
    "Schindler's List (1993)","Pulp Fiction (1994)","The Silence of the Lambs (1991)",
    "Forrest Gump (1994)","The Shawshank Redemption (1994)","Se7en (1995)",
]
base_titles = titles.copy()
while len(base_titles) < N_ITEMS:
    base_titles.append(f"Movie #{len(base_titles)+1} ({np.random.randint(1970,1999)})")

genre_matrix = np.zeros((N_ITEMS, len(GENRES)), dtype=int)
for i in range(N_ITEMS):
    chosen = np.random.choice(len(GENRES), size=np.random.randint(1,4), replace=False)
    genre_matrix[i, chosen] = 1

movie_df = pd.DataFrame({"item_id": range(1, N_ITEMS+1), "title": base_titles[:N_ITEMS]})
movie_df["release_date"] = [
    f"01-Jan-{np.random.randint(1970,2000)}" for _ in range(N_ITEMS)
]
movie_df["video_release_date"] = ""
movie_df["imdb_url"] = ""
genre_df = pd.DataFrame(genre_matrix, columns=GENRES)
full = pd.concat([movie_df, genre_df], axis=1)
full.insert(5, "unknown", 0)  # MovieLens has 'unknown' genre slot
full.to_csv(ML_DIR/"u.item", sep="|", index=False, header=False)

# ── Ratings ─────────────────────────────────────────────────────────────
# Simulate user-item preferences using latent factors
U = np.random.randn(N_USERS, 10)
V = np.random.randn(N_ITEMS, 10)
# Sample random (user,item) pairs
user_ids = np.random.randint(1, N_USERS+1, N_RATINGS)
item_ids = np.random.randint(1, N_ITEMS+1, N_RATINGS)
# Score = dot product + noise, clipped to [1,5]
scores = np.sum(U[user_ids-1] * V[item_ids-1], axis=1)
ratings_raw = np.clip(np.round((scores - scores.min()) / (scores.max()-scores.min()) * 4 + 1), 1, 5).astype(int)
timestamps  = np.random.randint(880000000, 900000000, N_RATINGS)

ratings_df = pd.DataFrame({
    "user_id":   user_ids,
    "item_id":   item_ids,
    "rating":    ratings_raw,
    "timestamp": timestamps,
}).drop_duplicates(subset=["user_id","item_id"])

ratings_df.to_csv(ML_DIR/"u.data", sep="\t", index=False, header=False)
print(f"Synthetic data generated: {len(users)} users, {N_ITEMS} items, {len(ratings_df)} ratings")
