# 🎬 CineMatch — Production-Ready Recommendation System

> A Netflix/Amazon-style AI recommendation engine using Collaborative Filtering, Content-Based Filtering, and a Hybrid approach — with a FastAPI backend, Streamlit frontend, and GenAI explanations powered by Claude.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture](#-architecture)
3. [Tech Stack](#-tech-stack)
4. [Project Structure](#-project-structure)
5. [Models Implemented](#-models-implemented)
6. [Evaluation Results](#-evaluation-results)
7. [API Endpoints](#-api-endpoints)
8. [Quick Start (Local)](#-quick-start-local)
9. [Docker Deployment](#-docker-deployment)
10. [GenAI Component](#-genai-component)
11. [Design Decisions](#-design-decisions)

---

## 🌟 Project Overview

**CineMatch** is a production-level movie recommendation system that demonstrates:

| Feature | Implementation |
|---|---|
| **Dataset** | MovieLens 100K (943 users, 1,682 movies, 100,000 ratings) |
| **CF Approaches** | User-User, Item-Item, SVD Matrix Factorization |
| **Content-Based** | TF-IDF on genres + title tokens |
| **Hybrid System** | Weighted ensemble (60% CF + 40% CB) |
| **Cold-Start** | Genre-preference based content-only fallback |
| **Explainability** | Feature importance, similar users, similarity scores |
| **API** | FastAPI REST with 10 endpoints |
| **Frontend** | Streamlit with interactive charts (Plotly) |
| **GenAI** | Claude API for natural-language explanations |
| **Deployment** | Docker + docker-compose ready |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND (Streamlit)                  │
│   • Recommendations Tab  • EDA Dashboard                │
│   • Explore Movies       • New User (Cold-Start)        │
└────────────────────┬────────────────────────────────────┘
                     │  HTTP / REST
┌────────────────────▼────────────────────────────────────┐
│                   BACKEND (FastAPI)                      │
│   /recommend/{user_id}  /similar/items/{id}             │
│   /cold-start           /explain/{user}/{item}          │
└──────────┬──────────────────────────┬───────────────────┘
           │                          │
┌──────────▼──────────┐   ┌───────────▼──────────────────┐
│  Collaborative       │   │  Content-Based Filter        │
│  Filter              │   │  • TF-IDF on genres+title    │
│  • User-User CF      │   │  • Cosine similarity         │
│  • Item-Item CF      │   │  • Cold-start fallback       │
│  • SVD (50 factors)  │   └───────────┬──────────────────┘
└──────────┬──────────┘               │
           └────────────┬─────────────┘
                        │
           ┌────────────▼──────────────┐
           │     Hybrid Recommender    │
           │  score = 0.6×CF + 0.4×CB │
           └───────────────────────────┘
                        │
           ┌────────────▼──────────────┐
           │     Data Layer            │
           │  MovieLens 100K           │
           │  Ratings / Users / Items  │
           └───────────────────────────┘
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Data Processing | Pandas, NumPy |
| ML / Similarity | Scikit-learn, SciPy, scikit-surprise |
| Matrix Factorization | TruncatedSVD (sklearn) + Surprise SVD |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit + Plotly |
| GenAI | Anthropic Claude API |
| Containerization | Docker + docker-compose |

---

## 📁 Project Structure

```
recommender_system/
│
├── data/
│   ├── data_loader.py          # MovieLens loader + EDA + preprocessing
│   ├── generate_synthetic.py   # Synthetic data fallback (offline use)
│   └── ml-100k/                # Dataset (auto-downloaded)
│
├── models/
│   ├── collaborative_filter.py  # User-User, Item-Item, SVD
│   ├── content_based.py         # TF-IDF content-based filter
│   └── hybrid.py                # Weighted ensemble + explainability
│
├── evaluation/
│   └── metrics.py              # P@K, R@K, NDCG@K, RMSE evaluator
│
├── api/
│   └── main.py                 # FastAPI 10-endpoint REST API
│
├── frontend/
│   └── app.py                  # Streamlit UI
│
├── notebooks/
│   └── eda_pipeline.py         # Full EDA + model comparison script
│
├── static/
│   ├── eda_report.html         # Generated EDA report
│   └── model_comparison.csv   # Evaluation results
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🤖 Models Implemented

### 1. User-User Collaborative Filtering
- Computes cosine similarity between user rating vectors
- Finds top-20 most similar users ("peers")
- Aggregates peer ratings weighted by similarity
- **Best for**: users with rich rating history

### 2. Item-Item Collaborative Filtering
- Computes cosine similarity between item rating vectors
- For each highly-rated item by user → finds similar items
- **Best for**: stable item catalog (items change less than users)

### 3. SVD Matrix Factorization
- TruncatedSVD (50 latent factors) on user-item matrix
- Decomposes into user factors × item factors
- Dot product gives preference scores for unseen items
- Also uses Surprise SVD for rating prediction with cross-validation

### 4. TF-IDF Content-Based Filtering
- "Soup" = `genres (×2 weight) + title tokens`
- TF-IDF vectorizer → 500 features, bigrams
- User profile = weighted average of rated-item vectors (by rating)
- Cosine similarity between profile and candidate items

### 5. Hybrid Recommender
- `hybrid_score = 0.6 × CF_score + 0.4 × CB_score`
- Both scores normalised to [0,1] before weighting
- Falls back to content-only for cold-start users

### 6. Cold-Start Handler
- New user provides genre preferences (e.g., ["action", "comedy"])
- Query vector built from genre tokens (tripled weight)
- TF-IDF cosine similarity ranks all items
- No rating history required

---

## 📊 Evaluation Results

Evaluated on 80 held-out users, 20% test split, threshold = 3.0 stars:

| Model | P@10 | R@10 | NDCG@10 |
|---|---|---|---|
| CF-UserUser | 0.0150 | 0.0059 | 0.0193 |
| CF-ItemItem | 0.0187 | 0.0071 | 0.0205 |
| CF-SVD | 0.0187 | 0.0073 | 0.0178 |
| **ContentBased** | **0.0250** | **0.0094** | **0.0259** |
| Hybrid-SVD | 0.0187 | 0.0070 | 0.0179 |

> **Why Content-Based leads?**  
> With synthetic data (random latent factors → low CF signal), content features dominate. On real MovieLens data, SVD and Hybrid typically outperform pure content-based, as collaborative signal is much stronger with real user preference patterns.

**Metrics Explained:**
- **P@10** (Precision@10): Of 10 recommended items, what fraction are actually relevant?
- **R@10** (Recall@10): Of all relevant items, what fraction appear in the top 10?
- **NDCG@10**: Discounted cumulative gain — rewards placing relevant items higher in the list

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/eda/summary` | Dataset statistics |
| `GET` | `/recommend/{user_id}` | Hybrid recommendations |
| `GET` | `/recommend/{user_id}/cf` | CF-only recommendations |
| `GET` | `/recommend/{user_id}/cb` | Content-based only |
| `POST` | `/cold-start` | New user (genre preferences) |
| `GET` | `/similar/items/{item_id}` | Item-item similarity |
| `GET` | `/similar/users/{user_id}` | User-user similarity |
| `GET` | `/explain/{user_id}/{item_id}` | Explainability report |
| `GET` | `/movies/{item_id}` | Movie details |

**Query parameters** (on `/recommend`):
- `top_k` — number of results (1–50, default 10)
- `method` — `svd` | `user_user` | `item_item`

**Cold-start body:**
```json
{
  "genres": ["action", "comedy", "thriller"],
  "top_k": 10
}
```

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.11+
- (Optional) Anthropic API key for GenAI explanations

### 1. Install Dependencies

```bash
git clone <repo>
cd recommender_system
pip install -r requirements.txt
```

### 2. Generate Synthetic Data (if offline)

```bash
python data/generate_synthetic.py
```

### 3. Run EDA & Evaluation Pipeline

```bash
python notebooks/eda_pipeline.py
# → static/eda_report.html
# → static/model_comparison.csv
```

### 4. Start the API

```bash
uvicorn api.main:app --reload --port 8000
# Docs at: http://localhost:8000/docs
```

> **Note:** On first start, the API downloads MovieLens 100K (~5 MB) and builds all models (~30–60s).

### 5. Start the Frontend

```bash
# In a new terminal:
streamlit run frontend/app.py
# Opens at: http://localhost:8501
```

### 6. (Optional) Enable GenAI Explanations

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## 🐳 Docker Deployment

### Build & Run Both Services

```bash
# Copy your API key
export ANTHROPIC_API_KEY=sk-ant-...

docker-compose up --build
```

- API: http://localhost:8000
- Frontend: http://localhost:8501
- API Docs: http://localhost:8000/docs

### Run Only the API

```bash
docker build -t cinematch .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY cinematch
```

---

## ✨ GenAI Component

CineMatch integrates **Claude** (Anthropic) to generate personalized, natural-language recommendation explanations:

**Example output:**
> *"User #42, these picks reflect your love of tightly-plotted thrillers with moral ambiguity — the same tension that made your 5-star ratings for crime dramas so consistent. We've blended what viewers like you adore with movies sharing those exact genre fingerprints."*

The GenAI layer:
1. Reads the top-5 recommended titles + inferred genres
2. Sends a structured prompt to `claude-sonnet-4-20250514`
3. Returns 2–3 warm, contextual sentences displayed in the UI

**Chatbot mode** (cold-start page): Users describe themselves in natural language, Claude extracts genre preferences and triggers the cold-start recommender.

---

## 🧩 Design Decisions

| Decision | Rationale |
|---|---|
| **TruncatedSVD over ALS/BPR** | No GPU required; scikit-learn compatible; sufficient for 100K ratings |
| **TF-IDF over embeddings** | Fast, interpretable; embeddings would require PyTorch + more data |
| **60/40 CF/CB weighting** | CF captures collaborative signal; CB handles sparse users; ratio tunable |
| **FastAPI over Flask** | Async, auto-docs, Pydantic validation, modern Python |
| **Synthetic data fallback** | Ensures the project runs fully offline for demos/interviews |
| **Explainability built-in** | `/explain` endpoint returns similar users + TF-IDF feature names |

---

## 📈 Scalability Notes

For production at Netflix/Amazon scale:
- Replace TruncatedSVD with **ALS** (implicit) or **LightFM** for implicit feedback
- Use **Redis** for caching user profiles and recommendation results
- Serve the user-item matrix from **S3/GCS** (sparse format)
- Deploy API on **Kubernetes** with horizontal pod autoscaling
- Add **A/B testing** framework to compare model versions live
- Use **Apache Kafka** for real-time rating ingestion

---

*Built with Python · FastAPI · Streamlit · scikit-learn · Surprise · Claude AI*
