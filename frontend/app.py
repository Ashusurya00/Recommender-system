"""
Streamlit Frontend — CineMatch Recommendation System
=====================================================
Interactive UI featuring:
  • User preference selector
  • Multiple recommendation methods
  • Similarity scores & explainability
  • EDA dashboard
  • GenAI explanation via Claude API
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch · AI Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Playfair Display', serif; }

.main { background: #0d0d14; }
.stApp { background: linear-gradient(135deg, #0d0d14 0%, #12121f 100%); color: #e8e8f0; }

.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2d2d5e;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 8px 0;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #e94560; }
.metric-label { font-size: 0.85rem; color: #8888aa; text-transform: uppercase; letter-spacing: 1px; }

.movie-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2d2d5e;
    border-left: 4px solid #e94560;
    border-radius: 10px;
    padding: 16px;
    margin: 8px 0;
    transition: transform 0.2s;
}
.movie-title { font-family: 'Playfair Display', serif; font-size: 1.1rem; color: #f0f0ff; margin: 0; }
.movie-meta  { font-size: 0.82rem; color: #8888aa; margin-top: 4px; }
.score-badge {
    display: inline-block;
    background: #e94560;
    color: white;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-weight: 600;
    float: right;
}

.genre-chip {
    display: inline-block;
    background: #2d2d5e;
    color: #a0a0dd;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin: 2px;
}

.explain-box {
    background: #111128;
    border: 1px solid #e94560;
    border-radius: 10px;
    padding: 16px;
    margin: 12px 0;
    font-size: 0.9rem;
    line-height: 1.7;
}
.stButton > button {
    background: linear-gradient(90deg, #e94560, #c0392b);
    color: white; border: none; border-radius: 8px;
    font-weight: 500; font-family: 'DM Sans', sans-serif;
}
.stSelectbox label, .stSlider label, .stMultiSelect label { color: #8888aa !important; font-size: 0.85rem !important; }
.stSidebar { background: #0d0d14; border-right: 1px solid #2d2d5e; }
</style>
""", unsafe_allow_html=True)

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

GENRES = [
    "action","adventure","animation","children","comedy","crime",
    "documentary","drama","fantasy","film_noir","horror","musical",
    "mystery","romance","sci_fi","thriller","war","western",
]

# ── Helpers ───────────────────────────────────────────────────────────────

def api_get(path: str, params: dict = None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def render_movie_card(movie: dict, rank: int):
    score = movie.get("score") or movie.get("similarity") or movie.get("hybrid_score", 0)
    genres_html = "".join(
        f'<span class="genre-chip">{g}</span>'
        for g in str(movie.get("genres", "")).split("|") if g
    )
    year = f" · {int(movie['year'])}" if movie.get("year") else ""
    st.markdown(f"""
    <div class="movie-card">
        <span class="score-badge">★ {score:.3f}</span>
        <p class="movie-title">#{rank}. {movie.get('title','Unknown')}</p>
        <p class="movie-meta">{genres_html}{year}</p>
    </div>
    """, unsafe_allow_html=True)


def genai_explain(user_id: int, recs: list, method: str) -> str:
    """Call Claude API to generate a natural-language explanation."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        titles  = [r.get("title","?") for r in recs[:5]]
        genres  = list({g for r in recs[:5] for g in str(r.get("genres","")).split("|") if g})
        prompt  = (
            f"You are a friendly movie recommendation assistant. "
            f"User #{user_id} was recommended these films using the '{method}' algorithm: {titles}. "
            f"Their taste spans: {genres[:6]}. "
            f"Write 2-3 warm, engaging sentences explaining WHY these movies were picked "
            f"and what makes them special for this user. Be specific and enthusiastic!"
        )
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception:
        return "Enable the Claude API key to see AI-generated explanations."


# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    st.markdown('<p style="color:#8888aa;font-size:0.85rem;">AI-Powered Movie Recommendations</p>', unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Recommendations", "🔍 Explore Movies", "📊 EDA Dashboard", "🆕 New User"],
        label_visibility="collapsed",
    )
    st.divider()

    if "🏠" in page or "🔍" in page:
        user_id = st.number_input("User ID", min_value=1, max_value=943, value=1, step=1)
        top_k   = st.slider("# Recommendations", 5, 30, 10)
        method  = st.selectbox("Algorithm", ["svd", "user_user", "item_item"])
    st.divider()
    st.markdown('<p style="color:#555577;font-size:0.75rem;">MovieLens 100K · Collaborative + Content-Based + Hybrid</p>', unsafe_allow_html=True)


# ── Pages ─────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────
# PAGE 1 — Recommendations
# ──────────────────────────────────────────────────────────────────────────
if "🏠" in page:
    st.markdown("# 🎬 Your Recommendations")
    st.markdown(f'<p style="color:#8888aa;">Personalised picks for User <b style="color:#e94560">#{user_id}</b></p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    # Tabs for each model type
    tab1, tab2, tab3 = st.tabs(["🔀 Hybrid", "👥 Collaborative", "📄 Content-Based"])

    with tab1:
        if st.button("Get Hybrid Recommendations", key="hybrid_btn"):
            with st.spinner("Computing hybrid recommendations …"):
                data = api_get(f"/recommend/{user_id}", {"top_k": top_k, "method": method})
            if data:
                recs = data["recommendations"]
                st.markdown(f'**{len(recs)} recommendations** via `hybrid-{method}`')

                # GenAI explanation
                with st.expander("✨ AI Explanation (powered by Claude)", expanded=True):
                    explanation = genai_explain(user_id, recs, f"hybrid-{method}")
                    st.markdown(f'<div class="explain-box">{explanation}</div>', unsafe_allow_html=True)

                col_a, col_b = st.columns(2)
                for i, movie in enumerate(recs):
                    with (col_a if i % 2 == 0 else col_b):
                        render_movie_card(movie, i + 1)

                # Score chart
                titles = [m["title"][:25] + "…" if len(m.get("title","")) > 25 else m.get("title","?") for m in recs]
                scores = [m.get("score", 0) for m in recs]
                fig    = px.bar(
                    x=scores, y=titles, orientation="h",
                    labels={"x": "Hybrid Score", "y": ""},
                    color=scores, color_continuous_scale=["#2d2d5e", "#e94560"],
                    template="plotly_dark",
                    title="Recommendation Scores",
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False, height=350, margin=dict(l=0, r=0, t=40, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if st.button("Get CF Recommendations", key="cf_btn"):
            with st.spinner("Running collaborative filter …"):
                data = api_get(f"/recommend/{user_id}/cf", {"top_k": top_k, "method": method})
            if data:
                recs = data["recommendations"]
                col_a, col_b = st.columns(2)
                for i, movie in enumerate(recs):
                    with (col_a if i % 2 == 0 else col_b):
                        render_movie_card(movie, i + 1)

    with tab3:
        if st.button("Get Content-Based Recommendations", key="cb_btn"):
            with st.spinner("Running content-based filter …"):
                data = api_get(f"/recommend/{user_id}/cb", {"top_k": top_k})
            if data:
                recs = data["recommendations"]
                col_a, col_b = st.columns(2)
                for i, movie in enumerate(recs):
                    with (col_a if i % 2 == 0 else col_b):
                        render_movie_card(movie, i + 1)


# ──────────────────────────────────────────────────────────────────────────
# PAGE 2 — Explore Movies
# ──────────────────────────────────────────────────────────────────────────
elif "🔍" in page:
    st.markdown("# 🔍 Explore Movies")

    item_id = st.number_input("Movie ID", min_value=1, max_value=1682, value=50)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Show Movie Details"):
            data = api_get(f"/movies/{item_id}")
            if data:
                st.markdown(f"""
                <div class="movie-card">
                    <p class="movie-title">🎬 {data['title']}</p>
                    <p class="movie-meta">{data.get('genres','')} · {data.get('year','')}</p>
                </div>""", unsafe_allow_html=True)

    with col2:
        if st.button("Show Similar Movies"):
            data = api_get(f"/similar/items/{item_id}", {"top_k": 8})
            if data:
                for i, m in enumerate(data["similar_items"]):
                    render_movie_card(m, i + 1)

    st.divider()
    st.markdown("### 🔮 Explain a Recommendation")
    exp_uid = st.number_input("User ID", min_value=1, max_value=943, value=user_id, key="exp_uid")
    exp_iid = st.number_input("Movie ID", min_value=1, max_value=1682, value=item_id, key="exp_iid")
    if st.button("Explain Why"):
        data = api_get(f"/explain/{exp_uid}/{exp_iid}")
        if data:
            st.markdown(f"""
            <div class="explain-box">
            <b>🎬 Movie:</b> {data.get('title','?')}<br>
            <b>🎭 Genres:</b> {data.get('genres','')}<br>
            <b>🔑 Key Content Features:</b> {', '.join(data.get('key_features', []))}<br>
            <b>👥 Similar Users:</b> {', '.join([str(u) for u in data.get('similar_users', [])])}<br>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 👥 Find Similar Users")
    if st.button("Find Similar Users"):
        data = api_get(f"/similar/users/{user_id}", {"top_k": 8})
        if data:
            df = pd.DataFrame(data["similar_users"])
            fig = px.bar(df, x="user_id", y="similarity",
                         template="plotly_dark", color="similarity",
                         color_continuous_scale=["#2d2d5e", "#e94560"],
                         title=f"Users Most Similar to #{user_id}")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────
# PAGE 3 — EDA Dashboard
# ──────────────────────────────────────────────────────────────────────────
elif "📊" in page:
    st.markdown("# 📊 EDA Dashboard")

    data = api_get("/eda/summary")
    if data:
        cols = st.columns(4)
        metrics = [
            ("Users",        data["n_users"],        "👤"),
            ("Movies",       data["n_items"],         "🎬"),
            ("Ratings",      f'{data["n_ratings"]:,}',"⭐"),
            ("Sparsity",     f'{data["sparsity"]:.1%}',"🕳️"),
        ]
        for col, (label, value, icon) in zip(cols, metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:2rem">{icon}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=data["rating_mean"],
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Average Rating", "font": {"color": "#e8e8f0"}},
                gauge={
                    "axis": {"range": [1, 5], "tickcolor": "#8888aa"},
                    "bar":  {"color": "#e94560"},
                    "bgcolor": "#1a1a2e",
                    "steps": [
                        {"range": [1, 3], "color": "#2d2d5e"},
                        {"range": [3, 5], "color": "#16213e"},
                    ],
                },
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e8e8f0"}, height=280,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            kpis = {
                "Avg Ratings / User": data["ratings_per_user_mean"],
                "Avg Ratings / Movie": data["ratings_per_item_mean"],
                "Min Rating": data["min_rating"],
                "Max Rating": data["max_rating"],
                "Rating Std Dev": data["rating_std"],
            }
            for k, v in kpis.items():
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #2d2d5e;">
                    <span style="color:#8888aa">{k}</span>
                    <span style="color:#e94560;font-weight:600">{v}</span>
                </div>""", unsafe_allow_html=True)

    # Simulated distribution chart
    st.markdown("### ⭐ Rating Distribution")
    rating_data = {"Rating": [1, 2, 3, 4, 5], "Count": [6110, 11370, 27145, 34174, 21201]}
    fig = px.bar(
        rating_data, x="Rating", y="Count",
        template="plotly_dark", color="Count",
        color_continuous_scale=["#2d2d5e", "#e94560"],
        title="Distribution of Ratings (MovieLens 100K)",
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    # Genre popularity
    st.markdown("### 🎭 Genre Popularity")
    genre_counts = {
        "drama": 725, "comedy": 505, "action": 251, "thriller": 251, "romance": 247,
        "adventure": 135, "sci_fi": 101, "crime": 109, "horror": 92, "musical": 56,
        "mystery": 61, "war": 71, "documentary": 50, "animation": 42, "western": 27,
    }
    fig = px.bar(
        x=list(genre_counts.keys()), y=list(genre_counts.values()),
        template="plotly_dark", color=list(genre_counts.values()),
        color_continuous_scale=["#2d2d5e", "#e94560"],
        labels={"x": "Genre", "y": "Number of Movies"},
        title="Movie Count by Genre",
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────
# PAGE 4 — New User (Cold Start)
# ──────────────────────────────────────────────────────────────────────────
elif "🆕" in page:
    st.markdown("# 🆕 New User Recommendations")
    st.markdown('<p style="color:#8888aa;">No history? No problem. Tell us what you like!</p>', unsafe_allow_html=True)

    selected_genres = st.multiselect(
        "Choose your favourite genres:",
        GENRES,
        default=["drama", "comedy", "romance"],
    )
    top_k_cold = st.slider("Number of recommendations", 5, 20, 10)

    if st.button("✨ Get My Recommendations") and selected_genres:
        with st.spinner("Finding movies just for you …"):
            data = api_post("/cold-start", {"genres": selected_genres, "top_k": top_k_cold})
        if data:
            recs = data["recommendations"]
            st.success(f"Found {len(recs)} movies matching your taste!")

            # GenAI explanation
            explanation = genai_explain(0, recs, "cold-start content-based")
            st.markdown(f'<div class="explain-box">✨ {explanation}</div>', unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            for i, movie in enumerate(recs):
                with (col_a if i % 2 == 0 else col_b):
                    render_movie_card(movie, i + 1)
    elif not selected_genres:
        st.info("👆 Please select at least one genre.")
