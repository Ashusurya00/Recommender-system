"""
Full EDA + Model Evaluation Pipeline
======================================
Generates all charts, analysis, and model comparison results.
Run this once to produce:
  - static/eda_plots.html  (interactive Plotly report)
  - static/model_comparison.csv
  - static/evaluation_report.txt
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from data.data_loader import MovieLensLoader
from models.collaborative_filter import CollaborativeFilter
from models.content_based import ContentBasedFilter
from models.hybrid import HybridRecommender
from evaluation.metrics import Evaluator

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_eda(loader: MovieLensLoader):
    """Generate EDA plots and return a combined HTML report."""
    ratings = loader.get_ratings()
    movies  = loader.get_movies()
    users   = loader.get_users()
    summary = loader.eda_summary()

    figs = []

    # ── 1. Rating distribution ───────────────────────────────────────────
    rc = ratings["rating"].value_counts().sort_index()
    fig1 = px.bar(
        x=rc.index, y=rc.values,
        labels={"x": "Rating", "y": "Count"},
        title="⭐ Rating Distribution",
        color=rc.values,
        color_continuous_scale=["#2d2d5e", "#e94560"],
        template="plotly_dark",
    )
    fig1.update_layout(coloraxis_showscale=False, paper_bgcolor="#0d0d14", plot_bgcolor="#0d0d14")
    figs.append(("Rating Distribution", fig1))

    # ── 2. Ratings per user (histogram) ──────────────────────────────────
    rpu = ratings.groupby("user_id").size()
    fig2 = px.histogram(
        rpu, nbins=40,
        labels={"value": "# Ratings", "count": "# Users"},
        title="👤 Ratings per User",
        template="plotly_dark", color_discrete_sequence=["#e94560"],
    )
    fig2.update_layout(paper_bgcolor="#0d0d14", plot_bgcolor="#0d0d14")
    figs.append(("Ratings per User", fig2))

    # ── 3. Ratings per item (histogram) ──────────────────────────────────
    rpi = ratings.groupby("item_id").size()
    fig3 = px.histogram(
        rpi, nbins=40,
        labels={"value": "# Ratings", "count": "# Movies"},
        title="🎬 Ratings per Movie",
        template="plotly_dark", color_discrete_sequence=["#a855f7"],
    )
    fig3.update_layout(paper_bgcolor="#0d0d14", plot_bgcolor="#0d0d14")
    figs.append(("Ratings per Movie", fig3))

    # ── 4. Genre popularity ───────────────────────────────────────────────
    genre_cols = [
        "action","adventure","animation","children","comedy","crime",
        "documentary","drama","fantasy","film_noir","horror","musical",
        "mystery","romance","sci_fi","thriller","war","western",
    ]
    available_genre_cols = [c for c in genre_cols if c in movies.columns]
    genre_counts = movies[available_genre_cols].sum().sort_values(ascending=False)
    fig4 = px.bar(
        x=genre_counts.index, y=genre_counts.values,
        labels={"x": "Genre", "y": "# Movies"},
        title="🎭 Genre Popularity",
        color=genre_counts.values,
        color_continuous_scale=["#2d2d5e", "#e94560"],
        template="plotly_dark",
    )
    fig4.update_layout(coloraxis_showscale=False, paper_bgcolor="#0d0d14", plot_bgcolor="#0d0d14")
    figs.append(("Genre Popularity", fig4))

    # ── 5. Average rating by genre ────────────────────────────────────────
    merged = loader.get_merged()
    genre_avg = {}
    for g in available_genre_cols:
        if g in merged.columns:
            mask = merged[g] == 1
            if mask.sum() > 10:
                genre_avg[g] = merged.loc[mask, "rating"].mean()
    ga_df = pd.Series(genre_avg).sort_values(ascending=False)
    ga_frame = pd.DataFrame({"genre": list(ga_df.index), "rating": list(ga_df.values)})
    fig5 = px.bar(
        ga_frame, x="genre", y="rating",
        labels={"genre": "Genre", "rating": "Avg Rating"},
        title="⭐ Average Rating by Genre",
        color="rating",
        color_continuous_scale=["#2d2d5e", "#e94560"],
        template="plotly_dark",
    )
    fig5.update_layout(coloraxis_showscale=False, paper_bgcolor="#0d0d14", plot_bgcolor="#0d0d14")
    figs.append(("Avg Rating by Genre", fig5))

    # ── 6. Ratings over time ──────────────────────────────────────────────
    ratings_copy = ratings.copy()
    ratings_copy["month"] = ratings_copy["timestamp"].dt.to_period("M").astype(str)
    rt = ratings_copy.groupby("month").size().reset_index(name="count")
    rt = rt[rt["count"] > 0]
    fig6 = px.line(
        rt, x="month", y="count",
        labels={"month": "Month", "count": "# Ratings"},
        title="📈 Ratings Over Time",
        template="plotly_dark", color_discrete_sequence=["#e94560"],
    )
    fig6.update_layout(paper_bgcolor="#0d0d14", plot_bgcolor="#0d0d14")
    figs.append(("Ratings Over Time", fig6))

    # ── 7. User age distribution ──────────────────────────────────────────
    if "age" in users.columns:
        fig7 = px.histogram(
            users, x="age", nbins=20,
            title="👥 User Age Distribution",
            template="plotly_dark", color_discrete_sequence=["#a855f7"],
        )
        fig7.update_layout(paper_bgcolor="#0d0d14", plot_bgcolor="#0d0d14")
        figs.append(("User Age Distribution", fig7))

    # ── 8. Rating heatmap: age group vs genre ────────────────────────────
    if "age_group" in users.columns and available_genre_cols:
        merged2 = loader.get_merged()
        if "age_group" in merged2.columns and available_genre_cols[0] in merged2.columns:
            heat = merged2.groupby("age_group")["rating"].mean().reset_index()
            fig8 = px.bar(
                heat, x="age_group", y="rating",
                title="📊 Average Rating by Age Group",
                template="plotly_dark", color="rating",
                color_continuous_scale=["#2d2d5e", "#e94560"],
            )
            fig8.update_layout(coloraxis_showscale=False,
                               paper_bgcolor="#0d0d14", plot_bgcolor="#0d0d14")
            figs.append(("Rating by Age Group", fig8))

    # ── Combine into HTML report ──────────────────────────────────────────
    html_parts = ["""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>CineMatch EDA Report</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400&display=swap');
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#0d0d14; color:#e8e8f0; font-family:'DM Sans',sans-serif; padding:40px; }
  h1   { font-family:'Playfair Display',serif; font-size:2.5rem; color:#e94560; margin-bottom:8px; }
  .subtitle { color:#8888aa; margin-bottom:40px; }
  .kpi-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:40px; }
  .kpi { background:#1a1a2e; border:1px solid #2d2d5e; border-radius:12px; padding:20px; text-align:center; }
  .kpi-val { font-size:2rem; font-weight:700; color:#e94560; }
  .kpi-lbl { font-size:0.8rem; color:#8888aa; text-transform:uppercase; letter-spacing:1px; margin-top:4px; }
  .chart-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:24px; }
  .chart-box { background:#1a1a2e; border:1px solid #2d2d5e; border-radius:12px; padding:16px; }
  .model-table { width:100%; border-collapse:collapse; margin-top:40px; }
  .model-table th { background:#e94560; color:#fff; padding:12px 16px; text-align:left; }
  .model-table td { padding:10px 16px; border-bottom:1px solid #2d2d5e; color:#c0c0d0; }
  .model-table tr:nth-child(even) td { background:#111128; }
  .best { color:#e94560; font-weight:700; }
  h2 { font-family:'Playfair Display',serif; font-size:1.5rem; margin:40px 0 16px; }
</style>
</head>
<body>
<h1>🎬 CineMatch EDA Report</h1>
<p class="subtitle">MovieLens 100K — Exploratory Data Analysis & Model Evaluation</p>
"""]

    # KPI summary
    html_parts.append(f"""
<div class="kpi-grid">
  <div class="kpi"><div class="kpi-val">{summary['n_users']:,}</div><div class="kpi-lbl">Users</div></div>
  <div class="kpi"><div class="kpi-val">{summary['n_items']:,}</div><div class="kpi-lbl">Movies</div></div>
  <div class="kpi"><div class="kpi-val">{summary['n_ratings']:,}</div><div class="kpi-lbl">Ratings</div></div>
  <div class="kpi"><div class="kpi-val">{summary['sparsity']:.1%}</div><div class="kpi-lbl">Sparsity</div></div>
  <div class="kpi"><div class="kpi-val">{summary['rating_mean']:.2f}</div><div class="kpi-lbl">Avg Rating</div></div>
  <div class="kpi"><div class="kpi-val">{summary['rating_std']:.2f}</div><div class="kpi-lbl">Rating Std</div></div>
  <div class="kpi"><div class="kpi-val">{summary['ratings_per_user_mean']:.1f}</div><div class="kpi-lbl">Ratings/User</div></div>
  <div class="kpi"><div class="kpi-val">{summary['ratings_per_item_mean']:.1f}</div><div class="kpi-lbl">Ratings/Movie</div></div>
</div>
<h2>📊 Data Visualizations</h2>
<div class="chart-grid">
""")

    for title, fig in figs:
        fig_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id=title.replace(" ", "_"))
        html_parts.append(f'<div class="chart-box">{fig_html}</div>')

    html_parts.append("</div>")  # close chart-grid
    html_parts.append("</body></html>")

    report_html = "\n".join(html_parts)
    out_path = os.path.join(OUTPUT_DIR, "eda_report.html")
    with open(out_path, "w") as f:
        f.write(report_html)
    print(f"EDA report saved → {out_path}")
    return summary


def run_model_comparison(loader: MovieLensLoader):
    """Train all models and compare evaluation metrics."""
    from evaluation.metrics import Evaluator, precision_at_k, recall_at_k, ndcg_at_k

    ev = Evaluator(loader.ratings, threshold=3.0, k=10)

    def make_cf_uu(train_df):
        cf = CollaborativeFilter(train_df, n_factors=50)
        return lambda uid, k: cf.recommend_user_user(uid, k)

    def make_cf_ii(train_df):
        cf = CollaborativeFilter(train_df, n_factors=50)
        return lambda uid, k: cf.recommend_item_item(uid, k)

    def make_cf_svd(train_df):
        cf = CollaborativeFilter(train_df, n_factors=50)
        return lambda uid, k: cf.recommend_svd(uid, k)

    def make_cb(train_df):
        cb = ContentBasedFilter(loader.movies, train_df)
        return lambda uid, k: cb.recommend(uid, k)

    def make_hybrid(train_df):
        cf = CollaborativeFilter(train_df, n_factors=50)
        cb = ContentBasedFilter(loader.movies, train_df)
        h  = HybridRecommender(cf, cb, loader.movies)
        return lambda uid, k: h.recommend(uid, top_k=k, method="svd")

    factories = {
        "CF-UserUser":  make_cf_uu,
        "CF-ItemItem":  make_cf_ii,
        "CF-SVD":       make_cf_svd,
        "ContentBased": make_cb,
        "Hybrid-SVD":   make_hybrid,
    }

    print("Running model comparison (80 users) …")
    results = ev.compare_models_from_train(factories, n_users=80)

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    results.to_csv(csv_path)
    print(f"Model comparison CSV → {csv_path}")
    print(results.to_string())

    # Plot comparison
    df_plot = results.reset_index()
    fig = go.Figure()
    metrics_cols = [c for c in results.columns if c != "n_users_eval"]
    colors = ["#e94560", "#a855f7", "#06b6d4"]
    for i, metric in enumerate(metrics_cols):
        fig.add_trace(go.Bar(
            name=metric,
            x=df_plot["model"],
            y=df_plot[metric],
            marker_color=colors[i % len(colors)],
        ))
    fig.update_layout(
        barmode="group",
        title="📊 Model Comparison: P@10, R@10, NDCG@10",
        template="plotly_dark",
        paper_bgcolor="#0d0d14",
        plot_bgcolor="#0d0d14",
        xaxis_title="Model",
        yaxis_title="Score",
        legend_title="Metric",
    )

    # Append to EDA report
    eda_path = os.path.join(OUTPUT_DIR, "eda_report.html")
    if os.path.exists(eda_path):
        with open(eda_path, "r") as f:
            content = f.read()
        model_html = f"""
<h2>🏆 Model Comparison</h2>
{fig.to_html(full_html=False, include_plotlyjs=False, div_id='model_comparison')}
<table class="model-table">
  <thead><tr><th>Model</th>{''.join(f'<th>{c}</th>' for c in results.columns)}</tr></thead>
  <tbody>
"""
        best_p    = results["P@10"].idxmax()
        best_ndcg = results["NDCG@10"].idxmax()
        for model, row in results.iterrows():
            model_html += "<tr>"
            is_best = model in (best_p, best_ndcg)
            cls = ' class="best"' if is_best else ""
            model_html += f"<td{cls}>{model}{'  ★' if is_best else ''}</td>"
            for col in results.columns:
                v = row[col]
                model_html += f"<td{cls}>{v}</td>"
            model_html += "</tr>\n"
        model_html += "</tbody></table>"

        # Insert before </body>
        content = content.replace("</body>", model_html + "\n</body>")
        with open(eda_path, "w") as f:
            f.write(content)
        print("Model comparison appended to EDA report.")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("  CineMatch — EDA & Evaluation Pipeline")
    print("=" * 60)

    print("\n1. Loading data …")
    loader = MovieLensLoader()

    print("\n2. Running EDA …")
    summary = run_eda(loader)

    print("\n3. Running model evaluation …")
    results = run_model_comparison(loader)

    print("\n✅ Pipeline complete!")
    print(f"   EDA report  → static/eda_report.html")
    print(f"   Model CSV   → static/model_comparison.csv")
