"""Run the full evaluation protocol and write a markdown report + figures.

Protocol
--------
* **Validation tasks** (seed 7) are used for hyper-parameter sweeps
  (genre weight / representation, MMR diversity).
* **Test tasks** (seed 42, disjoint random draw) are used once to report the
  final comparison against baselines, so reported numbers are not tuned on.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from music_recommender.config import REPORTS_DIR  # noqa: E402
from music_recommender.evaluation import (  # noqa: E402
    ContentRecommender,
    Evaluator,
    default_model_suite,
    diversity_sweep,
    make_artist_tasks,
    weight_sweep,
)

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="both", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _figure(*args, **kw):
    fig, axes = plt.subplots(*args, **kw)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def plot_model_comparison(results: pd.DataFrame, path: Path) -> None:
    df = results.sort_values("ndcg")
    fig, ax = _figure(figsize=(8, 4.2), dpi=160)
    _style(ax)
    ax.grid(axis="y", visible=False)
    colors = [ORANGE if "final" in n else BLUE for n in df.index]
    ax.barh(
        df.index,
        df["ndcg"],
        xerr=df["ndcg_ci95"],
        color=colors,
        height=0.55,
        error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3},
    )
    for y, (v, ci) in enumerate(zip(df["ndcg"], df["ndcg_ci95"], strict=True)):
        ax.text(v + ci + 0.003, y, f"{v:.3f}", va="center", fontsize=9, color=INK)
    ax.set_xlabel("NDCG@10 (artist continuation, test set, 95% CI)", color=INK_2, fontsize=9)
    ax.set_title("Content model vs. baselines", loc="left", color=INK, fontsize=12)
    ax.tick_params(axis="y", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_weight_sweep(sweep: pd.DataFrame, path: Path) -> None:
    fig, ax = _figure(figsize=(7, 4), dpi=160)
    _style(ax)
    series = (
        ("onehot", BLUE, "One-hot genres", 8),
        ("embedding", ORANGE, "Genre embedding (PPMI+SVD)", -14),
    )
    for mode, color, label, dy in series:
        d = sweep[sweep["genre_mode"] == mode]
        ax.plot(
            d["genre_weight"],
            d["ndcg"],
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            label=label,
        )
        ax.annotate(
            label,
            (d["genre_weight"].iloc[-1], d["ndcg"].iloc[-1]),
            xytext=(0, dy),
            textcoords="offset points",
            ha="right",
            fontsize=9,
            color=INK,
        )
    ax.set_xlabel("Genre weight (0 = audio features only)", color=INK_2, fontsize=9)
    ax.set_ylabel("NDCG@10 (validation)", color=INK_2, fontsize=9)
    ax.set_title("How much should genre matter?", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_diversity_sweep(sweep: pd.DataFrame, path: Path) -> None:
    fig, axes = _figure(1, 2, figsize=(9, 3.6), dpi=160, sharex=True)
    panels = (("hit_rate", "HitRate@10 (validation)"), ("diversity", "Intra-list diversity"))
    for ax, (col, title) in zip(axes, panels, strict=True):
        _style(ax)
        ax.plot(
            sweep["diversity_lambda"], sweep[col], color=BLUE, linewidth=2, marker="o", markersize=5
        )
        ax.axvline(0.3, color=ORANGE, linewidth=1.2, linestyle="--")
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_xlabel("MMR diversity λ", color=INK_2, fontsize=9)
    axes[0].annotate(
        "chosen λ = 0.3",
        (0.3, sweep["hit_rate"].min()),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=9,
        color=INK,
    )
    fig.suptitle(
        "Diversity re-ranking: accuracy vs. variety", x=0.01, ha="left", color=INK, fontsize=12
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _md_table(df: pd.DataFrame, cols: dict[str, str], index_name: str | None = None) -> str:
    out = df[list(cols)].rename(columns=cols)
    if index_name:
        out = out.rename_axis(index_name).reset_index()
    return out.to_markdown(index=False, floatfmt=".3f")


def run(
    tracks: pd.DataFrame, out_dir: Path = REPORTS_DIR, n_tasks: int = 1000, k: int = 10
) -> pd.DataFrame:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    val = Evaluator(tracks, make_artist_tasks(tracks, n_tasks=n_tasks, seed=7), k=k)
    test = Evaluator(tracks, make_artist_tasks(tracks, n_tasks=n_tasks, seed=42), k=k)

    w_sweep = weight_sweep(val, tracks)
    d_sweep = diversity_sweep(val, ContentRecommender(diversity=0.0).fit(tracks))
    results = test.compare(default_model_suite(tracks))

    results.to_csv(out_dir / "model_comparison.csv")
    w_sweep.to_csv(out_dir / "genre_weight_sweep.csv", index=False)
    d_sweep.to_csv(out_dir / "diversity_sweep.csv", index=False)
    plot_model_comparison(results, fig_dir / "model_comparison.png")
    plot_weight_sweep(w_sweep, fig_dir / "genre_weight_sweep.png")
    plot_diversity_sweep(d_sweep, fig_dir / "diversity_sweep.png")

    metric_cols = {
        "hit_rate": "HitRate@10",
        "recall": "Recall@10",
        "ndcg": "NDCG@10",
        "ndcg_ci95": "NDCG ±95% CI",
        "genre_precision": "Genre precision",
        "diversity": "Diversity",
        "coverage": "Coverage",
        "avg_popularity": "Avg popularity",
    }
    md = [
        "# Offline evaluation",
        "",
        "Generated by `music-rec evaluate`. Do not edit by hand.",
        "",
        f"**Task:** artist continuation - {n_tasks} artists with ≥6 songs; the model sees 3 "
        f"songs and must retrieve the artist's hidden songs in its top {k} out of "
        f"{len(tracks):,} tracks. Artist is never a model feature.",
        "",
        "**Protocol:** hyper-parameters were tuned on a *validation* draw of artists (seed 7); "
        "the comparison below is on a separate *test* draw (seed 42).",
        "",
        "## Model comparison (test set)",
        "",
        _md_table(results, metric_cols, "Model"),
        "",
        "![model comparison](figures/model_comparison.png)",
        "",
        "## Genre weight and representation (validation set)",
        "",
        _md_table(
            w_sweep,
            {
                "genre_mode": "Genres",
                "genre_weight": "Weight",
                **{c: metric_cols[c] for c in ("hit_rate", "ndcg", "genre_precision", "diversity")},
            },
        ),
        "",
        "![genre weight sweep](figures/genre_weight_sweep.png)",
        "",
        "## MMR diversity (validation set)",
        "",
        _md_table(
            d_sweep,
            {
                "diversity_lambda": "λ",
                **{c: metric_cols[c] for c in ("hit_rate", "ndcg", "diversity", "coverage")},
            },
        ),
        "",
        "![diversity sweep](figures/diversity_sweep.png)",
        "",
        "## Metric definitions",
        "",
        "- **HitRate@10** - share of tasks where at least one hidden song is recommended.",
        "- **Recall@10** - hidden songs retrieved / min(#hidden, 10).",
        "- **NDCG@10** - rank-aware: hits near the top count more.",
        "- **Genre precision** - share of recommendations sharing a genre with the seeds.",
        "- **Diversity** - mean pairwise cosine distance of recommendations in audio-feature space.",
        "- **Coverage** - share of the catalogue recommended at least once across all tasks.",
        "- **Avg popularity** - mean Spotify popularity (0-100) of recommendations; lower = less popularity bias.",
        "",
    ]
    (out_dir / "evaluation.md").write_text("\n".join(md))
    return results
