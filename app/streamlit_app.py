"""Streamlit front-end for the music recommender.

Run locally with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from music_recommender import ContentRecommender
from music_recommender.config import PROFILE_FEATURES, REPORTS_DIR
from music_recommender.data import ensure_tracks

st.set_page_config(page_title="Music Recommender", page_icon="🎧", layout="wide")


@st.cache_data(show_spinner="Loading and cleaning the track catalogue…")
def load_tracks() -> pd.DataFrame:
    return ensure_tracks()


@st.cache_resource(show_spinner="Building track vectors…")
def load_model(genre_weight: float) -> ContentRecommender:
    return ContentRecommender(genre_weight=genre_weight).fit(load_tracks())


def track_label(row: pd.Series) -> str:
    return f"{row.track_name} — {row.artists}  ·  {', '.join(row.genres)}"


def profile_chart(seeds: pd.DataFrame, recs: pd.DataFrame) -> alt.Chart:
    data = (
        pd.concat(
            [
                seeds[PROFILE_FEATURES].mean().rename("Your seeds"),
                recs[PROFILE_FEATURES].mean().rename("Recommendations"),
            ],
            axis=1,
        )
        .reset_index(names="feature")
        .melt("feature", var_name="group", value_name="value")
    )
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=3, size=12)
        .encode(
            y=alt.Y("feature:N", title=None, sort=PROFILE_FEATURES),
            yOffset=alt.YOffset("group:N", sort=["Your seeds", "Recommendations"]),
            x=alt.X("value:Q", title="Mean value (0–1)", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "group:N",
                title=None,
                sort=["Your seeds", "Recommendations"],
                scale=alt.Scale(range=["#2a78d6", "#eb6834"]),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=["group", "feature", alt.Tooltip("value:Q", format=".2f")],
        )
        .properties(height=320)
    )


# ------------------------------------------------------------------ sidebar
st.sidebar.header("Settings")
k = st.sidebar.slider("Number of recommendations", 5, 30, 10)
diversity = st.sidebar.slider(
    "Diversity (MMR λ)",
    0.0,
    0.9,
    0.3,
    0.05,
    help="0 = most similar tracks only. Higher values trade similarity for variety.",
)
genre_weight = st.sidebar.slider(
    "Genre weight",
    0.0,
    4.0,
    1.5,
    0.25,
    help="0 = match purely on how tracks sound (audio features).",
)
exclude_artists = st.sidebar.checkbox("Only new artists", value=False)
allow_explicit = st.sidebar.checkbox("Allow explicit tracks", value=True)
min_pop = st.sidebar.slider("Minimum popularity", 0, 100, 0)

rec = load_model(genre_weight)
genre_filter = st.sidebar.multiselect("Restrict to genres", rec.features.genres)

# ---------------------------------------------------------------------- main
st.title("🎧 Music Recommender")
st.caption(
    f"Content-based recommendations over {len(rec.tracks_):,} Spotify tracks "
    f"in {len(rec.features.genres)} genres, using audio features + genres."
)

tab_rec, tab_eval = st.tabs(["Recommend", "How well does it work?"])

with tab_rec:
    if "seeds" not in st.session_state:
        st.session_state.seeds = []

    col_search, col_seeds = st.columns([3, 2])
    with col_search:
        query = st.text_input("Search for a song or artist", placeholder="e.g. Blinding Lights")
        matches = rec.search(query, limit=25) if query else rec.tracks_.head(0)
        if query and matches.empty:
            st.info("No matches. Try fewer words or a different spelling.")
        if not matches.empty:
            choice = st.selectbox(
                "Matches", matches.index, format_func=lambda i: track_label(matches.loc[i])
            )
            if st.button("➕ Add as seed", type="primary"):
                tid = matches.loc[choice, "track_id"]
                if tid not in st.session_state.seeds:
                    st.session_state.seeds.append(tid)

    with col_seeds:
        st.subheader("Seed tracks")
        if not st.session_state.seeds:
            st.write("Add one or more songs you like. Several seeds act like a playlist.")
        for tid in list(st.session_state.seeds):
            row = rec.tracks_.loc[rec.index_[tid]]
            c1, c2 = st.columns([5, 1])
            c1.write(f"**{row.track_name}** — {row.artists}")
            if c2.button("✕", key=f"rm-{tid}"):
                st.session_state.seeds.remove(tid)
                st.rerun()

    if st.session_state.seeds:
        seeds = st.session_state.seeds
        recs = rec.recommend(
            seeds,
            k=k,
            diversity=diversity,
            exclude_seed_artists=exclude_artists,
            allow_explicit=allow_explicit,
            min_popularity=min_pop,
            genres=genre_filter or None,
        )
        if recs.empty:
            st.warning("No tracks match these filters.")
        else:
            reasons = [rec.explain_text(seeds, tid) for tid in recs["track_id"]]
            recs["why"] = reasons
            recs["link"] = "https://open.spotify.com/track/" + recs["track_id"]

            st.subheader("Recommendations")
            st.dataframe(
                recs[["track_name", "artists", "similarity", "popularity", "why", "link"]],
                hide_index=True,
                width="stretch",
                column_config={
                    "track_name": "Track",
                    "artists": "Artists",
                    "similarity": st.column_config.ProgressColumn(
                        "Similarity", min_value=0.0, max_value=1.0, format="%.2f"
                    ),
                    "popularity": st.column_config.NumberColumn("Popularity", width="small"),
                    "why": st.column_config.TextColumn("Why it matches", width="large"),
                    "link": st.column_config.LinkColumn("Spotify", display_text="Open ↗"),
                },
            )
            left, right = st.columns([2, 3])
            with left:
                preview = st.selectbox(
                    "Preview a recommendation",
                    recs.index,
                    format_func=lambda i: f"{recs.loc[i, 'track_name']} — {recs.loc[i, 'artists']}",
                )
                st.iframe(
                    f"https://open.spotify.com/embed/track/{recs.loc[preview, 'track_id']}",
                    height=152,
                )
            with right:
                st.markdown("**Audio profile: seeds vs. recommendations**")
                seed_rows = rec.tracks_.loc[rec.to_indices(seeds)]
                st.altair_chart(profile_chart(seed_rows, recs), width="stretch")

with tab_eval:
    st.markdown(
        """
**Evaluation task — artist continuation.** For 1,000 random artists the model is shown
3 of their songs and must find the artist's *other* songs in its top 10, out of the whole
catalogue. The model never sees who the artist is, so it can only succeed by capturing
their sound and genre. Hyper-parameters were tuned on a separate validation draw.
"""
    )
    csv = REPORTS_DIR / "model_comparison.csv"
    if csv.exists():
        table = pd.read_csv(csv, index_col=0)
        st.dataframe(
            table[
                ["hit_rate", "ndcg", "genre_precision", "diversity", "coverage", "avg_popularity"]
            ]
            .rename(
                columns={
                    "hit_rate": "HitRate@10",
                    "ndcg": "NDCG@10",
                    "genre_precision": "Genre precision",
                    "diversity": "Diversity",
                    "coverage": "Coverage",
                    "avg_popularity": "Avg popularity",
                }
            )
            .style.format(precision=3),
            width="stretch",
        )
    for fig in ("model_comparison", "genre_weight_sweep", "diversity_sweep"):
        path = Path(REPORTS_DIR / "figures" / f"{fig}.png")
        if path.exists():
            st.image(str(path))
