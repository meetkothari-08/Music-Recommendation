"""Streamlit front-end: AI DJ set builder + similar-song recommender.

Run locally with:  python -m streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from music_recommender import ContentRecommender
from music_recommender.coach import PROVIDERS, DJCoach, LLMClient
from music_recommender.config import PROFILE_FEATURES, REPORTS_DIR
from music_recommender.data import ensure_tracks
from music_recommender.dj import ARCS, MOOD_DESCRIPTIONS, MOODS, DJSetBuilder, SetRequest

st.set_page_config(page_title="AI DJ & Music Recommender", page_icon="🎧", layout="wide")

BLUE, ORANGE, GRAY, INK_2 = "#2a78d6", "#eb6834", "#a8a7a2", "#52514e"
RATING_ICON = {"smooth": "🟢", "ok": "🟡", "tricky": "🔴"}


@st.cache_data(show_spinner="Loading and cleaning the track catalogue…")
def load_tracks() -> pd.DataFrame:
    return ensure_tracks()


@st.cache_resource(show_spinner="Building track vectors…")
def load_model(genre_weight: float = 1.5) -> ContentRecommender:
    return ContentRecommender(genre_weight=genre_weight).fit(load_tracks())


@st.cache_resource(show_spinner=False)
def load_builder() -> DJSetBuilder:
    return DJSetBuilder(load_model())


@st.cache_data(show_spinner=False)
def mood_background(n: int = 4000) -> pd.DataFrame:
    t = load_tracks()
    return t.sample(min(n, len(t)), random_state=0)[["valence", "energy"]]


def secret(name: str) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets.get(name)
    except Exception:  # no secrets file
        return None


def get_coach(choice: str, typed_key: str) -> DJCoach:
    if choice == "built-in":
        return DJCoach(None)
    key = typed_key.strip() or secret(PROVIDERS[choice]["key_env"])
    if not key:
        return DJCoach(None)
    cache = st.session_state.setdefault("llm_clients", {})
    if (choice, key) not in cache:
        cache[(choice, key)] = LLMClient(choice, key, secret("MUSIC_REC_LLM_MODEL"))
    return DJCoach(cache[(choice, key)])


def track_label(row: pd.Series) -> str:
    return f"{row.track_name} — {row.artists}  ·  {', '.join(row.genres)}"


# ----------------------------------------------------------------- charts
def mood_plane_chart(dj_set) -> alt.LayerChart:
    bg = (
        alt.Chart(mood_background())
        .mark_circle(size=6, opacity=0.25, color=GRAY)
        .encode(
            x=alt.X("valence:Q", title="Valence (sad → happy)", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("energy:Q", title="Energy (calm → intense)", scale=alt.Scale(domain=[0, 1])),
        )
    )
    labels = pd.DataFrame([{"mood": m, "valence": v, "energy": e} for m, (v, e) in MOODS.items()])
    mood_text = (
        alt.Chart(labels)
        .mark_text(fontSize=11, color=INK_2, fontWeight="bold")
        .encode(x="valence:Q", y="energy:Q", text="mood:N")
    )
    target = pd.DataFrame(
        {
            "valence": dj_set.targets[:, 0],
            "energy": dj_set.targets[:, 1],
            "position": range(1, len(dj_set.targets) + 1),
        }
    )
    target_line = (
        alt.Chart(target)
        .mark_line(color=GRAY, strokeDash=[5, 4], strokeWidth=2)
        .encode(x="valence:Q", y="energy:Q", order="position:Q")
    )
    t = dj_set.tracks
    path = alt.Chart(t).encode(x="valence:Q", y="energy:Q", order="position:Q")
    line = path.mark_line(color=BLUE, strokeWidth=2)
    points = path.mark_circle(size=140, color=BLUE, stroke="white", strokeWidth=2).encode(
        tooltip=[
            "position",
            "track_name",
            "artists",
            "bpm",
            "camelot",
            "mood",
            alt.Tooltip("valence:Q", format=".2f"),
            alt.Tooltip("energy:Q", format=".2f"),
        ]
    )
    nums = path.mark_text(fontSize=9, color="white", fontWeight="bold").encode(text="position:Q")
    return (bg + mood_text + target_line + line + points + nums).properties(height=380)


def timeline_chart(dj_set, field: str, title: str, target: str | None = None) -> alt.LayerChart:
    t = dj_set.tracks
    base = alt.Chart(t).encode(x=alt.X("position:O", title="Track", axis=alt.Axis(labelAngle=0)))
    line = base.mark_line(color=BLUE, strokeWidth=2, point=alt.OverlayMarkDef(size=60)).encode(
        y=alt.Y(f"{field}:Q", title=title, scale=alt.Scale(zero=False)),
        tooltip=["position", "track_name", field],
    )
    layers = [line]
    if target:
        layers.insert(0, base.mark_line(color=GRAY, strokeDash=[5, 4]).encode(y=f"{target}:Q"))
    return alt.layer(*layers).properties(height=200)


# ---------------------------------------------------------------- sidebar
st.sidebar.header("🤖 AI coach")
provider_names = {"built-in": "Built-in coach (no key)"} | {
    k: f"{v['label']} (free API key)" for k, v in PROVIDERS.items()
}
default_provider = next((k for k, v in PROVIDERS.items() if secret(v["key_env"])), "built-in")
choice = st.sidebar.selectbox(
    "Coach",
    list(provider_names),
    index=list(provider_names).index(default_provider),
    format_func=provider_names.get,
)
typed_key = ""
if choice != "built-in":
    has_saved = bool(secret(PROVIDERS[choice]["key_env"]))
    typed_key = st.sidebar.text_input(
        "API key",
        type="password",
        placeholder="Using saved key" if has_saved else "Paste your key",
        help="Used only for this browser session; never stored.",
    )
    st.sidebar.caption(
        "Free keys: [Google AI Studio](https://aistudio.google.com/apikey) · "
        "[Groq console](https://console.groq.com/keys)"
    )
coach = get_coach(choice, typed_key)
st.sidebar.caption(f"Active: **{coach.source}**")
st.sidebar.caption(
    "The AI only explains. Song choices and every number come from the code, "
    "and AI tips that mention numbers the code didn't compute are rejected."
)

rec = load_model()
builder = load_builder()

# ------------------------------------------------------------------- main
st.title("🎧 AI DJ & Music Recommender")
st.caption(
    f"{len(rec.tracks_):,} Spotify tracks in {len(rec.features.genres)} genres. "
    "Build a DJ set that follows a mood journey with smooth, beatmatched transitions, "
    "and learn how to mix it."
)
tab_dj, tab_rec, tab_eval = st.tabs(
    ["🎛️ DJ Set Builder", "🔎 Similar songs", "📊 How well does it work?"]
)

# ------------------------------------------------------------- DJ tab
with tab_dj:
    mode = st.radio(
        "How do you want to plan the set?", ["Describe it", "Choose settings"], horizontal=True
    )
    request: SetRequest | None = None

    if mode == "Describe it":
        text = st.text_area(
            "Describe your set",
            placeholder="e.g. chill house set for a sunset that builds into a party, about 40 minutes",
            height=80,
        )
        c1, c2 = st.columns([1, 3])
        strict = c2.checkbox("Only beatmatchable transitions", value=True, key="strict_text")
        if c1.button("✨ Plan my set", type="primary", disabled=not text.strip()):
            with st.spinner(f"Planning with {coach.source}…"):
                plan = coach.plan(text, rec.features.genres)
            st.session_state.dj_plan = plan
            request = plan.to_request(strict=strict)
        if plan := st.session_state.get("dj_plan"):
            p = plan.params
            st.info(
                f"**Plan ({plan.source}):** {p['start_mood']} → {p['end_mood']} · "
                f"{p['n_tracks']} tracks · {p['arc']} arc · genres: {', '.join(p['genres']) or 'any'}"
                + "".join(f"\n\n_{n}_" for n in plan.notes)
            )
    else:
        c1, c2, c3 = st.columns(3)
        start_opts = ["(start from a song)", *MOODS]
        start = c1.selectbox(
            "Start mood",
            start_opts,
            index=start_opts.index("chill"),
            format_func=lambda m: m if m not in MOODS else f"{m} — {MOOD_DESCRIPTIONS[m]}",
        )
        end = c2.selectbox(
            "End mood",
            list(MOODS),
            index=list(MOODS).index("happy"),
            format_func=lambda m: f"{m} — {MOOD_DESCRIPTIONS[m]}",
        )
        n_tracks = c3.slider("Tracks", 4, 25, 10)
        c4, c5, c6 = st.columns(3)
        arc = c4.radio("Energy shape", list(ARCS), format_func=lambda a: f"{a}: {ARCS[a]}")
        genres = c5.multiselect(
            "Genres (optional)",
            rec.features.genres,
            help="Leave empty to let the set pick a style.",
        )
        min_pop = c6.slider("Minimum popularity", 0, 90, 0, help="Higher = better-known songs")
        strict = c6.checkbox("Only beatmatchable transitions", value=True)
        seed_id = None
        if start == "(start from a song)":
            q = st.text_input("Opening song", placeholder="e.g. Blinding Lights")
            hits = rec.search(q, limit=20) if q else rec.tracks_.head(0)
            if not hits.empty:
                pick = st.selectbox(
                    "Match", hits.index, format_func=lambda i: track_label(hits.loc[i])
                )
                seed_id = hits.loc[pick, "track_id"]
            elif q:
                st.info("No matches.")
        ready = start != "(start from a song)" or seed_id is not None
        if st.button("🎛️ Build set", type="primary", disabled=not ready):
            request = SetRequest(
                None if start not in MOODS else start,
                end,
                n_tracks,
                arc,
                seed_track_id=seed_id,
                genres=genres,
                min_popularity=min_pop,
                strict=strict,
            )

    if request is not None:
        try:
            with st.spinner("Sequencing tracks with beam search…"):
                dj_set = builder.build(request)
            with st.spinner(f"Writing mixing tips ({coach.source})…"):
                notes = coach.explain(dj_set)
            st.session_state.dj_result = (dj_set, notes)
        except ValueError as exc:
            st.session_state.pop("dj_result", None)
            st.warning(f"{exc}")

    if result := st.session_state.get("dj_result"):
        dj_set, notes = result
        s = dj_set.summary
        t = dj_set.tracks
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Length", f"{len(t)} tracks", f"{s['minutes']:.0f} min", delta_color="off")
        m2.metric("Smooth transitions", f"{s['smooth_share']:.0%}")
        m3.metric("Key clashes", f"{s['key_clash_share']:.0%}")
        m4.metric("BPM range", f"{t.bpm.min()}–{t.bpm.max()}")

        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Mood journey** (blue = your set, dashed = target)")
            st.altair_chart(mood_plane_chart(dj_set), width="stretch")
        with right:
            st.markdown("**Tempo**")
            st.altair_chart(timeline_chart(dj_set, "bpm", "BPM"), width="stretch")
            st.markdown("**Energy** (dashed = target)")
            st.altair_chart(
                timeline_chart(dj_set, "energy", "Energy", "target_energy"), width="stretch"
            )

        st.subheader("Tracklist")
        show = t.assign(link="https://open.spotify.com/track/" + t["track_id"])
        st.dataframe(
            show[["position", "track_name", "artists", "bpm", "camelot", "energy", "mood", "link"]],
            hide_index=True,
            width="stretch",
            column_config={
                "position": st.column_config.NumberColumn("#", width="small"),
                "track_name": "Track",
                "artists": "Artists",
                "bpm": st.column_config.NumberColumn("BPM", width="small"),
                "camelot": st.column_config.TextColumn("Key", help="Camelot wheel notation"),
                "energy": st.column_config.ProgressColumn(
                    "Energy", min_value=0, max_value=1, format="%.2f"
                ),
                "mood": "Mood",
                "link": st.column_config.LinkColumn("Spotify", display_text="Open ↗"),
            },
        )
        pick = st.selectbox(
            "Preview a track",
            t.index,
            format_func=lambda i: f"{t.position[i]}. {t.track_name[i]} — {t.artists[i]}",
        )
        st.iframe(f"https://open.spotify.com/embed/track/{t.track_id[pick]}", height=152)

        st.subheader("🎓 How to mix it")
        st.caption(
            f"{notes.source}. Timings are estimated from track length and tempo "
            "(the dataset has no beat grid), so confirm by ear."
        )
        st.markdown(notes.overview)
        for w in notes.warnings:
            st.caption(f"ℹ️ {w}")
        for tr in dj_set.transitions:
            badge = "🤖 AI" if notes.sources.get(tr.position) == "AI" else "📘 built-in"
            title = (
                f"{RATING_ICON[tr.rating]} {tr.position} → {tr.position + 1} · "
                f"{tr.rating} ({tr.score:.0f}/100) · {tr.from_track} → {tr.to_track}"
            )
            with st.expander(title, expanded=tr.position == 1):
                st.caption(badge)
                st.markdown(notes.tips[tr.position])

        with st.expander("What do these terms mean?"):
            st.markdown(
                "- **BPM**: beats per minute (the track's tempo). Tracks within about 6% of each "
                "other can be beatmatched by nudging the tempo slider.\n"
                "- **Camelot key (e.g. 8A)**: a DJ-friendly way to write musical keys. Same "
                "number, ±1, or switching A/B on the same number mix without clashing.\n"
                "- **Beatmatching**: lining up the beats of two tracks so they play in time.\n"
                "- **Bass swap**: during a blend, turn down the low EQ of one track as you turn "
                "up the other's, so two basslines never play at full volume together.\n"
                "- **Echo out**: an effect that repeats the last beats of a track while you cut "
                "to the next; useful when tempos are too different to blend."
            )

# --------------------------------------------------------- similar songs tab
with tab_rec:
    with st.expander("⚙️ Settings"):
        c1, c2, c3 = st.columns(3)
        k = c1.slider("Number of recommendations", 5, 30, 10)
        diversity = c1.slider(
            "Diversity (MMR λ)",
            0.0,
            0.9,
            0.3,
            0.05,
            help="0 = most similar only. Higher trades similarity for variety.",
        )
        genre_weight = c2.slider(
            "Genre weight", 0.0, 4.0, 1.5, 0.25, help="0 = match purely on how tracks sound."
        )
        min_pop = c2.slider("Minimum popularity ", 0, 100, 0)
        exclude_artists = c3.checkbox("Only new artists", value=False)
        allow_explicit = c3.checkbox("Allow explicit tracks", value=True)
        genre_filter = c3.multiselect("Restrict to genres", rec.features.genres)
    model = load_model(genre_weight)

    if "seeds" not in st.session_state:
        st.session_state.seeds = []
    col_search, col_seeds = st.columns([3, 2])
    with col_search:
        query = st.text_input("Search for a song or artist", placeholder="e.g. Blinding Lights")
        matches = model.search(query, limit=25) if query else model.tracks_.head(0)
        if query and matches.empty:
            st.info("No matches. Try fewer words or a different spelling.")
        if not matches.empty:
            choice_idx = st.selectbox(
                "Matches", matches.index, format_func=lambda i: track_label(matches.loc[i])
            )
            if st.button("➕ Add as seed", type="primary"):
                tid = matches.loc[choice_idx, "track_id"]
                if tid not in st.session_state.seeds:
                    st.session_state.seeds.append(tid)
    with col_seeds:
        st.subheader("Seed tracks")
        if not st.session_state.seeds:
            st.write("Add one or more songs you like. Several seeds act like a playlist.")
        for tid in list(st.session_state.seeds):
            row = model.tracks_.loc[model.index_[tid]]
            c1, c2 = st.columns([5, 1])
            c1.write(f"**{row.track_name}** — {row.artists}")
            if c2.button("✕", key=f"rm-{tid}"):
                st.session_state.seeds.remove(tid)
                st.rerun()

    if st.session_state.seeds:
        seeds = st.session_state.seeds
        recs = model.recommend(
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
            recs["why"] = [model.explain_text(seeds, tid) for tid in recs["track_id"]]
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
            seed_rows = model.tracks_.loc[model.to_indices(seeds)]
            profile = (
                pd.concat(
                    [
                        seed_rows[PROFILE_FEATURES].mean().rename("Your seeds"),
                        recs[PROFILE_FEATURES].mean().rename("Recommendations"),
                    ],
                    axis=1,
                )
                .reset_index(names="feature")
                .melt("feature", var_name="group", value_name="value")
            )
            st.markdown("**Audio profile: seeds vs. recommendations**")
            st.altair_chart(
                alt.Chart(profile)
                .mark_bar(cornerRadiusEnd=3, size=12)
                .encode(
                    y=alt.Y("feature:N", title=None, sort=PROFILE_FEATURES),
                    yOffset=alt.YOffset("group:N", sort=["Your seeds", "Recommendations"]),
                    x=alt.X("value:Q", title="Mean value (0–1)", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color(
                        "group:N",
                        title=None,
                        sort=["Your seeds", "Recommendations"],
                        scale=alt.Scale(range=[BLUE, ORANGE]),
                        legend=alt.Legend(orient="top"),
                    ),
                    tooltip=["group", "feature", alt.Tooltip("value:Q", format=".2f")],
                )
                .properties(height=300),
                width="stretch",
            )

# -------------------------------------------------------------- eval tab
with tab_eval:
    st.markdown(
        "**DJ-set sequencing.** For 200 random set requests, each method's track order is scored "
        "by the same mixing rules. Beam search keeps nearly every transition beatmatchable and in "
        "a compatible key, while a 'similar songs' playlist clashes most of the time."
    )
    dj_csv = REPORTS_DIR / "dj_comparison.csv"
    if dj_csv.exists():
        dj_table = pd.read_csv(dj_csv, index_col=0)
        st.dataframe(
            dj_table[
                [
                    "smooth_share",
                    "key_clash_share",
                    "mean_tempo_change_pct",
                    "mood_error",
                    "genre_coherence",
                ]
            ]
            .rename(
                columns={
                    "smooth_share": "Smooth",
                    "key_clash_share": "Key clashes",
                    "mean_tempo_change_pct": "Avg tempo change %",
                    "mood_error": "Mood error",
                    "genre_coherence": "Genre coherence",
                }
            )
            .style.format(precision=3),
            width="stretch",
        )
    st.markdown(
        "**Similar-song recommender: artist continuation.** For 1,000 random artists the model "
        "sees 3 of their songs and must find the artist's *other* songs in its top 10. It never "
        "sees who the artist is. Hyper-parameters were tuned on a separate validation draw."
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
    for fig in ("dj_comparison", "model_comparison", "genre_weight_sweep", "diversity_sweep"):
        path = Path(REPORTS_DIR / "figures" / f"{fig}.png")
        if path.exists():
            st.image(str(path))
