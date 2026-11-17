# 🎧 AI DJ & Music Recommender

[![CI](https://github.com/meetkothari-08/Music-Recommendation/actions/workflows/ci.yml/badge.svg)](https://github.com/meetkothari-08/Music-Recommendation/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

Describe a vibe, like *"chill house set for a sunset that builds into a party"*, and get a
**DJ set that follows that mood journey**, ordered so every transition can be beatmatched and
mixed in a compatible key. An **AI coach** then explains, step by step, how a beginner should mix
each transition.

Built on **80,695 Spotify tracks across 114 genres**, with a data-cleaning pipeline, a
content-based recommender, beam-search sequencing and offline evaluation against baselines.

![App screenshot](docs/app_screenshot.png)

## Results at a glance

**DJ sets.** 200 random set requests (random start and end mood, energy shape, half restricted to
one genre), 10 tracks each. Every method's track order is scored by the same mixing rules:

| Method | Smooth transitions | Key clashes | Avg tempo change | Genre coherence |
|---|---:|---:|---:|---:|
| Random tracks matching each mood | 2% | 66% | 16.9% | 44% |
| "Similar songs" playlist | 7% | 63% | 12.7% | 100% |
| DJ objective, greedy (one step at a time) | 63% | 0% | 1.9% | 85% |
| **DJ objective, beam search (final)** | **81%** | **0%** | **1.5%** | **98%** |

- A typical "similar songs" playlist clashes harmonically on **63%** of transitions. The DJ
  sequencer has **none**, and keeps tempo changes small enough to beatmatch on every transition.
- **Beam search beats greedy** (81% vs 63% smooth, ±2.5%): picking the best next song at each step
  often paints the set into a corner later.
- **The trade-off:** the sequencer stays about 0.1 from the target mood on average, versus 0.02
  for a playlist that only matches mood. It gives up a little mood precision for mixability.

**Similar-song recommender.** The model gets 3 songs by an artist it has never been told about and
must find that artist's other songs among all 80k tracks. Tested on 1,000 held-out artists, it
finds at least one for **53%** of them (HitRate@10 0.527), **7.3x** the best simple baseline.

Full report with confidence intervals, ablations and sweeps: [`reports/evaluation.md`](reports/evaluation.md).

## How the DJ set builder works

```
"chill house set that builds into a party"
        │  AI (or built-in parser) → validated settings: chill → energetic, 10 tracks, genre: house
        ▼
target mood curve on the valence × energy plane
        │  beam search over 80k tracks: mood fit + tempo + key + energy flow + style
        ▼
ordered set ──► per-transition plan (technique, timing, tempo change) ──► AI coach explains it
```

### 1. Mood ([`dj.py`](src/music_recommender/dj.py))
Psychologists describe mood on two axes: **valence** (sad ↔ happy) and **energy** (calm ↔
intense). Spotify measures both for every track, so moods like *sad*, *chill*, *happy* and
*energetic* are regions of that plane. A set is a **path** across it: straight from start to end
mood, or a classic club "peak" shape (warm up, peak about 70% of the way in, cool down).

### 2. The DJ rules
| Rule | Why DJs care | How it's scored |
|---|---|---|
| **Tempo matching** | Beats can only be lined up if tempos are close | Change ≤ 6% to match; half/double-time pairs (70 ↔ 140 BPM) count too |
| **Harmonic mixing** | Overlapping songs in clashing keys sound wrong | **Camelot wheel**: same key, ±1, relative major/minor, +2 "energy boost" |
| **Energy flow** | Sudden jumps kill the dance floor | Energy change should follow the target curve |
| **Style** | DJs stay within a genre family | Shared genres and audio similarity with the previous and opening tracks |

### 3. Sequencing = path finding
Choosing the order is a search problem over 80k tracks. **Beam search** keeps the 20 best partial
sets at every step and extends each with candidates that fit the next target mood *and* make a
smooth transition. Repeated artists and other versions of the same song are ruled out, and so are
non-DJ genres (kids' music, comedy, sleep sounds) unless you ask for them. A set builds in
about 0.1 seconds.

### 4. Transition plans
For each transition the code computes a technique (**long blend**, **short blend** for key
clashes, or **echo out** when tempos are too far apart), how many beats to overlap, when to start
the next track, and the exact tempo adjustment, e.g. *"Speed up the incoming track by 0.7%"*.

### 5. The AI coach ([`coach.py`](src/music_recommender/coach.py))
The key design decision is **what the AI is allowed to do**:
- **AI:** turns plain-English requests into settings, and explains each transition in friendly,
  beginner language (what beatmatching or a bass swap is, and why this pair works).
- **Code:** chooses every song and computes every number. Language models are unreliable with
  numbers and will invent a song's BPM or key.
- **Guardrails:** the AI's settings are clamped to allowed moods, arcs and catalogue genres. Every
  number in an AI tip is checked against the computed facts; a tip with an invented number is
  rejected and replaced by the built-in explanation.
- **Works for free and offline:** it uses free API tiers ([Google Gemini](https://aistudio.google.com/apikey)
  or [Groq](https://console.groq.com/keys)), picks a currently available model automatically, and
  falls back to a rule-based coach when there's no key or the API fails.

## How the similar-song recommender works

| Stage | What it does |
|---|---|
| **Cleaning** ([`data.py`](src/music_recommender/data.py)) | 114k rows → 80.7k songs: merges tracks listed under several genres (24,245 rows) and re-releases of the same recording such as remasters (8,737), drops failed audio analyses |
| **Features** ([`features.py`](src/music_recommender/features.py)) | Quantile-transformed audio features (several are heavily skewed) + one-hot genres, with a tunable genre weight |
| **Model** ([`recommender.py`](src/music_recommender/recommender.py)) | Cosine kNN over a taste profile of one or more seeds, MMR re-ranking for variety, filters, and plain-English explanations |
| **Evaluation** ([`evaluation.py`](src/music_recommender/evaluation.py)) | Artist-continuation task; tuned on a validation draw of artists, reported on a separate test draw |

| Model | HitRate@10 | NDCG@10 | Diversity |
|---|---:|---:|---:|
| Random | 0.001 | 0.000 | 0.948 |
| Most popular | 0.001 | 0.000 | 0.801 |
| Most popular in the seeds' genres | 0.072 | 0.013 | 0.671 |
| Audio features only (kNN) | 0.123 | 0.025 | 0.081 |
| Audio + learned genre embedding | 0.470 | 0.119 | 0.185 |
| Audio + one-hot genre | 0.500 | 0.138 | 0.199 |
| **Audio + one-hot genre + MMR (final)** | **0.527** | **0.140** | **0.239** |

Findings:
1. **Genre is the strongest single signal**, lifting HitRate from 0.12 (audio only) to 0.50.
2. **A learned genre embedding (PPMI + SVD) lost to plain one-hot genres** (0.470 vs 0.500). The
   [EDA](notebooks/01_eda.ipynb) shows some genre co-occurrences are collection artefacts
   (`dub`/`dubstep`, `reggae`/`reggaeton`), so the embedding partly learns noise.
3. **Some diversity is free:** MMR re-ranking makes lists 20% more diverse while hit rate goes up
   (0.500 → 0.527).
4. Recommendations average **popularity 35, versus 97** for the popularity baseline, so the model
   surfaces long-tail tracks rather than only hits.

## Quickstart

```bash
git clone https://github.com/meetkothari-08/Music-Recommendation.git
```

```bash
cd Music-Recommendation
```

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\activate` on Windows, or `source .venv/bin/activate` on macOS/Linux.

```bash
python -m pip install -e ".[dev]"
```

```bash
python -m music_recommender build
```

```bash
python -m streamlit run app/streamlit_app.py
```

`build` downloads the [public dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)
(20 MB) from Hugging Face and cleans it. The app also does this on first launch if needed.

**Turn on the AI coach (optional, free):** get a key from
[Google AI Studio](https://aistudio.google.com/apikey) or [Groq](https://console.groq.com/keys) and
paste it in the app's sidebar. To save it instead, set `GEMINI_API_KEY` or `GROQ_API_KEY` as an
environment variable, or put it in `.streamlit/secrets.toml` (git-ignored). Set
`MUSIC_REC_LLM_MODEL` to pin a specific model.

Command-line versions:

```bash
python -m music_recommender djset "chill house set that builds into a party, 10 songs"
```

```bash
python -m music_recommender recommend "smells like teen spirit" -k 10 --new-artists
```

```bash
python -m music_recommender evaluate
```

```bash
python -m pytest
```

`evaluate` regenerates everything in `reports/` in about 2 minutes. The `python -m` form works
even on Windows machines where security policies block `.exe` launchers.

### Deploy for free
The repo is ready for [Streamlit Community Cloud](https://streamlit.io/cloud): create an app from
this repository with `app/streamlit_app.py` as the main file. Leave the AI key out of the public
app so visitors can't use up your quota; they can paste their own key, or use the built-in coach.

## Project structure

```
├── app/streamlit_app.py           # DJ set builder, similar songs, evaluation tabs
├── notebooks/01_eda.ipynb         # exploratory analysis → modelling decisions
├── reports/                       # generated evaluation report, CSVs and figures
├── src/music_recommender/
│   ├── data.py                    # download + cleaning pipeline (with audit trail)
│   ├── features.py                # quantile-scaled audio + genre features
│   ├── recommender.py             # kNN + MMR re-ranking + explanations + filters
│   ├── dj.py                      # moods, Camelot keys, tempo matching, beam search, transition plans
│   ├── coach.py                   # AI coach: request parsing, grounded explanations, fallbacks
│   ├── evaluation.py              # recommender evaluation: artist continuation, baselines, sweeps
│   ├── dj_evaluation.py           # DJ-set evaluation against simpler playlists
│   ├── reporting.py               # writes reports/evaluation.md and figures
│   └── cli.py                     # command-line interface
├── tests/                         # 69 tests on synthetic data; the AI is mocked, no key needed
└── .github/workflows/ci.yml       # lint + tests on Python 3.10 and 3.12
```

## Limitations and next steps

- **It plans mixes; it doesn't play them.** The dataset has audio features but no audio, so
  timings are estimated from track length and tempo rather than detected beats and phrases. The
  natural next step is to accept audio files the user owns, detect beats and keys from the audio,
  and render the actual crossfaded mix.
- **The mixing rules are standard DJ practice, not learned.** With a dataset of real DJ mixes
  (tracklists with transition timings), the transition score could be learned from what
  professional DJs actually do.
- **No listening data.** The recommender can't learn that people who like X also like Y when X and
  Y sound different. A hybrid with collaborative filtering on playlist data would fix that.
- **Static catalogue.** Spotify closed its audio-features API to new apps in late 2024, so new
  tracks would need audio features computed from the audio itself.

## Tech stack
Python · pandas · NumPy · scikit-learn · Streamlit · Altair · Matplotlib · LLM APIs (Gemini / Groq, OpenAI-compatible) · pytest · Ruff · GitHub Actions
