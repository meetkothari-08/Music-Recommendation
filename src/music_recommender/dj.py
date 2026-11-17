"""Mood-journey DJ sets: sequence tracks the way a DJ would.

A DJ moves from one song to the next following three rules:

* **Tempo matching** - the next track's BPM is close enough (about +/-6%)
  that its beats can be lined up by nudging the tempo fader. Half-time and
  double-time pairings (e.g. 70 and 140 BPM) also line up.
* **Harmonic mixing** - the two tracks are in compatible keys, so the overlap
  doesn't clash. DJs use the *Camelot wheel*: 12 numbers x 2 letters
  (A = minor, B = major). Same key, +/-1 on the wheel, or switching letter
  on the same number all mix cleanly; +2 is a deliberate "energy boost".
* **Energy flow** - energy rises and falls gradually.

On top of that, the set follows a **mood journey** on the valence (sad ->
happy) x energy (calm -> intense) plane, e.g. from "sad" to "happy".

Choosing the order is a path-finding problem. :class:`DJSetBuilder` solves
it with **beam search**: it keeps the ``beam_width`` best partial sets at
every step and extends each with the candidates that best fit the target
mood *and* make a smooth transition, so one good-looking early pick can't
force a run of bad transitions later.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from music_recommender.recommender import ContentRecommender

# --------------------------------------------------------------------- moods
# (valence, energy) centres on the circumplex mood plane.
MOODS: dict[str, tuple[float, float]] = {
    "sad": (0.20, 0.30),
    "calm": (0.40, 0.20),
    "chill": (0.60, 0.40),
    "happy": (0.80, 0.65),
    "energetic": (0.70, 0.88),
    "intense": (0.25, 0.90),
}
MOOD_DESCRIPTIONS: dict[str, str] = {
    "sad": "low valence, low energy - melancholic, reflective",
    "calm": "soft and peaceful",
    "chill": "relaxed but positive",
    "happy": "upbeat and positive",
    "energetic": "high energy, party / workout",
    "intense": "dark, aggressive, high energy",
}
ARCS: dict[str, str] = {
    "straight": "move steadily from the start mood to the end mood",
    "peak": "classic club set: warm up, peak around 70% of the way, then cool down",
}

# Genres that rarely belong in a DJ set; excluded unless the user asks for them.
NON_DJ_GENRES = frozenset(
    {"children", "kids", "comedy", "sleep", "study", "disney", "show-tunes", "opera", "classical"}
)

MIXABLE_TEMPO = 0.06  # max tempo change a beginner can make sound natural
TEMPO_SCORE_ZERO = 0.08  # tempo score falls to 0 at this change


def mood_point(mood: str | Sequence[float]) -> np.ndarray:
    if isinstance(mood, str):
        if mood not in MOODS:
            raise ValueError(f"Unknown mood '{mood}'. Choose from {sorted(MOODS)}")
        return np.array(MOODS[mood], dtype=float)
    point = np.asarray(mood, dtype=float)
    if point.shape != (2,):
        raise ValueError("A mood point must be (valence, energy)")
    return np.clip(point, 0.0, 1.0)


def nearest_mood(valence: float, energy: float) -> str:
    return min(MOODS, key=lambda m: (MOODS[m][0] - valence) ** 2 + (MOODS[m][1] - energy) ** 2)


def target_curve(
    start: str | Sequence[float], end: str | Sequence[float], n: int, arc: str = "straight"
) -> np.ndarray:
    """Target (valence, energy) for each position in the set, shape (n, 2)."""
    if arc not in ARCS:
        raise ValueError(f"Unknown arc '{arc}'. Choose from {sorted(ARCS)}")
    a, b = mood_point(start), mood_point(end)
    t = np.linspace(0.0, 1.0, n)[:, None]
    curve = a + (b - a) * t
    if arc == "peak":
        # bump peaks at t = 0.7: sin(pi * t**p) with 0.7**p = 0.5
        bump = 0.3 * np.sin(np.pi * t[:, 0] ** (np.log(0.5) / np.log(0.7)))
        curve[:, 1] += bump
    return np.clip(curve, 0.05, 0.98)


# ----------------------------------------------------------- harmonic keys
def camelot_number(key, mode):
    """Camelot wheel number (1-12) for a pitch class (C=0) and mode (1=major)."""
    key = np.asarray(key)
    mode = np.asarray(mode)
    return (7 * key + np.where(mode == 1, 7, 4)) % 12 + 1


def camelot(key: int, mode: int) -> str:
    if key < 0:
        return "?"
    return f"{int(camelot_number(key, mode))}{'B' if mode == 1 else 'A'}"


_KEY_LABELS = {
    "same": "same key",
    "adjacent": "adjacent key (±1)",
    "boost": "energy boost (+2)",
    "relative": "relative major/minor",
    "diagonal": "diagonal mood shift",
    "clash": "key clash",
}
_KEY_SCORES = {"same": 1.0, "adjacent": 0.9, "relative": 0.85, "boost": 0.6, "diagonal": 0.4}


def key_compatibility(num_a, minor_a, num_b, minor_b) -> np.ndarray:
    """Vectorised harmonic-mixing score in [0, 1] (0 = clash)."""
    d = (np.asarray(num_b) - np.asarray(num_a)) % 12
    step = np.minimum(d, 12 - d)
    same = np.asarray(minor_a) == np.asarray(minor_b)
    score = np.zeros(np.broadcast(d, same).shape)
    score = np.where(~same & (step == 1), _KEY_SCORES["diagonal"], score)
    score = np.where(same & (d == 2), _KEY_SCORES["boost"], score)
    score = np.where(~same & (step == 0), _KEY_SCORES["relative"], score)
    score = np.where(same & (step == 1), _KEY_SCORES["adjacent"], score)
    score = np.where(same & (step == 0), _KEY_SCORES["same"], score)
    return score


def key_relation(num_a: int, minor_a: bool, num_b: int, minor_b: bool) -> str:
    d = (num_b - num_a) % 12
    step = min(d, 12 - d)
    if minor_a == minor_b:
        name = {0: "same", 1: "adjacent"}.get(step, "boost" if d == 2 else "clash")
    else:
        name = {0: "relative", 1: "diagonal"}.get(step, "clash")
    return _KEY_LABELS[name]


# ------------------------------------------------------------------- tempo
_TEMPO_FACTORS = np.array([1.0, 2.0, 0.5])


def tempo_match(bpm_out, bpm_in) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """How the incoming track lines up with the outgoing one.

    Returns ``(change, factor, score)``: ``change`` is the fraction the
    incoming track's tempo must change by (e.g. +0.03 = speed up 3%),
    ``factor`` is 1 (normal), 2 (incoming plays half-time) or 0.5
    (incoming plays double-time), and ``score`` is in [0, 1].
    """
    bpm_out = np.asarray(bpm_out, dtype=float)
    bpm_in = np.asarray(bpm_in, dtype=float)
    effective = bpm_in[..., None] * _TEMPO_FACTORS
    dev = np.abs(np.log(effective / bpm_out[..., None]))
    best = np.argmin(dev, axis=-1)
    factor = _TEMPO_FACTORS[best]
    change = bpm_out / (bpm_in * factor) - 1.0
    score = np.clip(1.0 - np.abs(change) / TEMPO_SCORE_ZERO, 0.0, 1.0)
    score = np.where(factor == 1.0, score, 0.9 * score)
    return change, factor, score


# ------------------------------------------------------------- data classes
@dataclass
class SetRequest:
    start_mood: str | tuple[float, float] | None = "chill"
    end_mood: str | tuple[float, float] = "happy"
    n_tracks: int = 12
    arc: str = "straight"
    seed_track_id: str | None = None
    genres: Sequence[str] = ()
    allow_explicit: bool = True
    min_popularity: int = 0
    strict: bool = True

    def validate(self) -> None:
        if not 2 <= self.n_tracks <= 40:
            raise ValueError("n_tracks must be between 2 and 40")
        if self.start_mood is None and self.seed_track_id is None:
            raise ValueError("Give a start mood or a seed track")
        if self.arc not in ARCS:
            raise ValueError(f"Unknown arc '{self.arc}'")
        mood_point(self.end_mood)
        if self.start_mood is not None:
            mood_point(self.start_mood)


@dataclass
class TransitionPlan:
    """Grounded, computed instructions for one transition."""

    technique: str
    mix_beats: int
    mix_seconds: float
    start_at_seconds: float
    tempo_instruction: str
    steps: list[str]
    warnings: list[str] = field(default_factory=list)

    @property
    def start_at(self) -> str:
        m, s = divmod(int(round(self.start_at_seconds)), 60)
        return f"{m}:{s:02d}"


@dataclass
class Transition:
    position: int  # transition from track `position` to `position + 1` (1-based)
    from_track: str
    to_track: str
    bpm_from: float
    bpm_to: float
    tempo_change_pct: float
    tempo_relation: str
    key_from: str
    key_to: str
    key_relation: str
    energy_from: float
    energy_to: float
    score: float  # 0-100
    rating: str  # smooth / ok / tricky
    plan: TransitionPlan

    def facts(self) -> dict:
        """Every number the coach is allowed to talk about."""
        return {
            "position": self.position,
            "from": self.from_track,
            "to": self.to_track,
            "bpm_from": round(self.bpm_from),
            "bpm_to": round(self.bpm_to),
            "tempo_change_pct": round(self.tempo_change_pct, 1),
            "tempo_relation": self.tempo_relation,
            "key_from": self.key_from,
            "key_to": self.key_to,
            "key_relation": self.key_relation,
            "energy_from": round(self.energy_from, 2),
            "energy_to": round(self.energy_to, 2),
            "rating": self.rating,
            "technique": self.plan.technique,
            "mix_beats": self.plan.mix_beats,
            "mix_seconds": round(self.plan.mix_seconds),
            "start_incoming_at": self.plan.start_at,
            "tempo_instruction": self.plan.tempo_instruction,
            "warnings": self.plan.warnings,
        }


@dataclass
class DJSet:
    tracks: pd.DataFrame
    transitions: list[Transition]
    targets: np.ndarray
    request: SetRequest

    @property
    def summary(self) -> dict[str, float]:
        return set_metrics(self.tracks, self.transitions, self.targets)


def set_metrics(
    tracks: pd.DataFrame, transitions: list[Transition], targets: np.ndarray
) -> dict[str, float]:
    mood_err = np.hypot(
        tracks["valence"].to_numpy() - targets[:, 0], tracks["energy"].to_numpy() - targets[:, 1]
    )
    tempo = np.array([abs(t.tempo_change_pct) for t in transitions])
    energy_jump = np.array([abs(t.energy_to - t.energy_from) for t in transitions])
    return {
        "tracks": len(tracks),
        "minutes": tracks["duration_ms"].sum() / 60_000,
        "smooth_share": float(np.mean([t.rating == "smooth" for t in transitions])),
        "mixable_share": float(np.mean([t.rating != "tricky" for t in transitions])),
        "key_clash_share": float(np.mean([t.key_relation == "key clash" for t in transitions])),
        "mean_tempo_change_pct": float(tempo.mean()),
        "mean_energy_jump": float(energy_jump.mean()),
        "mood_error": float(mood_err.mean()),
        "mean_transition_score": float(np.mean([t.score for t in transitions])),
    }


# ----------------------------------------------------------- transition plan
def _fmt_pct(x: float) -> str:
    return f"{abs(x):.1f}%"


def plan_transition(
    out: pd.Series, inc: pd.Series, change: float, factor: float, key_score: float, key_rel: str
) -> TransitionPlan:
    """Rule-based mixing technique for a transition (no audio needed)."""
    bpm_out = float(out["tempo"])
    pct = abs(change) * 100
    tempo_ok = abs(change) <= MIXABLE_TEMPO
    key_ok = key_score >= 0.6
    energy_delta = float(inc["energy"] - out["energy"])
    warnings: list[str] = []

    if tempo_ok and key_ok:
        dance = min(out["danceability"], inc["danceability"])
        mix_beats = 32 if dance >= 0.6 and min(out["energy"], inc["energy"]) >= 0.5 else 16
        technique = "long blend"
    elif tempo_ok:
        mix_beats = 8
        technique = "short blend"
        warnings.append("Keys clash - keep the overlap short so the melodies don't fight.")
    else:
        mix_beats = 4
        technique = "echo out"
        warnings.append("Tempos are too far apart to beatmatch - transition with an echo instead.")

    mix_seconds = mix_beats * 60.0 / bpm_out
    outro_buffer = 8 * 60.0 / bpm_out  # leave the last bars of the outro playing
    start_at = max(0.0, float(out["duration_ms"]) / 1000 - mix_seconds - outro_buffer)

    if not tempo_ok:
        tempo_instruction = "Don't try to match tempos; let the outgoing track echo out."
    elif pct < 0.5:
        tempo_instruction = "Tempos already match - no tempo change needed."
    else:
        direction = "Speed up" if change > 0 else "Slow down"
        tempo_instruction = f"{direction} the incoming track by {_fmt_pct(change * 100)}."
    if factor != 1.0 and tempo_ok:
        kind = "half-time" if factor == 2.0 else "double-time"
        tempo_instruction += f" It plays {kind} against the outgoing track, so beats still line up."

    steps = [f"Cue the incoming track on its first beat. {tempo_instruction}"]
    if technique == "long blend":
        steps += [
            f"Start it when the outgoing track is at about {_fmt_time(start_at)}.",
            f"Raise the incoming volume over the first {mix_beats // 2} beats.",
            "Halfway through, swap the bass: turn the outgoing low EQ down as you turn the incoming one up.",
            f"Fade the outgoing track out over the last {mix_beats // 2} beats.",
        ]
    elif technique == "short blend":
        steps += [
            f"Start it when the outgoing track is at about {_fmt_time(start_at)}.",
            f"Overlap for only {mix_beats} beats, swapping the bass straight away.",
        ]
    else:
        steps += [
            "Turn on an echo effect on the outgoing track at the end of a phrase.",
            "Pull the outgoing volume down and start the incoming track on the next beat.",
        ]
    if energy_delta >= 0.25:
        steps.append("Big energy lift: start the incoming track right on a new phrase for impact.")
    elif energy_delta <= -0.25:
        steps.append(
            "Energy drops a lot: use a low-pass filter on the outgoing track to soften it."
        )
    if key_rel == "energy boost (+2)":
        steps.append("The key steps up two places on the wheel, which gives a deliberate lift.")
    return TransitionPlan(
        technique, mix_beats, mix_seconds, start_at, tempo_instruction, steps, warnings
    )


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


# ---------------------------------------------------------------- builder
W_MOOD, W_TEMPO, W_KEY, W_STYLE, W_ENERGY, W_POP = 1.2, 1.0, 0.8, 0.6, 0.4, 0.3
W_ANCHOR = 0.6  # similarity to the opening track: keeps the whole set in one style
W_GENRE_PREV, W_GENRE_OPEN = 1.0, 0.8  # sharing a genre with the previous / opening track
STRICT_PENALTY = 2.0


class DJSetBuilder:
    """Builds mood-journey DJ sets with beam search over a fitted recommender."""

    def __init__(
        self, recommender: ContentRecommender, beam_width: int = 20, pool_size: int = 2000
    ) -> None:
        self.rec = recommender
        self.beam_width = beam_width
        self.pool_size = pool_size
        t = recommender.tracks_
        self.tracks = t
        self.valence = t["valence"].to_numpy(float)
        self.energy = t["energy"].to_numpy(float)
        self.tempo = t["tempo"].to_numpy(float)
        self.key_num = camelot_number(t["key"].to_numpy(), t["mode"].to_numpy())
        self.minor = t["mode"].to_numpy() == 0
        self.popularity = t["popularity"].to_numpy(float) / 100
        self.song_group = pd.factorize(t["song_group"])[0]
        self.artist = pd.factorize(t["primary_artist"])[0]
        self.genre_hot = (
            MultiLabelBinarizer(sparse_output=True)
            .fit_transform(t["genres"])
            .astype(np.float32)
            .toarray()
        )
        # DJ-friendly: 4/4 time, not spoken word
        self.base_mask = (t["time_signature"].to_numpy() == 4) & (
            t["speechiness"].to_numpy() < 0.66
        )

    # ------------------------------------------------------------- scoring
    def _transition_scores(self, last: int, cand: np.ndarray, target_de: float):
        change, _, tempo_s = tempo_match(self.tempo[last], self.tempo[cand])
        key_s = key_compatibility(
            self.key_num[last], self.minor[last], self.key_num[cand], self.minor[cand]
        )
        style = self.rec.vectors_[cand] @ self.rec.vectors_[last]
        jump = np.abs(self.energy[cand] - self.energy[last] - target_de)
        energy_s = np.clip(1.0 - jump / 0.25, 0.0, 1.0)
        total = W_TEMPO * tempo_s + W_KEY * key_s + W_STYLE * style + W_ENERGY * energy_s
        mixable = (np.abs(change) <= MIXABLE_TEMPO) & (key_s > 0)
        return total, mixable

    def _mood_fit(self, idx: np.ndarray, target: np.ndarray) -> np.ndarray:
        dist = np.hypot(self.valence[idx] - target[0], self.energy[idx] - target[1])
        return np.clip(1.0 - dist / 0.3, 0.0, 1.0)

    def _mask(self, req: SetRequest) -> np.ndarray:
        t = self.tracks
        mask = self.base_mask & (t["popularity"].to_numpy() >= req.min_popularity)
        if not req.allow_explicit:
            mask &= ~t["explicit"].to_numpy()
        if req.genres:
            wanted = set(req.genres)
            mask &= t["genres"].map(lambda gs: bool(wanted.intersection(gs))).to_numpy()
        else:
            mask &= ~t["genres"].map(lambda gs: bool(NON_DJ_GENRES.intersection(gs))).to_numpy()
        return mask

    # --------------------------------------------------------------- build
    def build(self, req: SetRequest, beam_width: int | None = None) -> DJSet:
        req.validate()
        beam_width = beam_width or self.beam_width
        mask = self._mask(req)
        seed = None
        if req.seed_track_id is not None:
            (seed,) = self.rec.to_indices([req.seed_track_id])
        start = req.start_mood
        if start is None:
            start = (self.valence[seed], self.energy[seed])
        targets = target_curve(start, req.end_mood, req.n_tracks, req.arc)

        eligible = np.flatnonzero(mask)
        if seed is not None:
            eligible = eligible[eligible != seed]
        if len(eligible) < req.n_tracks:
            raise ValueError("Not enough tracks match these filters to build a set")

        def pool(target: np.ndarray) -> np.ndarray:
            dist = np.hypot(self.valence[eligible] - target[0], self.energy[eligible] - target[1])
            m = min(self.pool_size, len(eligible))
            return eligible[np.argpartition(dist, m - 1)[:m]]

        # beams: list of (cumulative score, path)
        if seed is not None:
            beams = [(0.0, [int(seed)])]
        else:
            first = pool(targets[0])
            s = W_MOOD * self._mood_fit(first, targets[0]) + W_POP * self.popularity[first]
            top = np.argsort(-s)[:beam_width]
            beams = [(float(s[i]), [int(first[i])]) for i in top]

        for step in range(1, req.n_tracks):
            cand = pool(targets[step])
            mood = W_MOOD * self._mood_fit(cand, targets[step]) + W_POP * self.popularity[cand]
            target_de = targets[step, 1] - targets[step - 1, 1]
            scored: list[tuple[float, list[int]]] = []
            for score, path in beams:
                trans, mixable = self._transition_scores(path[-1], cand, target_de)
                anchor = W_ANCHOR * (self.rec.vectors_[cand] @ self.rec.vectors_[path[0]])
                genre = W_GENRE_PREV * (self.genre_hot[cand] @ self.genre_hot[path[-1]] > 0)
                genre = genre + W_GENRE_OPEN * (self.genre_hot[cand] @ self.genre_hot[path[0]] > 0)
                total = score + mood + trans + anchor + genre
                if req.strict:
                    total = total - STRICT_PENALTY * ~mixable
                used = np.isin(self.song_group[cand], self.song_group[path]) | np.isin(
                    self.artist[cand], self.artist[path]
                )
                total = np.where(used, -np.inf, total)
                k = min(beam_width, len(cand))
                best = np.argpartition(-total, k - 1)[:k]
                scored += [
                    (float(total[i]), [*path, int(cand[i])]) for i in best if np.isfinite(total[i])
                ]
            if not scored:
                raise ValueError("Could not extend the set; try relaxing the filters")
            scored.sort(key=lambda x: -x[0])
            beams, seen = [], set()
            for s, p in scored:  # keep distinct endings so beams don't collapse
                if p[-1] in seen:
                    continue
                seen.add(p[-1])
                beams.append((s, p))
                if len(beams) == beam_width:
                    break

        return self.assemble(beams[0][1], targets, req)

    # ------------------------------------------------------------ assemble
    def assemble(self, path: Sequence[int], targets: np.ndarray, req: SetRequest) -> DJSet:
        """Turn an ordered list of track indices into a :class:`DJSet`."""
        path = list(path)
        t = self.tracks.loc[path].reset_index(drop=True)
        out = pd.DataFrame(
            {
                "position": np.arange(1, len(path) + 1),
                "track_id": t["track_id"],
                "track_name": t["track_name"],
                "artists": t["artists"],
                "bpm": t["tempo"].round(0).astype(int),
                "camelot": [
                    camelot(int(k), int(m)) for k, m in zip(t["key"], t["mode"], strict=True)
                ],
                "energy": t["energy"].round(2),
                "valence": t["valence"].round(2),
                "mood": [
                    nearest_mood(v, e) for v, e in zip(t["valence"], t["energy"], strict=True)
                ],
                "target_valence": targets[:, 0].round(2),
                "target_energy": targets[:, 1].round(2),
                "danceability": t["danceability"],
                "tempo": t["tempo"],
                "duration_ms": t["duration_ms"],
                "popularity": t["popularity"],
                "genres": t["genres"],
            }
        )
        transitions = []
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            change, factor, tempo_s = tempo_match(self.tempo[a], self.tempo[b])
            change, factor, tempo_s = float(change), float(factor), float(tempo_s)
            key_s = float(
                key_compatibility(self.key_num[a], self.minor[a], self.key_num[b], self.minor[b])
            )
            rel = key_relation(
                int(self.key_num[a]), bool(self.minor[a]), int(self.key_num[b]), bool(self.minor[b])
            )
            style = float(self.rec.vectors_[a] @ self.rec.vectors_[b])
            jump = abs(self.energy[b] - self.energy[a] - (targets[i + 1, 1] - targets[i, 1]))
            energy_s = max(0.0, 1.0 - jump / 0.25)
            score = 100 * (0.4 * tempo_s + 0.35 * key_s + 0.15 * energy_s + 0.10 * max(style, 0))
            mixable = abs(change) <= MIXABLE_TEMPO and key_s > 0
            rating = "smooth" if mixable and score >= 75 else ("ok" if mixable else "tricky")
            relation = {1.0: "normal", 2.0: "half-time", 0.5: "double-time"}[factor]
            transitions.append(
                Transition(
                    position=i + 1,
                    from_track=f"{out.track_name[i]} - {out.artists[i]}",
                    to_track=f"{out.track_name[i + 1]} - {out.artists[i + 1]}",
                    bpm_from=self.tempo[a],
                    bpm_to=self.tempo[b],
                    tempo_change_pct=change * 100,
                    tempo_relation=relation,
                    key_from=out.camelot[i],
                    key_to=out.camelot[i + 1],
                    key_relation=rel,
                    energy_from=self.energy[a],
                    energy_to=self.energy[b],
                    score=round(score, 1),
                    rating=rating,
                    plan=plan_transition(
                        self.tracks.loc[a], self.tracks.loc[b], change, factor, key_s, rel
                    ),
                )
            )
        return DJSet(out, transitions, targets, req)
