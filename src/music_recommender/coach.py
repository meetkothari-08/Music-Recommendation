"""AI DJ coach: plain-English set requests and beginner mixing instructions.

Division of labour (the important design decision):

* **Code** chooses the songs and computes every number - tempos, keys,
  timings, mix lengths (see :mod:`music_recommender.dj`).
* **The language model** only (1) turns a plain-English request into set
  parameters, validated against allowed values, and (2) explains the
  pre-computed transition facts in friendly, beginner language.

AI explanations are **grounded**: the model receives the computed facts as
JSON, and every number in its answer is checked against those facts. A tip
that mentions a number the code didn't compute is rejected and replaced by
the built-in explanation. Without an API key, or if the API fails, the coach
falls back to built-in, rule-based text, so the app always works.

Any OpenAI-compatible chat API works. Two free options are preset:

* Google Gemini - set ``GEMINI_API_KEY`` (https://aistudio.google.com/apikey)
* Groq          - set ``GROQ_API_KEY``   (https://console.groq.com/keys)

Optionally set ``MUSIC_REC_LLM_MODEL`` to pin a model. Otherwise the coach
asks the provider which models exist and picks a suitable one, because
free-tier model names change often.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from music_recommender.dj import ARCS, MOODS, DJSet, SetRequest

logger = logging.getLogger(__name__)

PROVIDERS: dict[str, dict] = {
    "gemini": {
        "label": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "prefer": [r"^gemini-[\d.]+-flash$", r"^gemini-[\d.]+-flash-lite$", r"flash"],
    },
    "groq": {
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1/",
        "key_env": "GROQ_API_KEY",
        "prefer": [r"gpt-oss-120b", r"gpt-oss-20b", r"llama.*70b", r"llama", r"qwen"],
    },
}
_NON_CHAT = re.compile(r"embed|whisper|tts|audio|image|vision|guard|imagen|veo|aqa|live", re.I)


class LLMError(RuntimeError):
    pass


# ------------------------------------------------------------------ client
class LLMClient:
    """Minimal OpenAI-compatible chat client using only the standard library."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 45.0,
    ) -> None:
        if provider not in PROVIDERS and not base_url:
            raise ValueError(f"Unknown provider '{provider}'; pass base_url for custom APIs")
        self.provider = provider
        self.api_key = api_key
        self.base_url = (base_url or PROVIDERS[provider]["base_url"]).rstrip("/") + "/"
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> LLMClient | None:
        """Build a client from environment variables, or return None."""
        env = os.environ if env is None else env
        model = env.get("MUSIC_REC_LLM_MODEL") or None
        if env.get("MUSIC_REC_LLM_API_KEY") and env.get("MUSIC_REC_LLM_BASE_URL"):
            return cls(
                "custom",
                env["MUSIC_REC_LLM_API_KEY"],
                model,
                base_url=env["MUSIC_REC_LLM_BASE_URL"],
            )
        for name, cfg in PROVIDERS.items():
            if env.get(cfg["key_env"]):
                return cls(name, env[cfg["key_env"]], model)
        return None

    @property
    def label(self) -> str:
        name = PROVIDERS.get(self.provider, {}).get("label", self.provider)
        return f"{name} · {self.model}" if self.model else name

    def _request(self, path: str, payload: dict | None = None) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            data=None if payload is None else json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise LLMError(f"{self.provider} API error {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMError(f"{self.provider} API unreachable: {exc}") from exc

    def resolve_model(self) -> str:
        """Pick a chat model the provider currently offers (cached)."""
        if self.model:
            return self.model
        ids = [
            m.get("id", "").removeprefix("models/") for m in self._request("models").get("data", [])
        ]
        ids = [i for i in ids if i and not _NON_CHAT.search(i)]
        for pattern in PROVIDERS.get(self.provider, {}).get("prefer", []):
            matches = [i for i in ids if re.search(pattern, i)]
            stable = [i for i in matches if not re.search(r"preview|exp", i)]
            if stable or matches:
                self.model = sorted(stable or matches, reverse=True)[0]
                return self.model
        if not ids:
            raise LLMError("The provider returned no usable chat models")
        self.model = sorted(ids)[0]
        return self.model

    def chat(self, system: str, user: str, temperature: float = 0.3) -> str:
        payload = {
            "model": self.resolve_model(),
            "temperature": temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        data = self._request("chat/completions", payload)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response: {str(data)[:200]}") from exc


def extract_json(text: str) -> dict:
    """Parse the first JSON object in a model reply (tolerates ```json fences)."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise LLMError("The model did not return JSON")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise LLMError(f"The model returned invalid JSON: {exc}") from exc


# ------------------------------------------------------ request planning
_MOOD_WORDS: dict[str, list[str]] = {
    "sad": ["sad", "melancholic", "melancholy", "heartbreak", "heartbroken", "cry", "lonely"],
    "calm": ["calm", "peaceful", "relax", "relaxing", "sleepy", "quiet", "soft", "mellow"],
    "chill": ["chill", "sunset", "laid back", "laid-back", "lounge", "easy", "cozy", "coffee"],
    "happy": ["happy", "upbeat", "joyful", "feel good", "feel-good", "sunny", "cheerful"],
    "energetic": ["energetic", "party", "dance", "workout", "gym", "hype", "club", "banger"],
    "intense": ["intense", "angry", "aggressive", "dark", "heavy", "rage"],
}
_END_MARKERS = (
    r"\b(?:to|into|then|ending|end with|end up|finish|finishing|towards|becomes|becoming"
    r"|gets|getting|turns|turning|builds|building|goes|going)\b"
)
# Genre names that double as mood words: treat them as moods, not genre filters.
_MOOD_LIKE_GENRES = frozenset({"chill", "happy", "sad", "party", "club", "dance", "romance"})


@dataclass
class PlannedRequest:
    params: dict
    source: str
    notes: list[str] = field(default_factory=list)

    def to_request(self, **overrides) -> SetRequest:
        return SetRequest(**{**self.params, **overrides})


def _find_mood(text: str) -> str | None:
    hits = []
    for mood, words in _MOOD_WORDS.items():
        for w in words:
            m = re.search(rf"\b{re.escape(w)}\b", text)
            if m:
                hits.append((m.start(), mood))
    return min(hits)[1] if hits else None


def keyword_plan(text: str, genres: Sequence[str]) -> PlannedRequest:
    """Rule-based fallback that understands simple requests without an API."""
    t = text.lower()
    notes = []
    parts = re.split(_END_MARKERS, t, maxsplit=1)
    start = _find_mood(parts[0])
    end = _find_mood(parts[1]) if len(parts) > 1 else None
    if start is None and end is None:
        start = end = "chill"
        notes.append("No mood words found, so I used 'chill'.")
    start = start or end
    end = end or start

    n_tracks = 12
    if m := re.search(r"(\d+)\s*(?:songs|tracks)", t):
        n_tracks = int(m.group(1))
    elif m := re.search(r"(\d+)\s*(?:min|minutes|mins)\b", t):
        n_tracks = round(int(m.group(1)) / 3.8)
    elif m := re.search(r"(\d+(?:\.\d+)?)\s*(?:hour|hours|hr)\b", t):
        n_tracks = round(float(m.group(1)) * 60 / 3.8)
    if not 2 <= n_tracks <= 40:
        notes.append(f"Set length {n_tracks} is out of range, so I used a value between 2 and 40.")
    n_tracks = int(min(max(n_tracks, 2), 40))

    peak_words = r"\bpeak|warm[ -]?up|club set|cool[ -]?down|build[ -]?up\b"
    arc = "peak" if re.search(peak_words, t) else "straight"
    found = []
    for g in sorted(genres, key=len, reverse=True):  # longest first: "indie-pop" before "pop"
        if g in _MOOD_LIKE_GENRES:
            continue
        pattern = rf"(?<![\w-]){re.escape(g)}(?![\w-])"
        if re.search(pattern, t):
            found.append(g)
            t = re.sub(pattern, " ", t)
    return PlannedRequest(
        {"start_mood": start, "end_mood": end, "n_tracks": n_tracks, "arc": arc, "genres": found},
        "built-in parser",
        notes,
    )


PLAN_SYSTEM = """You turn a DJ set request into JSON settings for a music app.
Reply with ONLY a JSON object with these keys:
- "start_mood": one of {moods}
- "end_mood": one of {moods}
- "n_tracks": integer 2-40 (about 3.8 minutes per track if a duration is given)
- "arc": one of {arcs} ("peak" = warm up, peak, cool down)
- "genres": list of 0-4 genres chosen ONLY from the allowed genre list; pick genres
  that fit the request's vibe even if not named, or [] if any genre is fine
- "reason": one short sentence explaining your choices
Moods: {mood_help}"""


def validate_plan(raw: dict, genres: Sequence[str]) -> tuple[dict, list[str]]:
    """Clamp an AI-proposed plan to allowed values; return (params, notes)."""
    notes = []
    allowed = set(genres)

    def mood(key: str, default: str) -> str:
        value = str(raw.get(key, "")).lower().strip()
        if value in MOODS:
            return value
        notes.append(f"Ignored unknown {key} '{value}', used '{default}'.")
        return default

    start = mood("start_mood", "chill")
    end = mood("end_mood", start)
    try:
        n = int(raw.get("n_tracks", 12))
    except (TypeError, ValueError):
        n = 12
    arc = raw.get("arc") if raw.get("arc") in ARCS else "straight"
    picked = [g for g in raw.get("genres") or [] if g in allowed][:4]
    dropped = [g for g in raw.get("genres") or [] if g not in allowed]
    if dropped:
        notes.append(f"Ignored genres not in the catalogue: {', '.join(map(str, dropped))}.")
    return {
        "start_mood": start,
        "end_mood": end,
        "n_tracks": min(max(n, 2), 40),
        "arc": arc,
        "genres": picked,
    }, notes


# ---------------------------------------------------------- explanations
def template_tip(facts: dict, steps: Sequence[str]) -> str:
    """Built-in beginner explanation of one transition."""
    why = []
    if facts["tempo_relation"] == "normal":
        why.append(f"{facts['bpm_from']} → {facts['bpm_to']} BPM")
    else:
        why.append(f"{facts['bpm_from']} → {facts['bpm_to']} BPM ({facts['tempo_relation']})")
    why.append(f"{facts['key_from']} → {facts['key_to']} ({facts['key_relation']})")
    lines = [f"**{facts['technique'].capitalize()}** · " + " · ".join(why)]
    lines += [f"{i}. {s}" for i, s in enumerate(steps, 1)]
    lines += [f"⚠️ {w}" for w in facts["warnings"]]
    return "\n".join(lines)


EXPLAIN_SYSTEM = """You are a friendly DJ teacher for complete beginners using a two-deck controller.
You get a DJ set as JSON: the tracks and, for each transition, facts computed by software.
Rules:
- Use ONLY numbers that appear in the JSON. Never invent BPMs, keys, times, beat counts or
  percentages. If you mention a number, copy it exactly.
- Follow the given technique, timing and tempo instruction; explain them, don't change them.
- Explain jargon briefly the first time (beatmatching, EQ bass swap, Camelot keys).
- Each tip: 2-4 short sentences, encouraging, practical.
Reply with ONLY a JSON object:
{"overview": "2-3 sentences on the shape of the set and what to focus on",
 "transitions": [{"position": <int>, "tip": "<text>"}]}"""

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_SAFE_NUMBERS = {"1", "2", "3", "4"}  # small counts ("two decks", "step 1")


def allowed_numbers(*objs) -> set[str]:
    found: set[str] = set(_SAFE_NUMBERS)
    for obj in objs:
        for n in _NUMBER.findall(json.dumps(obj, ensure_ascii=False)):
            found.add(n)
            if "." in n:
                found.add(n.rstrip("0").rstrip("."))
    return found


def is_grounded(text: str, allowed: set[str]) -> bool:
    return all(n in allowed for n in _NUMBER.findall(text))


@dataclass
class CoachNotes:
    overview: str
    tips: dict[int, str]
    sources: dict[int, str]  # position -> "AI" / "built-in"
    source: str
    warnings: list[str] = field(default_factory=list)


def _set_payload(dj_set: DJSet) -> dict:
    tracks = [
        {
            "position": int(r.position),
            "track": f"{r.track_name} - {r.artists}",
            "bpm": int(r.bpm),
            "key": r.camelot,
            "mood": r.mood,
            "energy": float(r.energy),
        }
        for r in dj_set.tracks.itertuples()
    ]
    return {"tracks": tracks, "transitions": [t.facts() for t in dj_set.transitions]}


def template_overview(dj_set: DJSet) -> str:
    s = dj_set.summary
    t = dj_set.tracks
    smooth = round(s["smooth_share"] * 100)
    return (
        f"This {len(t)}-track set runs about {s['minutes']:.0f} minutes, moving from "
        f"**{t.mood.iloc[0]}** to **{t.mood.iloc[-1]}**, between {t.bpm.min()} and "
        f"{t.bpm.max()} BPM. {smooth}% of the transitions are rated smooth. Practise the "
        "bass swap on the long blends first; it's the core skill of beatmatched mixing."
    )


class DJCoach:
    """Plans sets from text and explains transitions, with or without an LLM."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client

    @property
    def source(self) -> str:
        return f"AI coach ({self.client.label})" if self.client else "Built-in coach"

    def plan(self, text: str, genres: Sequence[str]) -> PlannedRequest:
        if not text.strip():
            raise ValueError("Describe the set you want")
        if self.client is None:
            return keyword_plan(text, genres)
        system = PLAN_SYSTEM.format(
            moods=sorted(MOODS),
            arcs=sorted(ARCS),
            mood_help="; ".join(f"{m} = {MOODS[m]} (valence, energy)" for m in MOODS),
        )
        user = f"Allowed genres: {', '.join(genres)}\n\nRequest: {text}"
        try:
            raw = extract_json(self.client.chat(system, user, temperature=0.2))
        except LLMError as exc:
            logger.warning("AI planning failed, using keyword parser: %s", exc)
            plan = keyword_plan(text, genres)
            plan.notes.insert(0, f"AI unavailable ({exc}); used the built-in parser.")
            return plan
        params, notes = validate_plan(raw, genres)
        if raw.get("reason"):
            notes.insert(0, str(raw["reason"]))
        return PlannedRequest(params, self.source, notes)

    def explain(self, dj_set: DJSet) -> CoachNotes:
        builtin = {t.position: template_tip(t.facts(), t.plan.steps) for t in dj_set.transitions}
        if self.client is None:
            return CoachNotes(
                template_overview(dj_set), builtin, dict.fromkeys(builtin, "built-in"), self.source
            )
        payload = _set_payload(dj_set)
        try:
            raw = extract_json(
                self.client.chat(
                    EXPLAIN_SYSTEM, json.dumps(payload, ensure_ascii=False), temperature=0.4
                )
            )
        except LLMError as exc:
            logger.warning("AI coach failed, using built-in coach: %s", exc)
            return CoachNotes(
                template_overview(dj_set),
                builtin,
                dict.fromkeys(builtin, "built-in"),
                "Built-in coach",
                [f"AI coach unavailable ({exc}); showing built-in tips."],
            )

        tips, sources, warnings = dict(builtin), dict.fromkeys(builtin, "built-in"), []
        facts = {t.position: t.facts() for t in dj_set.transitions}
        for item in raw.get("transitions") or []:
            try:
                pos, tip = int(item["position"]), str(item["tip"]).strip()
            except (KeyError, TypeError, ValueError):
                continue
            if pos not in facts or not tip:
                continue
            allowed = allowed_numbers(facts[pos], payload["tracks"][pos - 1 : pos + 1])
            if is_grounded(tip, allowed):
                tips[pos] = tip + "\n\n" + builtin[pos]
                sources[pos] = "AI"
            else:
                warnings.append(
                    f"Transition {pos}: AI tip mentioned numbers that weren't "
                    "computed, so it was replaced by the built-in tip."
                )
        overview = str(raw.get("overview") or "").strip()
        if not overview or not is_grounded(overview, allowed_numbers(payload)):
            overview = template_overview(dj_set)
        return CoachNotes(overview, tips, sources, self.source, warnings)
