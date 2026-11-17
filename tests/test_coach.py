import io
import json
import urllib.error

import pytest

from music_recommender import ContentRecommender
from music_recommender import coach as coach_mod
from music_recommender.coach import (
    DJCoach,
    LLMClient,
    LLMError,
    allowed_numbers,
    extract_json,
    is_grounded,
    keyword_plan,
    validate_plan,
)
from music_recommender.dj import DJSetBuilder, SetRequest

GENRES = ["house", "deep-house", "indie", "indie-pop", "pop", "chill", "happy", "techno"]


class FakeClient:
    """Stands in for an LLM API: returns canned replies in order."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []
        self.label = "Fake · test-model"

    def chat(self, system, user, temperature=0.3):
        self.calls.append((system, user))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture(scope="module")
def dj_set(tracks):
    b = DJSetBuilder(ContentRecommender().fit(tracks), beam_width=4, pool_size=100)
    return b.build(SetRequest("chill", "happy", 4, genres=["pop"]))


@pytest.mark.parametrize(
    ("text", "start", "end", "n", "arc", "genres"),
    [
        (
            "chill sunset set that slowly gets more upbeat, about 40 minutes",
            "chill",
            "happy",
            11,
            "straight",
            [],
        ),
        ("sad to happy, 10 songs of indie-pop", "sad", "happy", 10, "straight", ["indie-pop"]),
        ("house club set with a peak, 1 hour", "energetic", "energetic", 16, "peak", ["house"]),
        ("hello", "chill", "chill", 12, "straight", []),
        ("party 500 songs", "energetic", "energetic", 40, "straight", []),
    ],
)
def test_keyword_plan(text, start, end, n, arc, genres):
    p = keyword_plan(text, GENRES).params
    assert (p["start_mood"], p["end_mood"], p["n_tracks"], p["arc"], p["genres"]) == (
        start,
        end,
        n,
        arc,
        genres,
    )


def test_validate_plan_clamps_bad_ai_output():
    params, notes = validate_plan(
        {
            "start_mood": "Elated",
            "end_mood": "happy",
            "n_tracks": 99,
            "arc": "zigzag",
            "genres": ["house", "vaporwave"],
        },
        GENRES,
    )
    assert params == {
        "start_mood": "chill",
        "end_mood": "happy",
        "n_tracks": 40,
        "arc": "straight",
        "genres": ["house"],
    }
    assert any("vaporwave" in n for n in notes) and any("Elated".lower() in n for n in notes)


def test_extract_json():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(LLMError):
        extract_json("no json here")


def test_grounding_check():
    allowed = allowed_numbers({"bpm": 124, "pct": 2.5, "start": "2:37"})
    assert is_grounded("Speed up by 2.5% at 2:37, then blend for 4 bars", allowed)
    assert not is_grounded("Set the tempo to 128 BPM", allowed)


def test_ai_plan_is_validated():
    reply = json.dumps(
        {
            "start_mood": "chill",
            "end_mood": "energetic",
            "n_tracks": 8,
            "arc": "peak",
            "genres": ["deep-house", "made-up"],
            "reason": "Sunset vibe",
        }
    )
    plan = DJCoach(FakeClient(reply)).plan("sunset set", GENRES)
    assert plan.params["genres"] == ["deep-house"] and plan.params["arc"] == "peak"
    assert plan.notes[0] == "Sunset vibe" and plan.source.startswith("AI coach")


def test_ai_plan_falls_back_when_api_fails():
    plan = DJCoach(FakeClient(LLMError("rate limited"))).plan("sad to happy", GENRES)
    assert plan.source == "built-in parser"
    assert (plan.params["start_mood"], plan.params["end_mood"]) == ("sad", "happy")
    assert "rate limited" in plan.notes[0]


def test_builtin_explain(dj_set):
    notes = DJCoach(None).explain(dj_set)
    assert set(notes.tips) == {1, 2, 3} and set(notes.sources.values()) == {"built-in"}
    first = dj_set.transitions[0]
    assert first.key_from in notes.tips[1] and str(round(first.bpm_from)) in notes.tips[1]


def test_ai_explain_rejects_invented_numbers(dj_set):
    t1 = dj_set.transitions[0]
    reply = json.dumps(
        {
            "overview": "A smooth warm-up into a happy finish.",
            "transitions": [
                {
                    "position": 1,
                    "tip": f"Both tracks sit near {round(t1.bpm_from)} BPM, so this "
                    "is an easy blend. Focus on the bass swap.",
                },
                {"position": 2, "tip": "Push the tempo to 999 BPM for drama."},
            ],
        }
    )
    notes = DJCoach(FakeClient(reply)).explain(dj_set)
    assert notes.sources[1] == "AI" and "easy blend" in notes.tips[1]
    assert notes.sources[2] == "built-in" and "999" not in notes.tips[2]
    assert notes.sources[3] == "built-in"  # AI skipped it
    assert any("Transition 2" in w for w in notes.warnings)
    assert notes.overview.startswith("A smooth warm-up")


def test_ai_explain_falls_back_on_error(dj_set):
    notes = DJCoach(FakeClient(LLMError("timeout"))).explain(dj_set)
    assert notes.source == "Built-in coach" and "timeout" in notes.warnings[0]


def test_client_from_env_and_model_choice(monkeypatch):
    assert LLMClient.from_env({}) is None
    client = LLMClient.from_env({"GEMINI_API_KEY": "k"})
    assert client.provider == "gemini"
    groq = LLMClient.from_env({"GROQ_API_KEY": "k", "MUSIC_REC_LLM_MODEL": "pinned"})
    assert groq.provider == "groq" and groq.resolve_model() == "pinned"
    models = {
        "data": [
            {"id": f"models/{m}"}
            for m in (
                "text-embedding-004",
                "gemini-2.5-flash-lite",
                "gemini-3.1-flash-preview",
                "gemini-2.5-flash",
                "gemini-3.0-flash",
            )
        ]
    }
    monkeypatch.setattr(client, "_request", lambda path, payload=None: models)
    assert client.resolve_model() == "gemini-3.0-flash"  # newest stable non-lite flash


def test_client_chat_and_http_errors(monkeypatch):
    client = LLMClient("groq", "k", model="m")
    sent = {}

    def fake_urlopen(req, timeout):
        sent["url"], sent["body"] = req.full_url, json.loads(req.data)
        sent["auth"] = req.headers["Authorization"]
        body = json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode()
        return io.BytesIO(body)

    monkeypatch.setattr(coach_mod.urllib.request, "urlopen", fake_urlopen)
    assert client.chat("sys", "user") == "hi"
    assert sent["url"].endswith("/openai/v1/chat/completions")
    assert sent["body"]["model"] == "m" and sent["auth"] == "Bearer k"

    def failing(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many", {}, io.BytesIO(b"quota"))

    monkeypatch.setattr(coach_mod.urllib.request, "urlopen", failing)
    with pytest.raises(LLMError, match="429"):
        client.chat("sys", "user")
