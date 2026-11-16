import pytest

from music_recommender import cli, data


@pytest.fixture(autouse=True)
def synthetic_catalogue(monkeypatch, tracks):
    monkeypatch.setattr(data, "ensure_tracks", lambda *a, **k: tracks)


def test_recommend_prints_seed_and_results(capsys):
    cli.main(["recommend", "metal_artist_0 song 1", "-k", "3"])
    out = capsys.readouterr().out
    assert "Seed: metal_artist_0 song 1" in out
    assert out.count("metal_artist") >= 3


def test_recommend_unknown_song_exits_with_message():
    with pytest.raises(SystemExit, match="No track matches"):
        cli.main(["recommend", "definitely not a song"])


@pytest.mark.parametrize("args", [["-k", "0"], ["--diversity", "2"], ["-k", "abc"]])
def test_recommend_rejects_bad_arguments(args):
    with pytest.raises(SystemExit) as exc:
        cli.main(["recommend", "metal", *args])
    assert exc.value.code == 2  # argparse usage error


def test_command_is_required():
    with pytest.raises(SystemExit):
        cli.main([])
