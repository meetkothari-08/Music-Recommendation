import pandas as pd

from music_recommender import reporting


def test_report_generation(tracks, tmp_path):
    results = reporting.run(tracks, out_dir=tmp_path, n_tasks=8, k=5, n_dj_requests=3)
    assert len(results) == 7
    report = (tmp_path / "evaluation.md").read_text(encoding="utf-8")
    assert "## Model comparison (test set)" in report
    assert "## DJ-set sequencing" in report
    for name in ("model_comparison", "genre_weight_sweep", "diversity_sweep", "dj_comparison"):
        assert (tmp_path / "figures" / f"{name}.png").stat().st_size > 0
        assert not pd.read_csv(tmp_path / f"{name}.csv").empty
