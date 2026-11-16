import pandas as pd

from music_recommender import reporting


def test_report_generation(tracks, tmp_path):
    results = reporting.run(tracks, out_dir=tmp_path, n_tasks=8, k=5)
    assert len(results) == 7
    report = (tmp_path / "evaluation.md").read_text()
    assert "## Model comparison (test set)" in report
    for name in ("model_comparison", "genre_weight_sweep", "diversity_sweep"):
        assert (tmp_path / "figures" / f"{name}.png").stat().st_size > 0
        assert not pd.read_csv(tmp_path / f"{name}.csv").empty
