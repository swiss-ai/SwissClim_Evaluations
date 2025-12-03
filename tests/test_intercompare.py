import pandas as pd

from swissclim_evaluations.intercompare import (
    intercompare_deterministic_metrics,
    intercompare_probabilistic,
)


def test_intercompare_metrics_per_level(tmp_path):
    # Setup model directories
    model_a = tmp_path / "modelA"
    model_b = tmp_path / "modelB"
    model_a.mkdir()
    model_b.mkdir()

    # Setup deterministic output dirs
    (model_a / "deterministic").mkdir()
    (model_b / "deterministic").mkdir()

    # Create fake per-level metrics
    df_a = pd.DataFrame({"variable": ["temp", "temp"], "level": [500, 850], "MAE": [1.0, 1.2]})
    df_b = pd.DataFrame({"variable": ["temp", "temp"], "level": [500, 850], "MAE": [0.9, 1.1]})

    df_a.to_csv(
        model_a / "deterministic" / "deterministic_metrics_per_level_ensmean.csv", index=False
    )
    df_b.to_csv(
        model_b / "deterministic" / "deterministic_metrics_per_level_ensmean.csv", index=False
    )

    # Run intercomparison
    out_root = tmp_path / "output"
    intercompare_deterministic_metrics([model_a, model_b], ["ModelA", "ModelB"], out_root)

    # Check result
    out_file = out_root / "deterministic" / "metrics_per_level_combined.csv"
    assert out_file.exists()

    df_comb = pd.read_csv(out_file)
    assert len(df_comb) == 4
    assert "model" in df_comb.columns
    assert set(df_comb["model"].unique()) == {"ModelA", "ModelB"}


def test_intercompare_probabilistic_per_level(tmp_path):
    # Setup model directories
    model_a = tmp_path / "modelA"
    model_b = tmp_path / "modelB"
    model_a.mkdir()
    model_b.mkdir()

    # Setup probabilistic output dirs
    (model_a / "probabilistic").mkdir()
    (model_b / "probabilistic").mkdir()

    # Create fake per-level CRPS summaries
    df_a = pd.DataFrame({"variable": ["temp", "temp"], "level": [500, 850], "CRPS": [0.5, 0.6]})
    df_b = pd.DataFrame({"variable": ["temp", "temp"], "level": [500, 850], "CRPS": [0.4, 0.5]})

    df_a.to_csv(model_a / "probabilistic" / "crps_summary_per_level_ensprob.csv", index=False)
    df_b.to_csv(model_b / "probabilistic" / "crps_summary_per_level_ensprob.csv", index=False)

    # Run intercomparison
    out_root = tmp_path / "output"
    intercompare_probabilistic([model_a, model_b], ["ModelA", "ModelB"], out_root)

    # Check result
    out_file = out_root / "probabilistic" / "crps_summary_per_level_combined.csv"
    assert out_file.exists()

    df_comb = pd.read_csv(out_file)
    assert len(df_comb) == 4
    assert "model" in df_comb.columns
    assert set(df_comb["model"].unique()) == {"ModelA", "ModelB"}
