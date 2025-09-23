from swissclim_evaluations.helpers import build_output_filename


def test_minimal_metric_only():
    fn = build_output_filename(metric="metrics", ensemble=None)
    assert fn == "metrics_ensnone.csv"


def test_with_variable_and_level_and_qualifier():
    fn = build_output_filename(
        metric="map",
        variable="temperature",
        level=500,
        qualifier="averaged",
        init_time_range=("2023010100", "2023010200"),
        lead_time_range=("000h", "024h"),
        ensemble=0,
        ext="png",
    )
    assert fn == (
        "map_temperature_500_averaged_init2023010100-2023010200_lead000h-024h_ens0.png"
    )


def test_omit_variable_when_list():
    fn = build_output_filename(
        metric="crps_summary", variable=["a", "b"], ensemble="mean"
    )
    assert fn == "crps_summary_ensmean.csv"


def test_ensemble_number():
    fn = build_output_filename(
        metric="hist", variable="u10", ensemble=3, ext="npz"
    )
    assert fn == "hist_u10_ens3.npz"


def test_lead_only_range():
    fn = build_output_filename(
        metric="ets_metrics",
        lead_time_range=("000h", "048h"),
        ensemble=None,
    )
    assert fn == "ets_metrics_lead000h-048h_ensnone.csv"


def test_init_only_range():
    fn = build_output_filename(
        metric="wd_kde_wasserstein",
        init_time_range=("2023010100", "2023010300"),
        ensemble=None,
    )
    assert fn == "wd_kde_wasserstein_init2023010100-2023010300_ensnone.csv"
