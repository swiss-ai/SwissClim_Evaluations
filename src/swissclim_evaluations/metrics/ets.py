from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr
from scores.categorical import BinaryContingencyManager


def _calculate_ets_for_thresholds(
    ds_target: xr.Dataset, ds_prediction: xr.Dataset, thresholds: list[int]
) -> pd.DataFrame:
    # ds_target (ground truth), ds_prediction (model)
    variables = list(ds_target.data_vars)
    metrics_dict: dict[str, dict[str, float]] = {}

    for var in variables:
        da_target = ds_target[var]
        da_prediction = ds_prediction[var]
        metrics_dict[var] = {}
        for threshold in thresholds:
            # Dask-friendly quantile over all dims
            quantile = float(
                da_target.quantile(threshold / 100.0, skipna=True)
                .compute()
                .item()
            )
            obs_events = da_target >= quantile  # targets events
            fcst_events = da_prediction >= quantile  # predictions events
            bcm = BinaryContingencyManager(
                fcst_events=fcst_events, obs_events=obs_events
            )
            basic_cm = bcm.transform(reduce_dims="all")
            ets_score = basic_cm.equitable_threat_score()
            metrics_dict[var][f"ETS {threshold}%"] = float(ets_score.values)

    return pd.DataFrame.from_dict(metrics_dict, orient="index")


def run(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    out_root: Path | None = None,
    metrics_cfg: dict[str, Any] | None = None,
) -> None:
    ets_cfg = (metrics_cfg or {}).get("ets", {})
    thresholds = ets_cfg.get("thresholds", [50, 60, 70, 80, 90])
    df = _calculate_ets_for_thresholds(ds_target, ds_prediction, thresholds)

    # Quick console feedback
    print("Equitable Threat Score (targets vs predictions) — first 5 rows:")
    print(df.head())

    # Always export CSV
    if out_root is not None:
        section_output = out_root / "ets"
        section_output.mkdir(parents=True, exist_ok=True)
        out_csv = section_output / "ets_metrics.csv"
        df.to_csv(out_csv)
        print(f"[ets] saved {out_csv}")
