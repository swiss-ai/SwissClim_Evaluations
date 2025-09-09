from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scores.categorical import BinaryContingencyManager


def _calculate_ets_for_thresholds(
    ds: xr.Dataset, ds_ml: xr.Dataset, thresholds: list[int]
) -> pd.DataFrame:
    variables = list(ds.data_vars)
    metrics_dict: dict[str, dict[str, float]] = {}

    for var in variables:
        y_true = ds[var]
        y_pred = ds_ml[var]
        metrics_dict[var] = {}
        for threshold in thresholds:
            quantile = float(np.quantile(y_true, threshold / 100.0))
            obs_events = y_true >= quantile
            fcst_events = y_pred >= quantile
            bcm = BinaryContingencyManager(
                fcst_events=fcst_events, obs_events=obs_events
            )
            basic_cm = bcm.transform(reduce_dims="all")
            ets_score = basic_cm.equitable_threat_score()
            metrics_dict[var][f"ETS {threshold}%"] = float(ets_score.values)

    return pd.DataFrame.from_dict(metrics_dict, orient="index")


def run(
    ds: xr.Dataset,
    ds_ml: xr.Dataset,
    out_root: Path | None = None,
    metrics_cfg: dict[str, Any] | None = None,
) -> None:
    ets_cfg = (metrics_cfg or {}).get("ets", {})
    thresholds = ets_cfg.get("thresholds", [50, 60, 70, 80, 90])
    df = _calculate_ets_for_thresholds(ds, ds_ml, thresholds)

    # Quick console feedback
    print("Equitable Threat Score (first 5 rows):")
    print(df.head())

    # Optional CSV export
    save_csv = bool(ets_cfg.get("save_csv", True))
    if out_root is not None and save_csv:
        section_output = out_root / "ets"
        section_output.mkdir(parents=True, exist_ok=True)
        df.to_csv(section_output / "ets_metrics.csv")
