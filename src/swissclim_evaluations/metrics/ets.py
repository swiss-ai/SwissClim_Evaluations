from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scores.categorical import BinaryContingencyManager

from ..lead_time_policy import LeadTimePolicy


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
    lead_policy: LeadTimePolicy | None = None,
) -> None:
    ets_cfg = (metrics_cfg or {}).get("ets", {})
    thresholds = ets_cfg.get("thresholds", [50, 60, 70, 80, 90])
    multi_lead = (
        lead_policy is not None
        and "lead_time" in ds_prediction.dims
        and int(ds_prediction.lead_time.size) > 1
        and lead_policy.mode != "first"
    )

    if not multi_lead:
        df = _calculate_ets_for_thresholds(ds_target, ds_prediction, thresholds)
    else:
        lead_hours = (
            ds_prediction["lead_time"].values // np.timedelta64(1, "h")
        ).astype(int)
        rows = []
        for li, h in enumerate(lead_hours):
            slice_tgt = ds_target.isel(lead_time=li)
            slice_pred = ds_prediction.isel(lead_time=li)
            part = _calculate_ets_for_thresholds(
                slice_tgt, slice_pred, thresholds
            )
            part.insert(0, "lead_time_hours", int(h))
            rows.append(
                part.reset_index().rename(columns={"index": "variable"})
            )
        long_df = pd.concat(rows, ignore_index=True)
        df = long_df.pivot_table(
            index="variable",
            columns="lead_time_hours",
            values=[
                c
                for c in long_df.columns
                if c not in ("variable", "lead_time_hours")
            ],
        )
        # averaged single-lead summary
        try:
            # Modern replacement for deprecated groupby(axis=1): transpose, group on index, transpose back
            single_lead = (
                df.T.groupby(level=0).mean(numeric_only=True).T
            )  # single-lead (averaged) summary
        except Exception:
            single_lead = df

    # Quick console feedback
    if not multi_lead:
        print("Equitable Threat Score (targets vs predictions) — first 5 rows:")
        print(df.head())
    else:
        print(
            "Equitable Threat Score per lead (wide format) — columns grouped by lead_time_hours"
        )
        print(df.iloc[:5, :5])

    # Always export CSV
    if out_root is not None:
        section_output = out_root / "ets"
        section_output.mkdir(parents=True, exist_ok=True)
        if not multi_lead:
            out_csv = section_output / "ets_metrics.csv"
            df.to_csv(out_csv)
            print(f"[ets] saved {out_csv}")
        else:
            out_csv_wide = section_output / "ets_metrics_by_lead_wide.csv"
            df.to_csv(out_csv_wide)
            print(f"[ets] saved {out_csv_wide}")
            single_lead_csv = section_output / "ets_metrics.csv"
            single_lead.to_csv(single_lead_csv)
            print(f"[ets] saved averaged {single_lead_csv}")
