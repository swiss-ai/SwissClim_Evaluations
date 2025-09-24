from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scores.categorical import BinaryContingencyManager


def _calculate_ets_for_thresholds(
    ds_target: xr.Dataset, ds_prediction: xr.Dataset, thresholds: list[int]
) -> pd.DataFrame:
    # ds_target (ground truth), ds_prediction (model)
    variables = list(ds_target.data_vars)
    metrics_dict: dict[str, dict[str, float]] = {}

    try:
        from ..progress import iter_progress  # type: ignore

        _iter_vars = iter_progress(
            variables, module="ets", total=len(variables)
        )
    except Exception:  # pragma: no cover
        _iter_vars = variables
    for var in _iter_vars:
        da_target = ds_target[var]
        da_prediction = ds_prediction[var]
        metrics_dict[var] = {}
        # Nested threshold loop; lightweight progress (avoid nested rich bars) -> plain iteration
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
        from ..helpers import build_output_filename

        def _extract_init_range(ds: xr.Dataset):
            if "init_time" not in ds:
                return None
            try:
                vals = ds["init_time"].values
                if vals.size == 0:
                    return None
                start = np.datetime64(vals.min()).astype("datetime64[h]")
                end = np.datetime64(vals.max()).astype("datetime64[h]")

                def _fmt(x):
                    return (
                        np.datetime_as_string(x, unit="h")
                        .replace("-", "")
                        .replace(":", "")
                        .replace("T", "")
                    )

                return (_fmt(start), _fmt(end))
            except Exception:
                return None

        def _extract_lead_range(ds: xr.Dataset):
            if "lead_time" not in ds:
                return None
            try:
                vals = ds["lead_time"].values
                if vals.size == 0:
                    return None
                hours = (vals / np.timedelta64(1, "h")).astype(int)
                sh = int(hours.min())
                eh = int(hours.max())

                def _fmt(h: int) -> str:
                    return f"{h:03d}h"

                return (_fmt(sh), _fmt(eh))
            except Exception:
                return None

        init_range = _extract_init_range(ds_prediction)
        lead_range = _extract_lead_range(ds_prediction)
        section_output = out_root / "ets"
        section_output.mkdir(parents=True, exist_ok=True)
        out_csv = section_output / build_output_filename(
            metric="ets_metrics",
            variable=None,
            level=None,
            qualifier="averaged" if (init_range or lead_range) else None,
            init_time_range=init_range,
            lead_time_range=lead_range,
            ensemble=None,
            ext="csv",
        )
        df.to_csv(out_csv)
        print(f"[ets] saved {out_csv}")
