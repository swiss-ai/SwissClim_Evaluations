from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scores.categorical import BinaryContingencyManager


def _calculate_ets_for_thresholds(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    thresholds: list[int],
) -> pd.DataFrame:
    variables = list(ds_target.data_vars)
    metrics_dict: dict[str, dict[str, float]] = {}

    for var in variables:
        da_target = ds_target[var]
        da_prediction = ds_prediction[var]
        metrics_dict[var] = {}
        for threshold in thresholds:
            # Dask-friendly quantile over all dims
            quantile = float(da_target.quantile(threshold / 100.0, skipna=True).compute().item())
            obs_events = da_target >= quantile  # targets events
            fcst_events = da_prediction >= quantile  # predictions events

            bcm = BinaryContingencyManager(fcst_events=fcst_events, obs_events=obs_events)
            bcm = bcm.transform(reduce_dims="all")

            ets_score = bcm.equitable_threat_score()
            metrics_dict[var][f"ETS {threshold}%"] = float(ets_score.values)

    return pd.DataFrame.from_dict(metrics_dict, orient="index")


def _calculate_ets_per_level(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    thresholds: list[int],
) -> pd.DataFrame | None:
    variables_3d = [v for v in ds_target.data_vars if "level" in ds_target[v].dims]
    if not variables_3d:
        return None
    if "level" not in ds_target.dims:
        return None

    levels = ds_target.level.values
    dfs = []
    for level in levels:
        ds_t_lvl = ds_target[variables_3d].sel(level=level)
        ds_p_lvl = ds_prediction[variables_3d].sel(level=level)

        df = _calculate_ets_for_thresholds(ds_t_lvl, ds_p_lvl, thresholds)
        df["level"] = int(level)
        df["variable"] = df.index
        dfs.append(df)

    if not dfs:
        return None

    return pd.concat(dfs).reset_index(drop=True)


def run(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    out_root: Path | None = None,
    metrics_cfg: dict[str, Any] | None = None,
    ensemble_mode: str | None = None,
) -> None:
    ets_cfg = (metrics_cfg or {}).get("ets", {})
    thresholds = ets_cfg.get("thresholds", [50, 60, 70, 80, 90])
    report_per_level = bool(ets_cfg.get("report_per_level", True))
    reduce_ens_mean = True
    try:
        rem = ets_cfg.get("reduce_ensemble_mean")
        if rem is not None:
            reduce_ens_mean = bool(rem)
    except Exception:
        reduce_ens_mean = True
    aggregate_members_mean = bool(ets_cfg.get("aggregate_members_mean", True))

    # Track whether an ensemble dimension was present prior to any reduction
    had_ensemble_dim = ("ensemble" in ds_prediction.dims) or ("ensemble" in ds_target.dims)

    from ..helpers import ensemble_mode_to_token, resolve_ensemble_mode

    resolved_mode = resolve_ensemble_mode("ets", ensemble_mode, ds_target, ds_prediction)
    has_ens = "ensemble" in ds_prediction.dims
    if resolved_mode == "prob":
        raise ValueError("ensemble_mode=prob invalid for ETS metrics")
    members_indices: list[int] | None = None
    if resolved_mode == "members" and has_ens:
        members_indices = list(range(int(ds_prediction.sizes["ensemble"])))
    if resolved_mode == "mean" and has_ens and reduce_ens_mean:
        if "ensemble" in ds_prediction.dims:
            ds_prediction = ds_prediction.mean(dim="ensemble", keep_attrs=True)
        if "ensemble" in ds_target.dims:
            ds_target = ds_target.mean(dim="ensemble", keep_attrs=True)
    # pooled => leave as-is
    df = _calculate_ets_for_thresholds(ds_target, ds_prediction, thresholds)

    # Quick console feedback
    print("Equitable Threat Score (targets vs predictions) — first 5 rows:")
    print(df.head())

    # Always export CSV
    if out_root is not None:
        from ..helpers import build_output_filename, format_init_time_range

        def _extract_init_range(ds: xr.Dataset):
            if "init_time" not in ds:
                return None
            try:
                vals = ds["init_time"].values
                if vals.size == 0:
                    return None
                return format_init_time_range(vals)
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
        # Choose ensemble token: if we reduced an existing ensemble dimension, mark as ensmean.
        if members_indices is None:
            if resolved_mode == "mean" and had_ensemble_dim and reduce_ens_mean:
                ens_token = ensemble_mode_to_token("mean")
            elif resolved_mode == "pooled" and had_ensemble_dim:
                ens_token = ensemble_mode_to_token("pooled")
            else:
                ens_token = None
        else:
            ens_token = None
        out_csv = section_output / build_output_filename(
            metric="ets_metrics",
            variable=None,
            level=None,
            qualifier="averaged" if (init_range or lead_range) else None,
            init_time_range=init_range,
            lead_time_range=lead_range,
            ensemble=ens_token,
            ext="csv",
        )
        if members_indices is None:
            df.to_csv(out_csv, index_label="variable")
            print(f"[ets] saved {out_csv}")

            if report_per_level:
                per_level_df = _calculate_ets_per_level(ds_target, ds_prediction, thresholds)
                if per_level_df is not None:
                    out_csv_lvl = section_output / build_output_filename(
                        metric="ets_metrics",
                        variable=None,
                        level=None,
                        qualifier="per_level",
                        init_time_range=init_range,
                        lead_time_range=lead_range,
                        ensemble=ens_token,
                        ext="csv",
                    )
                    per_level_df.to_csv(out_csv_lvl, index=False)
                    print(f"[ets] saved {out_csv_lvl}")
        else:
            # per-member
            per_member_dfs = []
            for mi in members_indices:
                ds_pred_m = ds_prediction.isel(ensemble=mi)
                ds_tgt_m = (
                    ds_target.isel(ensemble=mi) if "ensemble" in ds_target.dims else ds_target
                )
                df_m = _calculate_ets_for_thresholds(ds_tgt_m, ds_pred_m, thresholds)
                token_m = ensemble_mode_to_token("members", mi)
                out_csv_m = section_output / build_output_filename(
                    metric="ets_metrics",
                    variable=None,
                    level=None,
                    qualifier="averaged" if (init_range or lead_range) else None,
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=token_m,
                    ext="csv",
                )
                df_m.to_csv(out_csv_m, index_label="variable")
                print(f"[ets] saved {out_csv_m}")
                per_member_dfs.append(df_m)

                if report_per_level:
                    per_level_m = _calculate_ets_per_level(ds_tgt_m, ds_pred_m, thresholds)
                    if per_level_m is not None:
                        out_csv_m_lvl = section_output / build_output_filename(
                            metric="ets_metrics",
                            variable=None,
                            level=None,
                            qualifier="per_level",
                            init_time_range=init_range,
                            lead_time_range=lead_range,
                            ensemble=token_m,
                            ext="csv",
                        )
                        per_level_m.to_csv(out_csv_m_lvl, index=False)
                        print(f"[ets] saved {out_csv_m_lvl}")

            if per_member_dfs and aggregate_members_mean:
                from ..helpers import aggregate_member_dfs

                pooled_df = aggregate_member_dfs(per_member_dfs)
                if not pooled_df.empty:
                    out_pool = section_output / build_output_filename(
                        metric="ets_metrics",
                        variable=None,
                        level=None,
                        qualifier="members_mean",
                        init_time_range=init_range,
                        lead_time_range=lead_range,
                        ensemble="enspooled",
                        ext="csv",
                    )
                    pooled_df.to_csv(out_pool, index_label="variable")
                    print(f"[ets] saved {out_pool}")
