"""Standalone FSS (Fractions Skill Score) module.

Supports multiple quantile thresholds and per-member ensemble mode,
mirroring the ETS module interface.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import dask
import numpy as np
import pandas as pd
import xarray as xr
from scores.spatial import fss_2d_single_field

from .. import console as c
from ..dask_utils import compute_global_quantile


def _window_size(ds: xr.Dataset) -> tuple[int, int]:
    max_spatial_dim = max(int(ds.longitude.size), int(ds.latitude.size))
    ws = max(1, max_spatial_dim // 10)
    return (ws, ws)


def _compute_fss_for_var(
    y_pred: xr.DataArray,
    y_true: xr.DataArray,
    event_threshold: Any,
    window_size: tuple[int, int],
    preserve_dims: list[str] | None = None,
) -> xr.DataArray | None:
    spatial_dims = ["latitude", "longitude"]
    try:
        if isinstance(event_threshold, xr.DataArray | xr.Variable):

            def _fss_wrapper(p, t, th, **kwargs):
                return fss_2d_single_field(p, t, event_threshold=th, **kwargs)

            out = xr.apply_ufunc(
                _fss_wrapper,
                y_pred,
                y_true,
                event_threshold,
                input_core_dims=[spatial_dims, spatial_dims, []],
                kwargs={"window_size": window_size},
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )
        else:
            out = xr.apply_ufunc(
                fss_2d_single_field,
                y_pred,
                y_true,
                input_core_dims=[spatial_dims, spatial_dims],
                kwargs={
                    "event_threshold": event_threshold,
                    "window_size": window_size,
                },
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )

        if not isinstance(out, xr.DataArray):
            return None

        with contextlib.suppress(Exception):
            evt_t = (y_true >= event_threshold).any(dim=spatial_dims)
            evt_p = (y_pred >= event_threshold).any(dim=spatial_dims)
            no_evt = (~evt_t) & (~evt_p)
            out = out.where(~no_evt, other=1.0)

        if preserve_dims:
            reduce_dims = [d for d in out.dims if d not in preserve_dims]
            return out.mean(dim=reduce_dims, skipna=True)
        return out.mean(skipna=True)
    except Exception as e:
        c.print(f"[fss] fss_2d failed: {e!r}")
        return None


def _compute_fss_all(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    quantiles: list[float],
    window_size: tuple[int, int],
    do_per_lead: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    variables = [str(v) for v in ds_target.data_vars if v in ds_prediction.data_vars]
    if not variables:
        return pd.DataFrame(), []

    # Phase 1: compute quantile thresholds
    lazy_quantiles = []
    quantile_meta: list[tuple[str, float]] = []
    for var in variables:
        for q in quantiles:
            q_lazy = compute_global_quantile(ds_target[var], q, skipna=True)
            lazy_quantiles.append(q_lazy)
            quantile_meta.append((var, q))

    c.print(f"[fss] Computing {len(lazy_quantiles)} quantile thresholds...")
    computed_q = list(dask.compute(*lazy_quantiles))
    thresholds: dict[tuple[str, float], Any] = {}
    for (var, q), val in zip(quantile_meta, computed_q, strict=False):
        thresholds[(var, q)] = float(val.item() if hasattr(val, "item") else val)

    # Phase 2: compute FSS scores
    lazy_tasks: list[tuple[Any, Any]] = []

    # Overall FSS (reduce all dims)
    for var in variables:
        for q in quantiles:
            fss_lazy = _compute_fss_for_var(
                ds_prediction[var],
                ds_target[var],
                thresholds[(var, q)],
                window_size,
            )
            if fss_lazy is not None:
                lazy_tasks.append((("overall", var, q), fss_lazy))

    # Per-lead FSS
    if do_per_lead and "lead_time" in ds_prediction.dims:
        for var in variables:
            for q in quantiles:
                fss_lazy = _compute_fss_for_var(
                    ds_prediction[var],
                    ds_target[var],
                    thresholds[(var, q)],
                    window_size,
                    preserve_dims=["lead_time"],
                )
                if fss_lazy is not None:
                    lazy_tasks.append((("per_lead", var, q), fss_lazy))

    if not lazy_tasks:
        return pd.DataFrame(), []

    c.print(f"[fss] Computing {len(lazy_tasks)} FSS scores in one dask graph...")
    tags = [t for t, _ in lazy_tasks]
    lazies = [l for _, l in lazy_tasks]
    computed = dask.compute(*lazies)
    results = dict(zip(tags, computed, strict=False))

    # Unpack overall
    metrics_dict: dict[str, dict[str, float]] = {}
    for var in variables:
        metrics_dict[var] = {}
        for q in quantiles:
            val = results.get(("overall", var, q))
            if val is not None:
                metrics_dict[var][f"FSS {int(q * 100)}%"] = float(
                    val.item() if hasattr(val, "item") else val
                )
    overall_df = pd.DataFrame.from_dict(metrics_dict, orient="index")

    # Unpack per-lead
    per_lead_rows: list[dict[str, Any]] = []
    if do_per_lead and "lead_time" in ds_prediction.dims:
        leads_arr = ds_prediction["lead_time"].values
        is_td = np.issubdtype(np.asarray(leads_arr).dtype, np.timedelta64)
        lead_hours = [
            int(lt / np.timedelta64(1, "h")) if is_td else int(i)
            for i, lt in enumerate(leads_arr)
        ]
        for var in variables:
            for q in quantiles:
                val = results.get(("per_lead", var, q))
                if val is None:
                    continue
                for li, hours in enumerate(lead_hours):
                    row: dict[str, Any] = {
                        "lead_time_hours": float(hours),
                        "variable": str(var),
                    }
                    fss_val = val.isel(lead_time=li, drop=True).values
                    row[f"FSS {int(q * 100)}%"] = float(
                        fss_val.item() if hasattr(fss_val, "item") else fss_val
                    )
                    per_lead_rows.append(row)

    return overall_df, per_lead_rows


def run(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    out_root: Path | None = None,
    metrics_cfg: dict[str, Any] | None = None,
    plotting_cfg: dict[str, Any] | None = None,
    ensemble_mode: str | None = None,
    lead_policy: Any | None = None,
    performance_cfg: dict[str, Any] | None = None,
) -> None:
    fss_cfg = (metrics_cfg or {}).get("fss", {})
    mode = str((plotting_cfg or {}).get("output_mode", "plot")).lower()
    save_npz = mode in ("npz", "both")

    # Parse thresholds (percentiles like [75, 95])
    raw_thresholds = fss_cfg.get("thresholds", [90])
    quantiles = []
    for t in raw_thresholds:
        q = float(t)
        if q > 1.0:
            q /= 100.0
        quantiles.append(q)

    ws_raw = fss_cfg.get("window_size")
    if isinstance(ws_raw, int):
        window_size = (max(1, ws_raw), max(1, ws_raw))
    elif isinstance(ws_raw, list) and len(ws_raw) >= 2:
        window_size = (max(1, int(ws_raw[0])), max(1, int(ws_raw[1])))
    else:
        window_size = _window_size(ds_target)

    report_per_level = bool(fss_cfg.get("report_per_level", True))
    aggregate_members_mean = bool(fss_cfg.get("aggregate_members_mean", True))

    from ..helpers import (
        build_output_filename,
        ensemble_mode_to_token,
        format_init_time_range,
        resolve_ensemble_mode,
        save_dataframe,
        save_metric_by_lead_tables,
    )

    resolved_mode = resolve_ensemble_mode("fss", ensemble_mode, ds_target, ds_prediction)
    has_ens = "ensemble" in ds_prediction.dims
    if resolved_mode == "prob":
        raise ValueError("ensemble_mode=prob invalid for FSS metrics")
    members_indices: list[int] | None = None
    if resolved_mode == "members" and has_ens:
        members_indices = list(range(int(ds_prediction.sizes["ensemble"])))

    if resolved_mode == "mean" and has_ens:
        if "ensemble" in ds_prediction.dims:
            ds_prediction = ds_prediction.mean(dim="ensemble", keep_attrs=True)
        if "ensemble" in ds_target.dims:
            ds_target = ds_target.mean(dim="ensemble", keep_attrs=True)

    had_ensemble_dim = has_ens

    def _extract_init_range(ds: xr.Dataset):
        if "init_time" not in ds:
            return None
        try:
            vals = ds["init_time"].values
            return format_init_time_range(vals) if vals.size > 0 else None
        except Exception:
            return None

    def _extract_lead_range(ds: xr.Dataset):
        if "lead_time" not in ds:
            return None
        vals = ds["lead_time"].values
        if getattr(vals, "size", 0) == 0:
            return None
        hours = (vals / np.timedelta64(1, "h")).astype(int)
        return (f"{int(hours.min()):03d}h", f"{int(hours.max()):03d}h")

    init_range = _extract_init_range(ds_prediction)
    lead_range = _extract_lead_range(ds_prediction)

    _has_multi_lead = (
        "lead_time" in ds_prediction.dims
        and int(ds_prediction.sizes.get("lead_time", 0)) > 1
    )

    if out_root is None:
        return

    section_output = out_root / "fss"
    section_output.mkdir(parents=True, exist_ok=True)

    if members_indices is None:
        # Ensemble-mean or no-ensemble path
        if resolved_mode == "mean" and had_ensemble_dim:
            ens_token = ensemble_mode_to_token("mean")
        else:
            ens_token = None

        df, per_lead_rows = _compute_fss_all(
            ds_target, ds_prediction, quantiles, window_size,
            do_per_lead=_has_multi_lead,
        )

        c.print("[fss] Fractions Skill Score — first 5 rows:")
        c.print(df.head())

        out_csv = section_output / build_output_filename(
            metric="fss_metrics",
            variable=None,
            level=None,
            qualifier="averaged" if (init_range or lead_range) else None,
            init_time_range=init_range,
            lead_time_range=lead_range,
            ensemble=ens_token,
            ext="csv",
        )
        save_dataframe(df, out_csv, index_label="variable", module="fss")

        if _has_multi_lead and per_lead_rows:
            long_df = pd.DataFrame(per_lead_rows)
            save_metric_by_lead_tables(
                long_df=long_df,
                section_output=section_output,
                metric="fss_metrics",
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token,
                module="fss",
            )
    else:
        # Per-member path
        per_member_dfs: list[pd.DataFrame] = []
        all_member_lead_rows: list[dict[str, Any]] = []

        for mi in members_indices:
            ds_pred_m = ds_prediction.isel(ensemble=mi)
            if "ensemble" in ds_target.dims:
                if ds_target.sizes["ensemble"] == 1:
                    ds_tgt_m = ds_target.isel(ensemble=0)
                else:
                    ds_tgt_m = ds_target.isel(ensemble=mi)
            else:
                ds_tgt_m = ds_target

            df_m, per_lead_rows_m = _compute_fss_all(
                ds_tgt_m, ds_pred_m, quantiles, window_size,
                do_per_lead=_has_multi_lead,
            )
            token_m = ensemble_mode_to_token("members", mi)
            out_csv_m = section_output / build_output_filename(
                metric="fss_metrics",
                variable=None,
                level=None,
                qualifier="averaged" if (init_range or lead_range) else None,
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=token_m,
                ext="csv",
            )
            save_dataframe(df_m, out_csv_m, index_label="variable", module="fss")
            per_member_dfs.append(df_m)

            if per_lead_rows_m:
                for row in per_lead_rows_m:
                    row["member"] = mi
                all_member_lead_rows.extend(per_lead_rows_m)

        if all_member_lead_rows:
            lead_df = pd.DataFrame(all_member_lead_rows)
            out_lead = section_output / build_output_filename(
                metric="fss_metrics",
                variable=None,
                level=None,
                qualifier="per_member_per_lead",
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble="ensmembers",
                ext="csv",
            )
            save_dataframe(lead_df, out_lead, index=False, module="fss")

        if per_member_dfs and aggregate_members_mean:
            from ..helpers import aggregate_member_dfs

            pooled_df = aggregate_member_dfs(per_member_dfs)
            if not pooled_df.empty:
                out_pool = section_output / build_output_filename(
                    metric="fss_metrics",
                    variable=None,
                    level=None,
                    qualifier="members_mean",
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble="enspooled",
                    ext="csv",
                )
                save_dataframe(pooled_df, out_pool, index_label="variable", module="fss")
