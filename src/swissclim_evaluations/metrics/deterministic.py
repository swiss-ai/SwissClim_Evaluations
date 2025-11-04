from __future__ import annotations

import contextlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scores.continuous import mae, mse, rmse
from scores.continuous.correlation import pearsonr
from scores.spatial import fss_2d


def _window_size(ds: xr.Dataset) -> tuple[int, int]:
    """Heuristic FSS window size based on grid shape.

    Uses latitude/longitude dims; pick ~10% of the larger dimension, min 1.
    """
    max_spatial_dim = max(int(ds.longitude.size), int(ds.latitude.size))
    ws = max(1, max_spatial_dim // 10)
    return (ws, ws)


def _calculate_all_metrics(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    calc_relative: bool,
    n_points: int,
    include: list[str] | None,
    fss_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute scalar deterministic metrics per variable.

    Metrics supported:
      - MAE, RMSE, MSE, Pearson R, FSS
      - Relative MAE, Relative L1, Relative L2 (when calc_relative=True)
    """
    variables = list(ds_target.data_vars)
    metrics_dict: dict[str, dict[str, float]] = {}

    # ---- FSS configuration
    auto_window_size = _window_size(ds_target)
    fss_quantile = 0.90
    fss_window_size: tuple[int, int] | None = None
    fss_thresholds_per_var: dict[str, float] | None = None
    no_event_value: float = 1.0  # perfect score if both fields have no events

    if isinstance(fss_cfg, dict):
        # quantile: allow 0-100 as percent or 0-1 as fraction
        q_raw = fss_cfg.get("quantile")
        if q_raw is not None:
            try:
                q_val = float(q_raw)
                if q_val > 1.0:
                    q_val /= 100.0
                if 0.0 < q_val < 1.0:
                    fss_quantile = q_val
            except Exception:
                pass
        # window size: int -> square; iterable -> first two
        ws = fss_cfg.get("window_size")
        if ws is not None:
            try:
                if isinstance(ws, int):
                    fss_window_size = (max(1, int(ws)), max(1, int(ws)))
                elif isinstance(ws, Iterable) and not isinstance(ws, (str | bytes)):
                    ws_list = list(ws)
                    if len(ws_list) >= 2:
                        fss_window_size = (
                            max(1, int(ws_list[0])),
                            max(1, int(ws_list[1])),
                        )
            except Exception:
                fss_window_size = None
        # thresholds: per-variable mapping only
        th_map = fss_cfg.get("thresholds")
        if isinstance(th_map, dict):
            tmp: dict[str, float] = {}
            for k, v in th_map.items():
                try:
                    tmp[str(k)] = float(v)
                except Exception:
                    continue
            if tmp:
                fss_thresholds_per_var = tmp

    if fss_window_size is None:
        fss_window_size = auto_window_size

    all_metric_names = {
        "MAE",
        "RMSE",
        "MSE",
        "Relative MAE",
        "Pearson R",
        "FSS",
        "Relative L1",
        "Relative L2",
    }
    metrics_to_compute = all_metric_names if include is None else set(include)

    for var in variables:
        y_true = ds_target[var]
        y_pred = ds_prediction[var]
        row: dict[str, float] = {}

        # Base error metrics
        if (include is None) or ("MAE" in metrics_to_compute):
            mae_val = float(mae(y_true, y_pred))
            row["MAE"] = mae_val
        else:
            mae_val = np.nan

        if (include is None) or ("RMSE" in metrics_to_compute):
            rmse_val = float(rmse(y_true, y_pred))
            row["RMSE"] = rmse_val
        else:
            rmse_val = np.nan

        if (include is None) or ("MSE" in metrics_to_compute):
            mse_val = float(mse(y_true, y_pred))
            row["MSE"] = mse_val

        # Correlation
        if (include is None) or ("Pearson R" in metrics_to_compute):
            row["Pearson R"] = float(pearsonr(y_true, y_pred))

        # FSS
        if (include is None) or ("FSS" in metrics_to_compute):
            # Determine event threshold: explicit per-variable threshold wins, else quantile
            if fss_thresholds_per_var and (var in fss_thresholds_per_var):
                event_threshold = float(fss_thresholds_per_var[var])
            else:
                q_da = y_true.quantile(fss_quantile, skipna=True)
                event_threshold = float(q_da.compute().item())
            try:
                # Early exit: if both fields have no events anywhere → perfect score
                yt_evt = bool((y_true >= event_threshold).any().compute().item())
                yp_evt = bool((y_pred >= event_threshold).any().compute().item())
                if (not yt_evt) and (not yp_evt):
                    row["FSS"] = no_event_value
                else:
                    spatial_dims = ["latitude", "longitude"]

                    # Proper call to fss_2d with config-driven arguments only
                    out = fss_2d(
                        y_pred.compute(),
                        y_true.compute(),
                        event_threshold=event_threshold,
                        window_size=fss_window_size,
                        spatial_dims=spatial_dims,
                        # this has no effect here, unfortunately we need to compute() above
                        dask="parallelized",
                    )
                    # Reduce to scalar
                    if isinstance(out, xr.DataArray):
                        try:
                            evt_t = (y_true >= event_threshold).any(dim=spatial_dims, skipna=True)
                            evt_p = (y_pred >= event_threshold).any(dim=spatial_dims, skipna=True)
                            no_evt = (~evt_t) & (~evt_p)
                            out = out.where(~no_evt, other=1.0)
                        except Exception:
                            pass
                        fss_scalar = float(out.mean(skipna=True).compute().item())
                    else:
                        fss_scalar = float(out)
                    row["FSS"] = fss_scalar
            except Exception as e:
                # Surface the error context once while keeping pipeline resilient
                with contextlib.suppress(Exception):
                    print(f"[deterministic:FSS] fss_2d failed for var='{var}': {e!r}")
                row["FSS"] = float("nan")

        # Relative metrics
        if calc_relative:
            # Denominator norms on the target
            l1_norm = float(np.abs(y_true).sum(skipna=True).compute())
            l2_norm = float(((y_true**2).sum(skipna=True).compute()) ** 0.5)
            mean_abs = float(np.abs(y_true).mean(skipna=True).compute())

            # Mean-based relative MAE (keep for continuity and interpretability)
            if (include is None) or ("Relative MAE" in metrics_to_compute):
                row["Relative MAE"] = (mae_val / mean_abs) if mean_abs else float("nan")

            # True norm-based relative errors: ||e||1 / ||y||1 and ||e||2 / ||y||2
            err = y_pred - y_true
            l1_err = float(np.abs(err).sum(skipna=True).compute())
            l2_err = float(((err**2).sum(skipna=True).compute()) ** 0.5)

            eps = 1e-12
            if (include is None) or ("Relative L1" in metrics_to_compute):
                row["Relative L1"] = (l1_err / l1_norm) if (abs(l1_norm) > eps) else float("nan")
            if (include is None) or ("Relative L2" in metrics_to_compute):
                row["Relative L2"] = (l2_err / l2_norm) if (abs(l2_norm) > eps) else float("nan")

        # If include provided, trim extra keys
        if include is not None:
            row = {k: v for k, v in row.items() if k in include}

        metrics_dict[var] = row

    return pd.DataFrame.from_dict(metrics_dict, orient="index")


def run(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    ds_target_std: xr.Dataset,
    ds_prediction_std: xr.Dataset,
    out_root: Path,
    plotting_cfg: dict[str, Any] | None,
    metrics_cfg: dict[str, Any] | None,
    ensemble_mode: str | None = None,
    lead_policy: Any | None = None,
) -> None:
    """Compute and write deterministic metrics CSVs.

    Produces two CSV files by default under out_root/deterministic:
      - deterministic_metrics_[...].csv (relative metrics included)
      - deterministic_metrics_standardized_[...].csv (on standardized fields)

    When ensemble_mode='members', writes per-member CSVs (ens0..ensN) plus an
    aggregated 'members_mean' CSV if aggregate_members_mean=True.
    """
    cfg = (metrics_cfg or {}).get("deterministic", {})
    include = cfg.get("include")
    std_include = cfg.get("standardized_include")
    fss_cfg = cfg.get("fss", {})
    reduce_ens_mean = True
    try:
        rem = cfg.get("reduce_ensemble_mean")
        if rem is not None:
            reduce_ens_mean = bool(rem)
    except Exception:
        reduce_ens_mean = True
    aggregate_members_mean = bool(cfg.get("aggregate_members_mean", True))

    # Track whether an ensemble dimension was present originally
    had_ensemble_dim = ("ensemble" in ds_prediction.dims) or ("ensemble" in ds_target.dims)

    from ..helpers import build_output_filename, ensemble_mode_to_token, resolve_ensemble_mode

    resolved_mode = resolve_ensemble_mode("deterministic", ensemble_mode, ds_target, ds_prediction)
    has_ens = "ensemble" in ds_prediction.dims
    if resolved_mode == "prob":
        raise ValueError("ensemble_mode=prob invalid for deterministic metrics")
    members_indices: list[int] | None = None
    if resolved_mode == "members" and has_ens:
        members_indices = list(range(int(ds_prediction.sizes["ensemble"])))
    if resolved_mode == "none" and has_ens:
        resolved_mode = "mean"  # match historical behaviour

    # Reduce ensemble by mean if requested
    if resolved_mode == "mean" and has_ens and reduce_ens_mean:
        if "ensemble" in ds_prediction.dims:
            ds_prediction = ds_prediction.mean(dim="ensemble", keep_attrs=True)
        if "ensemble" in ds_target.dims:
            ds_target = ds_target.mean(dim="ensemble", keep_attrs=True)
        if "ensemble" in ds_prediction_std.dims:
            ds_prediction_std = ds_prediction_std.mean(dim="ensemble", keep_attrs=True)
        if "ensemble" in ds_target_std.dims:
            ds_target_std = ds_target_std.mean(dim="ensemble", keep_attrs=True)

    # Compute metrics
    n_points = int(sum(ds_target[v].size for v in ds_target.data_vars))
    section_output = out_root / "deterministic"

    def _extract_init_range(ds: xr.Dataset):
        if "init_time" not in ds.coords and "init_time" not in ds.dims:
            return None
        try:
            vals = ds["init_time"].values
            if vals.size == 0:
                return None
            start = np.datetime64(vals.min()).astype("datetime64[h]")
            end = np.datetime64(vals.max()).astype("datetime64[h]")

            def _fmt_init(x):
                return (
                    np.datetime_as_string(x, unit="h")
                    .replace("-", "")
                    .replace(":", "")
                    .replace("T", "")
                )

            return (_fmt_init(start), _fmt_init(end))
        except Exception:
            return None

    def _extract_lead_range(ds: xr.Dataset):
        if "lead_time" not in ds.coords and "lead_time" not in ds.dims:
            return None
        try:
            vals = ds["lead_time"].values
            if vals.size == 0:
                return None
            hours = (vals / np.timedelta64(1, "h")).astype(int)
            start_h = int(hours.min())
            end_h = int(hours.max())

            def _fmt_lead(h: int) -> str:
                return f"{h:03d}h"

            return (_fmt_lead(start_h), _fmt_lead(end_h))
        except Exception:
            return None

    init_range = _extract_init_range(ds_prediction)
    lead_range = _extract_lead_range(ds_prediction)

    section_output.mkdir(parents=True, exist_ok=True)

    # Choose ensemble token for aggregate files
    ens_token: str | None = None
    if resolved_mode == "mean" and had_ensemble_dim and reduce_ens_mean:
        ens_token = ensemble_mode_to_token("mean")
    elif resolved_mode == "pooled" and had_ensemble_dim:
        ens_token = ensemble_mode_to_token("pooled")
    elif resolved_mode == "members" and had_ensemble_dim:
        ens_token = None  # per-member handled below

    if members_indices is None:
        # Aggregate (non per-member) outputs
        regular_metrics = _calculate_all_metrics(
            ds_target,
            ds_prediction,
            calc_relative=True,
            n_points=n_points,
            include=include,
            fss_cfg=fss_cfg,
        )
        standardized_metrics = _calculate_all_metrics(
            ds_target_std,
            ds_prediction_std,
            calc_relative=False,
            n_points=n_points,
            include=std_include,
            fss_cfg=fss_cfg,
        )

        out_csv = section_output / build_output_filename(
            metric="deterministic_metrics",
            variable=None,
            level=None,
            qualifier="averaged" if (init_range or lead_range) else None,
            init_time_range=init_range,
            lead_time_range=lead_range,
            ensemble=ens_token,
            ext="csv",
        )
        out_csv_std = section_output / build_output_filename(
            metric="deterministic_metrics",
            variable=None,
            level=None,
            qualifier="standardized",
            init_time_range=init_range,
            lead_time_range=lead_range,
            ensemble=ens_token,
            ext="csv",
        )
        regular_metrics.to_csv(out_csv)
        standardized_metrics.to_csv(out_csv_std)
        print(f"[deterministic] saved {out_csv}")
        print(f"[deterministic] saved {out_csv_std}")

        # Console summary
        try:
            print("Deterministic metrics (targets vs predictions) — first 5 rows:")
            print(regular_metrics.head())
        except Exception:
            pass
        try:
            print("Deterministic standardized metrics (targets vs predictions) — first 5 rows:")
            print(standardized_metrics.head())
        except Exception:
            pass
    else:
        # Per-member metrics
        from ..helpers import aggregate_member_dfs, build_output_filename, ensemble_mode_to_token

        pooled_metrics: list[pd.DataFrame] = []
        first_reg_df: pd.DataFrame | None = None
        first_std_df: pd.DataFrame | None = None
        for mi in members_indices:
            ds_pred_m = ds_prediction
            if "ensemble" in ds_prediction.dims:
                ds_pred_m = ds_prediction.isel(ensemble=mi)
            ds_pred_m_std = (
                ds_prediction_std.isel(ensemble=mi)
                if "ensemble" in ds_prediction_std.dims
                else ds_prediction_std
            )
            ds_tgt_m = ds_target.isel(ensemble=mi) if "ensemble" in ds_target.dims else ds_target
            ds_tgt_m_std = (
                ds_target_std.isel(ensemble=mi)
                if "ensemble" in ds_target_std.dims
                else ds_target_std
            )

            reg_m = _calculate_all_metrics(
                ds_tgt_m,
                ds_pred_m,
                calc_relative=True,
                n_points=n_points,
                include=include,
                fss_cfg=fss_cfg,
            )
            std_m = _calculate_all_metrics(
                ds_tgt_m_std,
                ds_pred_m_std,
                calc_relative=False,
                n_points=n_points,
                include=std_include,
                fss_cfg=fss_cfg,
            )
            if first_reg_df is None:
                first_reg_df = reg_m.copy()
            if first_std_df is None:
                first_std_df = std_m.copy()
            token_m = ensemble_mode_to_token("members", mi)
            out_csv_m = section_output / build_output_filename(
                metric="deterministic_metrics",
                variable=None,
                level=None,
                qualifier="averaged" if (init_range or lead_range) else None,
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=token_m,
                ext="csv",
            )
            out_csv_m_std = section_output / build_output_filename(
                metric="deterministic_metrics",
                variable=None,
                level=None,
                qualifier="standardized",
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=token_m,
                ext="csv",
            )
            reg_m.to_csv(out_csv_m)
            std_m.to_csv(out_csv_m_std)
            print(f"[deterministic] saved {out_csv_m}")
            print(f"[deterministic] saved {out_csv_m_std}")
            pooled_metrics.append(reg_m)

        # Aggregate pooled metrics across members if requested
        if pooled_metrics and aggregate_members_mean:
            pooled_df = aggregate_member_dfs(pooled_metrics)
            if not pooled_df.empty:
                out_csv_pool = section_output / build_output_filename(
                    metric="deterministic_metrics",
                    variable=None,
                    level=None,
                    qualifier="members_mean",
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble="enspooled",
                    ext="csv",
                )
                pooled_df.to_csv(out_csv_pool)
                print(f"[deterministic] saved {out_csv_pool}")

        # Console previews for members mode
        try:
            print("Deterministic metrics (targets vs predictions) — first 5 rows:")
            df_show = (
                pooled_df
                if ("pooled_df" in locals() and pooled_df is not None and not pooled_df.empty)
                else first_reg_df
            )
            if df_show is not None:
                print(df_show.head())
        except Exception:
            pass
        try:
            print("Deterministic standardized metrics (targets vs predictions) — first 5 rows:")
            if first_std_df is not None:
                print(first_std_df.head())
        except Exception:
            pass

    # Console summary for aggregate mode
    if members_indices is None:
        try:
            print("Deterministic metrics (targets vs predictions) — first 5 rows:", flush=True)
            print(regular_metrics.head(), flush=True)
        except Exception:
            pass

    # Optional per-lead artifacts when a multi-lead policy is provided
    try:
        multi_lead = (
            (lead_policy is not None)
            and ("lead_time" in ds_prediction.dims)
            and int(ds_prediction.sizes.get("lead_time", 0)) > 1
            and getattr(lead_policy, "mode", "first") != "first"
        )
    except Exception:
        multi_lead = False
    if members_indices is None and multi_lead:
        rows = []
        wide_rows = []
        leads = list(ds_prediction["lead_time"].values)
        for i, lt in enumerate(leads):
            try:
                hours = int(np.timedelta64(lt) / np.timedelta64(1, "h"))
            except Exception:
                hours = int(i)
            ds_t_i = (
                ds_target.isel(lead_time=i, drop=False)
                if "lead_time" in ds_target.dims
                else ds_target
            )
            ds_p_i = ds_prediction.isel(lead_time=i, drop=False)
            df_i = _calculate_all_metrics(
                ds_t_i,
                ds_p_i,
                calc_relative=True,
                n_points=n_points,
                include=include,
                fss_cfg=fss_cfg,
            )
            df_i = df_i.reset_index().rename(columns={"index": "variable"})
            df_i.insert(0, "lead_time_hours", hours)
            rows.append(df_i)
            # Build a wide row with columns like var_metric
            flat: dict[str, float] = {"lead_time_hours": float(hours)}
            for _idx, row in df_i.iterrows():
                var = str(row["variable"]) if "variable" in row else None
                for col, val in row.items():
                    if col in {"lead_time_hours", "variable"}:
                        continue
                    key = f"{var}_{col}" if var is not None else col
                    with contextlib.suppress(Exception):
                        flat[key] = float(val)
            wide_rows.append(flat)
        if rows:
            long_df = pd.concat(rows, ignore_index=True)
            out_long = section_output / "metrics_by_lead_long.csv"
            long_df.to_csv(out_long, index=False)
            print(f"[deterministic] saved {out_long}")
        if wide_rows:
            wide_df = pd.DataFrame(wide_rows)
            out_wide = section_output / "metrics_by_lead_wide.csv"
            wide_df.to_csv(out_wide, index=False)
            print(f"[deterministic] saved {out_wide}")

        # Optional quick line plots over lead_time
        try:
            if (plotting_cfg or {}).get("line_plots", False):
                import matplotlib.pyplot as _plt

                # For each variable, plot MAE vs lead_time if present
                if not wide_df.empty:
                    var_cols = [c for c in wide_df.columns if c != "lead_time_hours"]
                    # Infer vars and metrics from var_metric columns
                    split = [c.split("_", 1) for c in var_cols if "_" in c]
                    by_var: dict[str, list[str]] = {}
                    for v, m in split:
                        by_var.setdefault(v, []).append(m)
                    for v, metrics in by_var.items():
                        # choose a small subset to avoid clutter
                        for m in metrics[:3]:
                            col = f"{v}_{m}"
                            fig, ax = _plt.subplots(figsize=(6, 3))
                            ax.plot(wide_df["lead_time_hours"], wide_df[col], marker="o")
                            ax.set_xlabel("lead_time (h)")
                            ax.set_ylabel(m)
                            ax.set_title(f"{v} — {m} vs lead_time")
                            out_png = section_output / f"{v}_{m}_line_vs_lead.png"
                            _plt.savefig(out_png, bbox_inches="tight", dpi=150)
                            _plt.close(fig)
        except Exception:
            pass
        try:
            print(
                "Deterministic standardized metrics (targets vs predictions) — first 5 rows:",
                flush=True,
            )
            print(standardized_metrics.head(), flush=True)
        except Exception:
            pass
