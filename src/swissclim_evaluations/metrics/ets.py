from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scores.categorical import BinaryContingencyManager

from .. import console as c
from ..dask_utils import calculate_dynamic_chunk_size, compute_jobs, compute_quantile_preserving


def _compute_ets_raw(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    thresholds: list[int],
    preserve_dims: list[str] | None = None,
) -> dict[str, xr.DataArray]:
    variables = list(ds_target.data_vars)
    computed_results = {}
    q_values = [t / 100.0 for t in thresholds]

    for var in variables:
        da_target = ds_target[var]
        da_prediction = ds_prediction[var]

        # Determine dimensions to reduce for quantile calculation
        reduce_dims = list(da_target.dims)
        if preserve_dims:
            reduce_dims = [d for d in reduce_dims if d not in preserve_dims]

        # Compute quantiles lazily.
        # Use custom quantile to avoid memory issues
        quantiles = compute_quantile_preserving(da_target, q_values, preserve_dims)

        # Create events. Broadcasting will add 'quantile' dimension.
        obs_events = da_target >= quantiles
        fcst_events = da_prediction >= quantiles

        bcm = BinaryContingencyManager(fcst_events=fcst_events, obs_events=obs_events)
        bcm = bcm.transform(reduce_dims=reduce_dims)

        ets_score = bcm.equitable_threat_score()

        # Return lazy object
        computed_results[var] = ets_score

    return computed_results


def _calculate_ets_for_thresholds(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    thresholds: list[int],
    chunk_size: int = 20,
) -> pd.DataFrame:
    raw_results = _compute_ets_raw(ds_target, ds_prediction, thresholds)

    # Compute all at once
    if not raw_results:
        return pd.DataFrame()

    def _process_batch(batch_jobs: list[dict[str, Any]]):
        # We don't save to disk here, but we could populate a shared list.
        # However, passing a callback allows for potential memory clearing
        # if job results were large.
        pass

    jobs = [{"var": var, "lazy": val} for var, val in raw_results.items()]
    compute_jobs(
        jobs,
        key_map={"lazy": "res"},
        chunk_size=chunk_size,
        desc="Computing ETS scores",
    )

    metrics_dict: dict[str, dict[str, float]] = {}

    for job in jobs:
        var = job["var"]
        ets_score = job["res"]
        metrics_dict[var] = {}
        ets_values = ets_score.values
        for i, threshold in enumerate(thresholds):
            metrics_dict[var][f"ETS {threshold}%"] = float(ets_values[i].item())

    return pd.DataFrame.from_dict(metrics_dict, orient="index")


def _calculate_ets_per_level(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    thresholds: list[int],
    chunk_size: int = 20,
) -> pd.DataFrame | None:
    variables_3d = [v for v in ds_target.data_vars if "level" in ds_target[v].dims]
    if not variables_3d:
        return None
    if "level" not in ds_target.dims:
        return None

    # Vectorized computation preserving 'level' - Optimized to batch per level
    # raw_results = _compute_ets_raw(
    #    ds_target[variables_3d], ds_prediction[variables_3d], thresholds, preserve_dims=["level"]
    # )

    jobs: list[dict[str, Any]] = []

    for var in variables_3d:
        da_target_full = ds_target[var]
        # da_pred_full = ds_prediction[var] # Unused

        # Safe access to levels
        if "level" not in da_target_full.dims:
            continue

        levels = da_target_full["level"].values
        for lvl in levels:
            # Slice for single level
            # Note: _compute_ets_raw expects Dataset input
            ds_t_slice = ds_target[[var]].sel(level=lvl, drop=True)
            ds_p_slice = ds_prediction[[var]].sel(level=lvl, drop=True)

            # Compute ETS for this level (scaleless w.r.t level)
            # The result from _compute_ets_raw is {var: DataArray(dims=[quantile])}
            # We don't use preserve_dims=["level"] because we sliced it out.
            raw_res_dict = _compute_ets_raw(ds_t_slice, ds_p_slice, thresholds)

            if var in raw_res_dict:
                jobs.append({"var": var, "level": lvl, "lazy": raw_res_dict[var]})

    if not jobs:
        return None

    def _process_batch(batch_jobs: list[dict[str, Any]]):
        pass

    compute_jobs(
        jobs,
        key_map={"lazy": "res"},
        chunk_size=chunk_size,
        desc="Computing ETS per level",
    )

    dfs = []
    for job in jobs:
        var = job["var"]
        lvl = job["level"]
        if job.get("res") is None:
            continue

        ets_score = job["res"]  # dims: [quantile] (level is dropped)

        row = {"variable": var, "level": int(lvl) if hasattr(lvl, "item") else lvl}
        vals = ets_score.values
        for i, threshold in enumerate(thresholds):
            row[f"ETS {threshold}%"] = float(
                vals[i].item() if hasattr(vals[i], "item") else vals[i]
            )
        dfs.append(row)

    if not dfs:
        return None

    df = pd.DataFrame(dfs)
    # Set index to variable to match original structure
    df = df.set_index("variable", drop=False)
    return df


def run(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    out_root: Path | None = None,
    metrics_cfg: dict[str, Any] | None = None,
    ensemble_mode: str | None = None,
    lead_policy: Any | None = None,
    performance_cfg: dict[str, Any] | None = None,
) -> None:
    ets_cfg = (metrics_cfg or {}).get("ets", {})
    chunk_size_cfg = (performance_cfg or {}).get("chunk_size")
    thresholds = ets_cfg.get("thresholds", [50, 60, 70, 80, 90])

    # Compute metrics
    # n_points = int(sum(ds_target[v].size for v in ds_target.data_vars))
    n_points = None
    num_vars = len(ds_target.data_vars)
    dynamic_chunk = calculate_dynamic_chunk_size(
        n_points=n_points,
        num_vars=num_vars,
        config_chunk_size=chunk_size_cfg,
        ds=ds_target,
    )
    report_per_level = bool(ets_cfg.get("report_per_level", True))
    reduce_ens_mean = True
    rem = ets_cfg.get("reduce_ensemble_mean")
    if rem is not None:
        reduce_ens_mean = bool(rem)
    aggregate_members_mean = bool(ets_cfg.get("aggregate_members_mean", True))

    had_ensemble_dim = ("ensemble" in ds_prediction.dims) or ("ensemble" in ds_target.dims)
    from ..helpers import (
        ensemble_mode_to_token,
        format_variable_name,
        get_colormap_for_variable,
        resolve_ensemble_mode,
        save_dataframe,
        save_figure,
    )

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

    df = _calculate_ets_for_thresholds(
        ds_target, ds_prediction, thresholds, chunk_size=dynamic_chunk
    )

    c.print("Equitable Threat Score (targets vs predictions) — first 5 rows:")
    c.print(df.head())

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
            vals = ds["lead_time"].values
            if getattr(vals, "size", 0) == 0:
                return None
            hours = (vals / np.timedelta64(1, "h")).astype(int)
            sh = int(hours.min())
            eh = int(hours.max())

            def _fmt(h: int) -> str:
                return f"{h:03d}h"

            return (_fmt(sh), _fmt(eh))

        init_range = _extract_init_range(ds_prediction)
        lead_range = _extract_lead_range(ds_prediction)

        # Compute metrics
        section_output = out_root / "ets"
        section_output.mkdir(parents=True, exist_ok=True)

        if members_indices is None:
            if resolved_mode == "mean" and had_ensemble_dim and reduce_ens_mean:
                ens_token = ensemble_mode_to_token("mean")
            elif resolved_mode == "pooled" and had_ensemble_dim:
                ens_token = ensemble_mode_to_token("pooled")
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
            save_dataframe(df, out_csv, index_label="variable", module="ets")
            # Optional per-lead wide CSV when multi-lead policy provided
            multi_lead = ("lead_time" in ds_prediction.dims) and int(
                ds_prediction.sizes.get("lead_time", 0)
            ) > 1
            if multi_lead:
                rows = []
                leads = list(ds_prediction["lead_time"].values)
                # Determine hours deterministically based on dtype of leads
                is_td = np.issubdtype(np.asarray(leads).dtype, np.timedelta64)
                for i, lt in enumerate(leads):
                    hours = int(lt / np.timedelta64(1, "h")) if is_td else int(i)
                    ds_t_i = (
                        ds_target.isel(lead_time=i, drop=False)
                        if "lead_time" in ds_target.dims
                        else ds_target
                    )
                    ds_p_i = ds_prediction.isel(lead_time=i, drop=False)
                    df_i = _calculate_ets_for_thresholds(
                        ds_t_i, ds_p_i, thresholds, chunk_size=dynamic_chunk
                    )
                    # Flatten columns with variable names to create a wide row
                    flat: dict[str, float] = {"lead_time_hours": float(hours)}
                    for var, row in df_i.iterrows():
                        for col, val in row.items():
                            flat[f"{var}_{col}"] = float(val)
                    rows.append(flat)
                if rows:
                    wide_df = pd.DataFrame(rows)
                    out_wide = section_output / "ets_metrics_by_lead_wide.csv"
                    save_dataframe(wide_df, out_wide, index=False, module="ets")
                    # Optional: line plot of thresholds vs lead_time per variable
                    # Default to True so ETS thresholds per lead are visualized
                    do_plot = bool(ets_cfg.get("line_plot", True))
                    if do_plot:
                        from ..helpers import build_output_filename

                        hours = wide_df["lead_time_hours"].values
                        # For each variable present in columns, collect threshold series
                        var_cols = [col for col in wide_df.columns if col != "lead_time_hours"]
                        # Expect columns like "<var>_ETS 50%"
                        by_var: dict[str, list[tuple[str, str]]] = {}
                        for col in var_cols:
                            if "_ETS " in col and col.endswith("%"):
                                v, rest = col.split("_ETS ", 1)
                                by_var.setdefault(v, []).append((col, rest))
                        for v, pairs in by_var.items():
                            fig, ax = plt.subplots(figsize=(10, 6))
                            pairs_sorted = sorted(pairs, key=lambda kv: int(kv[1].rstrip("%")))

                            # Use variable-specific colormap to distinguish thresholds
                            cmap_res = get_colormap_for_variable(v)
                            cmap = plt.get_cmap(cmap_res) if isinstance(cmap_res, str) else cmap_res

                            # Sample colors (avoiding very light start for sequential maps)
                            n_thresh = len(pairs_sorted)
                            colors = [cmap(i) for i in np.linspace(0.4, 1.0, n_thresh)]

                            for i, (col, tlabel) in enumerate(pairs_sorted):
                                ax.plot(
                                    hours,
                                    wide_df[col].values,
                                    marker="o",
                                    label=f"ETS {tlabel}",
                                    color=colors[i],
                                )
                            ax.set_xlabel("Lead Time [h]")
                            ax.set_ylabel("ETS")
                            display_var = str(v).split(".", 1)[1] if "." in str(v) else str(v)
                            ax.set_title(
                                f"{format_variable_name(display_var)} — ETS thresholds vs "
                                f"Lead Time",
                                fontsize=10,
                            )
                            ax.legend(ncols=min(3, len(pairs_sorted)), fontsize=10)
                            out_png = section_output / build_output_filename(
                                metric="ets_line",
                                variable=str(v),
                                level=None,
                                qualifier=None,
                                init_time_range=init_range,
                                lead_time_range=lead_range,
                                ensemble=ens_token,
                                ext="png",
                            )
                            plt.tight_layout()
                            save_figure(fig, out_png, module="ets")
                            from numpy import savez as _savez

                            hours_arr = np.asarray(hours, dtype=float)
                            data = {"lead_hours": hours_arr}
                            for col, tlabel in pairs_sorted:
                                key = f"ETS_{tlabel.rstrip('%')}"
                                data[key] = wide_df[col].values.astype(float)
                            out_npz = section_output / build_output_filename(
                                metric="ets_line",
                                variable=str(v),
                                level=None,
                                qualifier="data",
                                init_time_range=init_range,
                                lead_time_range=lead_range,
                                ensemble=ens_token,
                                ext="npz",
                            )
                            # Cast to Any to bypass mypy strict kwargs check on numpy.savez
                            _savez(str(out_npz), **data)
                            c.print(f"[ets] Saved {out_npz}")
                            out_csv = section_output / build_output_filename(
                                metric="ets_line",
                                variable=str(v),
                                level=None,
                                qualifier="by_lead",
                                init_time_range=init_range,
                                lead_time_range=lead_range,
                                ensemble=ens_token,
                                ext="csv",
                            )
                            # Build a tidy CSV: lead_hours, threshold_label, value
                            rows_csv = []
                            for col, tlabel in pairs_sorted:
                                hours_list = list(np.asarray(hours_arr, dtype=float))
                                for h, val in zip(
                                    hours_list,
                                    wide_df[col].values.tolist(),
                                    strict=False,
                                ):
                                    rows_csv.append({
                                        "variable": v,
                                        "lead_time_hours": float(h),
                                        "threshold": tlabel,
                                        "ETS": float(val),
                                    })
                            save_dataframe(
                                pd.DataFrame(rows_csv), out_csv, index=False, module="ets"
                            )
            if report_per_level:
                per_level_df = _calculate_ets_per_level(
                    ds_target, ds_prediction, thresholds, chunk_size=dynamic_chunk
                )
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
                    save_dataframe(per_level_df, out_csv_lvl, index=False, module="ets")
        else:
            per_member_dfs = []
            for mi in members_indices:
                ds_pred_m = ds_prediction.isel(ensemble=mi)
                if "ensemble" in ds_target.dims:
                    if ds_target.sizes["ensemble"] == 1:
                        ds_tgt_m = ds_target.isel(ensemble=0)
                    else:
                        ds_tgt_m = ds_target.isel(ensemble=mi)
                else:
                    ds_tgt_m = ds_target

                df_m = _calculate_ets_for_thresholds(
                    ds_tgt_m, ds_pred_m, thresholds, chunk_size=dynamic_chunk
                )
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
                save_dataframe(df_m, out_csv_m, index_label="variable", module="ets")
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
                        save_dataframe(per_level_m, out_csv_m_lvl, index=False, module="ets")

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
                    save_dataframe(pooled_df, out_pool, index_label="variable", module="ets")
