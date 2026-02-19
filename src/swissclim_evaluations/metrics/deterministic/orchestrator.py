from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scores.functions import create_latitude_weights

from ... import console as c
from ...dask_utils import resolve_module_batching_options
from ...helpers import (
    aggregate_member_dfs,
    build_output_filename,
    ensemble_mode_to_token,
    format_init_time_range,
    format_variable_name,
    get_variable_units,
    resolve_ensemble_mode,
    save_data,
    save_dataframe,
    save_figure,
    save_metric_by_lead_tables,
)
from . import calc


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
    performance_cfg: dict[str, Any] | None = None,
) -> None:
    """Compute and write deterministic metrics CSVs."""
    cfg = (metrics_cfg or {}).get("deterministic", {})
    mode = str((plotting_cfg or {}).get("output_mode", "plot")).lower()
    save_fig = mode in ("plot", "both")
    save_npz = mode in ("npz", "both")
    include = cfg.get("include")
    std_include = cfg.get("standardized_include")
    fss_cfg = cfg.get("fss", {})
    seeps_climatology_path = cfg.get("seeps_climatology_path")
    report_per_level = bool(cfg.get("report_per_level", True))
    perf_cfg = performance_cfg or {}
    batch_opts = resolve_module_batching_options(
        performance_cfg=perf_cfg,
        default_split_level=True,
    )
    split_3d_by_level = bool(batch_opts["split_level"])
    reduce_ens_mean = True
    try:
        rem = cfg.get("reduce_ensemble_mean")
        if rem is not None:
            reduce_ens_mean = bool(rem)
    except Exception:
        reduce_ens_mean = True
    aggregate_members_mean = bool(cfg.get("aggregate_members_mean", True))

    had_ensemble_dim = ("ensemble" in ds_prediction.dims) or ("ensemble" in ds_target.dims)

    resolved_mode = resolve_ensemble_mode("deterministic", ensemble_mode, ds_target, ds_prediction)
    has_ens = "ensemble" in ds_prediction.dims
    if resolved_mode == "prob":
        raise ValueError("ensemble_mode=prob invalid for deterministic metrics")
    members_indices: list[int] | None = None
    if resolved_mode == "members" and has_ens:
        members_indices = list(range(int(ds_prediction.sizes["ensemble"])))

    if resolved_mode == "mean" and has_ens and reduce_ens_mean:
        if "ensemble" in ds_prediction.dims:
            ds_prediction = ds_prediction.mean(dim="ensemble", keep_attrs=True)
        if "ensemble" in ds_target.dims:
            ds_target = ds_target.mean(dim="ensemble", keep_attrs=True)
        if "ensemble" in ds_prediction_std.dims:
            ds_prediction_std = ds_prediction_std.mean(dim="ensemble", keep_attrs=True)
        if "ensemble" in ds_target_std.dims:
            ds_target_std = ds_target_std.mean(dim="ensemble", keep_attrs=True)

    section_output = out_root / "deterministic"

    def _extract_init_range(ds: xr.Dataset):
        if "init_time" not in ds.coords and "init_time" not in ds.dims:
            return None
        try:
            vals = ds["init_time"].values
            if vals.size == 0:
                return None
            return format_init_time_range(vals)
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
            return (f"{start_h:03d}h", f"{end_h:03d}h")
        except Exception:
            return None

    init_range = _extract_init_range(ds_prediction)
    lead_range = _extract_lead_range(ds_prediction)

    section_output.mkdir(parents=True, exist_ok=True)

    ens_token: str | None = None
    if resolved_mode == "mean" and had_ensemble_dim and reduce_ens_mean:
        ens_token = ensemble_mode_to_token("mean")
    elif resolved_mode == "pooled" and had_ensemble_dim:
        ens_token = ensemble_mode_to_token("pooled")
    elif resolved_mode == "members" and had_ensemble_dim:
        ens_token = None

    df_all_lead = pd.DataFrame()

    weights = None
    if "latitude" in ds_target.dims:
        weights = create_latitude_weights(ds_target.latitude)
        weights = weights.clip(min=0.0)

    if members_indices is None:
        try:
            multi_lead = (
                (lead_policy is not None)
                and ("lead_time" in ds_prediction.dims)
                and int(ds_prediction.sizes.get("lead_time", 0)) > 1
                and getattr(lead_policy, "mode", "first") != "first"
            )
        except Exception:
            multi_lead = False

        regular_metrics = pd.DataFrame()
        standardized_metrics = pd.DataFrame()
        per_level_metrics = pd.DataFrame()
        per_level_std = pd.DataFrame()

        if not multi_lead:
            c.info("[deterministic] Phase 1/4: regular metrics (full-field reduction)")
            regular_metrics = calc.calculate_all_metrics(
                ds_target,
                ds_prediction,
                calc_relative=True,
                include=include,
                fss_cfg=fss_cfg,
                seeps_climatology_path=seeps_climatology_path,
                compute=True,
                weights=weights,
                performance_cfg=perf_cfg,
                log_variable_progress=True,
            )
            c.info("[deterministic] Phase 2/4: standardized metrics")
            standardized_metrics = calc.calculate_all_metrics(
                ds_target_std,
                ds_prediction_std,
                calc_relative=False,
                include=std_include,
                fss_cfg=fss_cfg,
                seeps_climatology_path=seeps_climatology_path,
                compute=True,
                weights=weights,
                performance_cfg=perf_cfg,
                log_variable_progress=False,
            )

            if report_per_level:
                c.info("[deterministic] Phase 3/4: per-level metrics")
                res_lvl = calc.calculate_per_level_metrics(
                    ds_target,
                    ds_prediction,
                    calc_relative=True,
                    include=include,
                    fss_cfg=fss_cfg,
                    compute=True,
                    weights=weights,
                    performance_cfg=perf_cfg,
                    log_variable_progress=False,
                )
                if isinstance(res_lvl, pd.DataFrame):
                    per_level_metrics = res_lvl

                c.info("[deterministic] Phase 4/4: standardized per-level metrics")
                res_lvl_std = calc.calculate_per_level_metrics(
                    ds_target_std,
                    ds_prediction_std,
                    calc_relative=False,
                    include=std_include,
                    fss_cfg=fss_cfg,
                    compute=True,
                    weights=weights,
                    performance_cfg=perf_cfg,
                    log_variable_progress=False,
                )
                if isinstance(res_lvl_std, pd.DataFrame):
                    per_level_std = res_lvl_std

        df_all_lead = pd.DataFrame()
        if multi_lead:
            c.info("[deterministic] Multi-lead mode: computing lead-resolved metrics")
            df_all_lead = calc.calculate_multi_lead_metrics_split(
                ds_target=ds_target,
                ds_prediction=ds_prediction,
                include=include,
                fss_cfg=fss_cfg,
                seeps_climatology_path=seeps_climatology_path,
                weights=weights,
                split_3d_by_level=split_3d_by_level,
                performance_cfg=perf_cfg,
            )

        if not per_level_metrics.empty and "level" in per_level_metrics.columns:
            per_level_metrics["level"] = per_level_metrics["level"].astype(int)
        if not per_level_std.empty and "level" in per_level_std.columns:
            per_level_std["level"] = per_level_std["level"].astype(int)

        out_csv = section_output / build_output_filename(
            metric="deterministic_metrics",
            qualifier="averaged" if (init_range or lead_range) else None,
            init_time_range=init_range,
            lead_time_range=lead_range,
            ensemble=ens_token,
            ext="csv",
        )
        out_csv_std = section_output / build_output_filename(
            metric="deterministic_metrics",
            qualifier="standardized",
            init_time_range=init_range,
            lead_time_range=lead_range,
            ensemble=ens_token,
            ext="csv",
        )
        if not regular_metrics.empty:
            save_dataframe(regular_metrics, out_csv, index_label="variable", module="deterministic")
        if not standardized_metrics.empty:
            save_dataframe(
                standardized_metrics, out_csv_std, index_label="variable", module="deterministic"
            )

        if per_level_metrics is not None and not per_level_metrics.empty:
            out_csv_lvl = section_output / build_output_filename(
                metric="deterministic_metrics",
                qualifier="per_level",
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token,
                ext="csv",
            )
            save_dataframe(per_level_metrics, out_csv_lvl, index=False, module="deterministic")

        if per_level_std is not None and not per_level_std.empty:
            out_csv_lvl_std = section_output / build_output_filename(
                metric="deterministic_metrics",
                qualifier="standardized_per_level",
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token,
                ext="csv",
            )
            save_dataframe(per_level_std, out_csv_lvl_std, index=False, module="deterministic")

        with contextlib.suppress(Exception):
            if not regular_metrics.empty:
                c.print("Deterministic metrics (targets vs predictions) — first 5 rows:")
                c.print(regular_metrics.head())
        with contextlib.suppress(Exception):
            if not standardized_metrics.empty:
                c.print(
                    "Deterministic standardized metrics (targets vs predictions) — first 5 rows:"
                )
                c.print(standardized_metrics.head())
    else:
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

            if "ensemble" in ds_target.dims:
                if ds_target.sizes["ensemble"] == 1:
                    ds_tgt_m = ds_target.isel(ensemble=0)
                else:
                    ds_tgt_m = ds_target.isel(ensemble=mi)
            else:
                ds_tgt_m = ds_target

            if "ensemble" in ds_target_std.dims:
                if ds_target_std.sizes["ensemble"] == 1:
                    ds_tgt_m_std = ds_target_std.isel(ensemble=0)
                else:
                    ds_tgt_m_std = ds_target_std.isel(ensemble=mi)
            else:
                ds_tgt_m_std = ds_target_std

            reg_m = calc.calculate_all_metrics(
                ds_tgt_m,
                ds_pred_m,
                calc_relative=True,
                include=include,
                fss_cfg=fss_cfg,
                seeps_climatology_path=seeps_climatology_path,
                compute=True,
                weights=weights,
                performance_cfg=perf_cfg,
                log_variable_progress=True,
            )

            std_m = calc.calculate_all_metrics(
                ds_tgt_m_std,
                ds_pred_m_std,
                calc_relative=False,
                include=std_include,
                fss_cfg=fss_cfg,
                seeps_climatology_path=seeps_climatology_path,
                compute=True,
                weights=weights,
                performance_cfg=perf_cfg,
                log_variable_progress=False,
            )

            per_level_m = None
            per_level_m_std = None
            if report_per_level:
                res = calc.calculate_per_level_metrics(
                    ds_tgt_m,
                    ds_pred_m,
                    calc_relative=True,
                    include=include,
                    fss_cfg=fss_cfg,
                    compute=True,
                    weights=weights,
                    performance_cfg=perf_cfg,
                    log_variable_progress=False,
                )
                if isinstance(res, pd.DataFrame):
                    per_level_m = res

                res_std = calc.calculate_per_level_metrics(
                    ds_tgt_m_std,
                    ds_pred_m_std,
                    calc_relative=False,
                    include=std_include,
                    fss_cfg=fss_cfg,
                    compute=True,
                    weights=weights,
                    performance_cfg=perf_cfg,
                    log_variable_progress=False,
                )
                if isinstance(res_std, pd.DataFrame):
                    per_level_m_std = res_std

            if first_reg_df is None and isinstance(reg_m, pd.DataFrame):
                first_reg_df = reg_m.copy()
            if first_std_df is None and isinstance(std_m, pd.DataFrame):
                first_std_df = std_m.copy()
            token_m = ensemble_mode_to_token("members", mi)
            out_csv_m = section_output / build_output_filename(
                metric="deterministic_metrics",
                qualifier="averaged" if (init_range or lead_range) else None,
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=token_m,
                ext="csv",
            )
            out_csv_m_std = section_output / build_output_filename(
                metric="deterministic_metrics",
                qualifier="standardized",
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=token_m,
                ext="csv",
            )
            save_dataframe(reg_m, out_csv_m, index_label="variable", module="deterministic")
            save_dataframe(std_m, out_csv_m_std, index_label="variable", module="deterministic")
            pooled_metrics.append(reg_m)

            if per_level_m is not None and not per_level_m.empty:
                out_csv_m_lvl = section_output / build_output_filename(
                    metric="deterministic_metrics",
                    qualifier="per_level",
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=token_m,
                    ext="csv",
                )
                save_dataframe(per_level_m, out_csv_m_lvl, index=False, module="deterministic")

            if per_level_m_std is not None and not per_level_m_std.empty:
                out_csv_m_lvl_std = section_output / build_output_filename(
                    metric="deterministic_metrics",
                    qualifier="standardized_per_level",
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=token_m,
                    ext="csv",
                )
                save_dataframe(
                    per_level_m_std, out_csv_m_lvl_std, index=False, module="deterministic"
                )

        if pooled_metrics and aggregate_members_mean:
            pooled_df = aggregate_member_dfs(pooled_metrics)
            if not pooled_df.empty:
                out_csv_pool = section_output / build_output_filename(
                    metric="deterministic_metrics",
                    qualifier="members_mean",
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble="enspooled",
                    ext="csv",
                )
                save_dataframe(
                    pooled_df, out_csv_pool, index_label="variable", module="deterministic"
                )

    try:
        multi_lead_check = ("lead_time" in ds_prediction.dims) and int(
            ds_prediction.sizes.get("lead_time", 0)
        ) > 1
    except Exception:
        multi_lead_check = False

    if members_indices is None and multi_lead_check:
        if "df_all_lead" in locals() and not df_all_lead.empty:
            df_all = df_all_lead
        else:
            df_all = calc.calculate_multi_lead_metrics_split(
                ds_target=ds_target,
                ds_prediction=ds_prediction,
                include=include,
                fss_cfg=fss_cfg,
                seeps_climatology_path=seeps_climatology_path,
                weights=weights,
                split_3d_by_level=split_3d_by_level,
                performance_cfg=perf_cfg,
            )

        wide_df = pd.DataFrame()

        if not df_all.empty and "lead_time" in df_all.columns:

            def _to_hours(val):
                try:
                    return int(val / np.timedelta64(1, "h"))
                except Exception:
                    try:
                        return int(val)
                    except Exception:
                        return val

            df_all["lead_time_hours"] = df_all["lead_time"].apply(_to_hours)
            df_all = df_all.drop(columns=["lead_time"])
            id_cols = ["lead_time_hours", "variable"]
            if "level" in df_all.columns:
                id_cols.append("level")
            cols = id_cols + [c for c in df_all.columns if c not in id_cols]
            long_df = df_all[cols]

            long_df, wide_df = save_metric_by_lead_tables(
                long_df=long_df,
                section_output=section_output,
                metric="deterministic_metrics",
                init_time_range=init_range,
                lead_time_range=_extract_lead_range(ds_prediction),
                ensemble=ens_token,
                module="deterministic",
            )

        try:
            import matplotlib.pyplot as _plt

            if "long_df" in locals() and not long_df.empty:
                plot_id_vars = ["lead_time_hours", "variable"]
                if "level" in long_df.columns:
                    plot_id_vars.append("level")
                plot_df = long_df.melt(id_vars=plot_id_vars, var_name="metric", value_name="value")

                for v in plot_df["variable"].unique():
                    v_df = plot_df[plot_df["variable"] == v]
                    level_values: list[int | None] = [None]
                    if "level" in v_df.columns:
                        level_values = [
                            int(level_item)
                            for level_item in sorted(v_df["level"].dropna().unique().tolist())
                        ]
                        if v_df["level"].isna().any():
                            level_values.insert(0, None)
                        if not level_values:
                            level_values = [None]

                    for level_val in level_values:
                        v_lvl = v_df if level_val is None else v_df[v_df["level"] == level_val]
                        for m in v_lvl["metric"].unique():
                            subset = v_lvl[v_lvl["metric"] == m].sort_values("lead_time_hours")
                            if subset.empty:
                                continue

                            x = subset["lead_time_hours"].values
                            y = subset["value"].values
                            display_metric = str(m).replace("_", " ")
                            if save_fig:
                                fig, ax = _plt.subplots(figsize=(10, 6))
                                ax.plot(x, y, marker="o")
                                ax.set_xlabel("Lead Time [h]")
                                units = get_variable_units(ds_target, str(v))
                                ylabel = display_metric
                                if units:
                                    if m in ["MAE", "RMSE", "Bias"]:
                                        ylabel += f" [{units}]"
                                    elif m == "MSE":
                                        ylabel += f" [{units}$^2$]"
                                ax.set_ylabel(ylabel)
                                display_var = str(v).split(".", 1)[1] if "." in str(v) else str(v)
                                lvl_str = f" @ {level_val}" if level_val is not None else ""
                                title = (
                                    f"{format_variable_name(display_var)}"
                                    f"{lvl_str} — {display_metric}"
                                )
                                ax.set_title(
                                    title,
                                    fontsize=10,
                                )
                                out_png = section_output / build_output_filename(
                                    metric="det_line",
                                    variable=str(v),
                                    level=level_val,
                                    qualifier=str(m).replace(" ", "_"),
                                    init_time_range=init_range,
                                    lead_time_range=_extract_lead_range(ds_prediction),
                                    ensemble=ens_token,
                                    ext="png",
                                )
                                _plt.tight_layout()
                                save_figure(fig, out_png, module="deterministic")
                                _plt.close(fig)

                            if save_npz:
                                out_npz = section_output / build_output_filename(
                                    metric="det_line",
                                    variable=str(v),
                                    level=level_val,
                                    qualifier=str(m).replace(" ", "_") + "_data",
                                    init_time_range=init_range,
                                    lead_time_range=_extract_lead_range(ds_prediction),
                                    ensemble=ens_token,
                                    ext="npz",
                                )
                                save_data(
                                    out_npz,
                                    lead_hours=x.astype(float),
                                    values=y.astype(float),
                                    metric=str(m),
                                    variable=str(v),
                                    level=level_val,
                                    module="deterministic",
                                )
                            out_csv = section_output / build_output_filename(
                                metric="det_line",
                                variable=str(v),
                                level=level_val,
                                qualifier=str(m).replace(" ", "_") + "_by_lead",
                                init_time_range=init_range,
                                lead_time_range=_extract_lead_range(ds_prediction),
                                ensemble=ens_token,
                                ext="csv",
                            )
                            df_out = pd.DataFrame({"lead_time_hours": x, m: y})
                            if level_val is not None:
                                df_out.insert(1, "level", int(level_val))
                            save_dataframe(df_out, out_csv, index=False, module="deterministic")
        except Exception:
            pass
