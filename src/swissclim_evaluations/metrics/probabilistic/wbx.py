from __future__ import annotations

from pathlib import Path
from typing import Any

import dask
import numpy as np
import xarray as xr
from weatherbenchX import aggregation, binning
from weatherbenchX.metrics import base as metrics_base
from weatherbenchX.metrics.probabilistic import CRPSEnsemble

from ... import console as c
from ...helpers import build_output_filename, ensemble_mode_to_token, format_init_time_range
from . import calc, io


def _compute_wbx_metric_dataset(
    metric_aggregator: aggregation.Aggregator,
    metrics: dict[str, Any],
    ds_prediction: xr.Dataset,
    ds_target: xr.Dataset,
) -> xr.Dataset:
    if hasattr(metric_aggregator, "compute_metric_values"):
        return metric_aggregator.compute_metric_values(ds_prediction, ds_target)

    # Use unique statistics computation to optimize IO
    if hasattr(metrics_base, "compute_unique_statistics_for_all_metrics"):
        statistics = metrics_base.compute_unique_statistics_for_all_metrics(
            metrics=metrics,
            predictions=ds_prediction.data_vars,
            targets=ds_target.data_vars,
        )
        aggregation_state = metric_aggregator.aggregate_statistics(statistics)
        return aggregation_state.metric_values(metrics)

    # Fallback for older wbx versions if needed
    return metric_aggregator.aggregate_stat_vars(ds_prediction, ds_target, metrics=metrics)


def run_probabilistic_wbx(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    out_root: Path,
    plotting_cfg: dict[str, Any],
    all_cfg: dict[str, Any],
    performance_cfg: dict[str, Any] | None = None,
) -> None:
    """Compute WBX temporal/spatial metrics and CSV summaries."""
    section = out_root / "probabilistic"
    section.mkdir(parents=True, exist_ok=True)
    mode = str((plotting_cfg or {}).get("output_mode", "plot")).lower()
    save_npz = mode in ("npz", "both")

    if "ensemble" not in ds_prediction.dims:
        c.print("[probabilistic] Skipping: model dataset has no 'ensemble' dimension.")
        return

    common_vars = [v for v in ds_prediction.data_vars if v in ds_target.data_vars]
    ds_pred = ds_prediction[common_vars]

    m_ens = int(getattr(ds_pred, "sizes", {}).get("ensemble", 0))
    if m_ens < 2:
        raise RuntimeError(f"WBX probabilistic metrics require ensemble size >=2. Found {m_ens}.")

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
        try:
            vals = ds["lead_time"].values
            if vals.size == 0:
                return None
            hours = (vals / np.timedelta64(1, "h")).astype(int)
            return (f"{int(hours.min()):03d}h", f"{int(hours.max()):03d}h")
        except Exception:
            return None

    init_range = _extract_init_range(ds_prediction)
    lead_range = _extract_lead_range(ds_prediction)

    seasonal = bool((plotting_cfg or {}).get("group_by_season", False))
    time_dim = (
        "init_time" if "init_time" in ds_pred.dims else ("time" if "time" in ds_pred.dims else None)
    )
    reduce_dims = [time_dim] if time_dim is not None else []
    temporal_bin_by = [binning.ByTimeUnit("season", time_dim)] if (seasonal and time_dim) else None
    metric_aggregator = aggregation.Aggregator(
        reduce_dims=reduce_dims,
        bin_by=temporal_bin_by,
        skipna=True,
    )

    metrics: dict[str, Any] = {
        "SSR": calc.RobustUnbiasedSpreadSkillRatio(ensemble_dim="ensemble"),
        "CRPS": CRPSEnsemble(ensemble_dim="ensemble"),
    }

    import gc

    variables = [str(v) for v in ds_pred.data_vars]
    results_spatial = xr.Dataset()

    for variable in variables:
        c.print(f"[probabilistic] Computing WBX metrics for {variable} (full graph)...")
        ds_p_var = ds_prediction[[variable]]
        ds_t_var = ds_target[[variable]]

        res_lazy = _compute_wbx_metric_dataset(metric_aggregator, metrics, ds_p_var, ds_t_var)
        res_computed = dask.compute(res_lazy)[0]
        # Merge incrementally instead of accumulating all results in a list;
        # this lets previous results be GC'd if no longer referenced.
        results_spatial = xr.merge([results_spatial, res_computed], compat="override")
        del res_lazy, res_computed, ds_p_var, ds_t_var
        gc.collect()
    ens_token_prob = ensemble_mode_to_token("prob")

    # Save CSV summaries (Spatially aggregated)
    if results_spatial.data_vars:
        _save_probabilistic_summaries(
            results_spatial, section, ens_token_prob, init_range, lead_range
        )

    # Save NPZ artifacts (Spatial fields)
    if save_npz and results_spatial.data_vars:
        for var_name, _da_tmp in results_spatial.data_vars.items():
            parts = var_name.split(".", 1)
            if len(parts) == 2:
                metric_name, variable = parts
            else:
                metric_name, variable = "metric", var_name

            da_metric = results_spatial[var_name]

            if "level" in da_metric.dims:
                for lvl in da_metric["level"].values:
                    npz_path = section / build_output_filename(
                        metric=f"{metric_name.lower()}_spatial",
                        variable=variable,
                        level=lvl,
                        init_time_range=init_range,
                        lead_time_range=lead_range,
                        ensemble=ens_token_prob,
                        ext="npz",
                    )
                    io.save_npz_with_coords(
                        npz_path,
                        da_metric.sel(level=lvl, drop=True),
                        module="probabilistic",
                        level=lvl,
                    )
            else:
                npz_path = section / build_output_filename(
                    metric=f"{metric_name.lower()}_spatial",
                    variable=variable,
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token_prob,
                    ext="npz",
                )
                io.save_npz_with_coords(npz_path, da_metric, module="probabilistic")


def _save_probabilistic_summaries(
    results_spatial: xr.Dataset,
    section: Path,
    ensemble_token: str | None,
    init_range: tuple[str, str] | str | None,
    lead_range: tuple[str, str] | None,
) -> None:
    """Compute spatial means and save CSV summaries."""
    import pandas as pd
    from scores.functions import create_latitude_weights

    def _lead_time_to_hours(series: pd.Series) -> pd.Series:
        vals = pd.to_timedelta(series, errors="coerce")
        if vals.notna().any():
            hours = vals.dt.total_seconds() / 3600.0
            return hours.round().astype("Int64")
        return pd.to_numeric(series, errors="coerce").round().astype("Int64")

    def _normalize_summary_lead_time(df_sub: pd.DataFrame) -> pd.DataFrame:
        if "lead_time" not in df_sub.columns:
            return df_sub
        out = df_sub.copy()
        out["lead_time"] = _lead_time_to_hours(out["lead_time"])
        return out

    def _format_prob_line_df(
        df_sub: pd.DataFrame,
        metric_name: str,
        variable: str,
        level: Any | None = None,
    ) -> pd.DataFrame:
        if "lead_time_hours" not in df_sub.columns:
            if "lead_time" in df_sub.columns:
                df_sub["lead_time_hours"] = _lead_time_to_hours(df_sub["lead_time"])
            else:
                df_sub["lead_time_hours"] = pd.Series(dtype="Int64")

        metric_col = str(metric_name).upper()
        if metric_col not in df_sub.columns:
            if "value" in df_sub.columns:
                df_sub = df_sub.rename(columns={"value": metric_col})
            else:
                value_cols = [
                    c
                    for c in df_sub.columns
                    if c not in {"lead_time", "lead_time_hours", "variable", "level"}
                ]
                if value_cols:
                    df_sub = df_sub.rename(columns={value_cols[0]: metric_col})
                else:
                    df_sub[metric_col] = np.nan

        df_sub["variable"] = str(variable)
        if level is not None:
            df_sub["level"] = int(level)

        keep_cols = ["lead_time_hours", "variable"]
        if "level" in df_sub.columns:
            keep_cols.append("level")
        keep_cols.append(metric_col)

        out = df_sub[keep_cols].copy()
        out = out.dropna(subset=["lead_time_hours"]).sort_values("lead_time_hours")
        return out.reset_index(drop=True)

    # Identify metrics present
    metrics_found = set()
    for var_name in results_spatial.data_vars:
        parts = var_name.split(".", 1)
        if len(parts) == 2:
            metrics_found.add(parts[0])

    # Compute spatial mean
    weights = None
    if "latitude" in results_spatial.coords:
        weights = create_latitude_weights(results_spatial.latitude)

    spatial_means = results_spatial.copy()
    spatial_dims = [d for d in ["latitude", "longitude"] if d in results_spatial.dims]

    if spatial_dims:
        if weights is not None:
            spatial_means = results_spatial.weighted(weights).mean(dim=spatial_dims, skipna=True)
        else:
            spatial_means = results_spatial.mean(dim=spatial_dims, skipna=True)

    # Process each metric
    for metric in metrics_found:
        # Filter variables for this metric
        # results_spatial keys are "Metric.Variable"
        metric_vars = [v for v in spatial_means.data_vars if v.startswith(f"{metric}.")]
        if not metric_vars:
            continue

        ds_metric = spatial_means[metric_vars]
        # Rename vars to remove prefix
        rename_map = {v: v.split(".", 1)[1] for v in metric_vars}
        ds_metric = ds_metric.rename(rename_map)

        # Convert to DataFrame
        if ds_metric.dims:
            df = ds_metric.to_dataframe().reset_index()
        else:
            rows: list[dict[str, Any]] = []
            for var in ds_metric.data_vars:
                rows.append(
                    {
                        "variable": str(var),
                        metric: float(np.asarray(ds_metric[var]).item()),
                    }
                )
            df = pd.DataFrame(rows)

        # Add basic metadata
        if init_range:
            init_label = (
                f"{init_range[0]}_{init_range[1]}"
                if isinstance(init_range, tuple)
                else str(init_range)
            )
            df["init_time_range"] = init_label

        df = _normalize_summary_lead_time(df)

        summary_filename = build_output_filename(
            metric=f"{metric.lower()}_summary",
            qualifier="averaged" if (init_range or lead_range) else None,
            init_time_range=init_range if isinstance(init_range, tuple) else None,
            lead_time_range=lead_range,
            ensemble=ensemble_token,
            ext="csv",
        )
        df.to_csv(section / summary_filename, index=False)

        # Save "line" format (by lead time)
        # Expected: crps_line_{var}_by_lead...
        if "lead_time" in df.columns:
            # Save per variable/level line plots data
            # crps_line_{var}_by_lead...
            for var in ds_metric.data_vars:
                # If variable has levels, we might need multiple files or one file with level col
                # For now assuming simple case or handled by build_output_filename logic if extended

                # Check for levels
                if "level" in ds_metric[var].dims:
                    for lvl in ds_metric[var]["level"].values:
                        sub = ds_metric[var].sel(level=lvl)
                        # Use Series path to avoid potential column name conflicts.
                        df_sub = sub.to_series().reset_index(name="value")
                        df_sub = _format_prob_line_df(
                            df_sub=df_sub,
                            metric_name=metric,
                            variable=str(var),
                            level=lvl,
                        )

                        line_filename = build_output_filename(
                            metric=f"{metric.lower()}_line",
                            variable=str(var),
                            level=lvl,
                            qualifier="by_lead",
                            ensemble=ensemble_token,
                            ext="csv",
                        )
                        df_sub.to_csv(section / line_filename, index=False)

                        if str(metric).upper() == "CRPS":
                            legacy_filename = build_output_filename(
                                metric="temporal_probabilistic_metrics",
                                variable=str(var),
                                level=lvl,
                                qualifier="per_lead_time",
                                ensemble=ensemble_token,
                                ext="csv",
                            )
                            df_sub.to_csv(section / legacy_filename, index=False)
                else:
                    sub = ds_metric[var]
                    df_sub = sub.to_series().reset_index(name="value")
                    df_sub = _format_prob_line_df(
                        df_sub=df_sub,
                        metric_name=metric,
                        variable=str(var),
                    )
                    line_filename = build_output_filename(
                        metric=f"{metric.lower()}_line",
                        variable=str(var),
                        qualifier="by_lead",
                        ensemble=ensemble_token,
                        ext="csv",
                    )
                    df_sub.to_csv(section / line_filename, index=False)

                    if str(metric).upper() == "CRPS":
                        legacy_filename = build_output_filename(
                            metric="temporal_probabilistic_metrics",
                            variable=str(var),
                            qualifier="per_lead_time",
                            ensemble=ensemble_token,
                            ext="csv",
                        )
                        df_sub.to_csv(section / legacy_filename, index=False)
        else:
            continue
