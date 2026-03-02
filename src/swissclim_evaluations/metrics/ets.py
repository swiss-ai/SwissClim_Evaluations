from __future__ import annotations

from pathlib import Path
from typing import Any

import dask
import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scores.categorical import BinaryContingencyManager

from .. import console as c
from ..dask_utils import (
    apply_split_to_dataarray,
    build_variable_level_lead_splits,
    compute_quantile_preserving,
)


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
) -> pd.DataFrame:
    variables = [str(v) for v in ds_target.data_vars if v in ds_prediction.data_vars]
    if not variables:
        return pd.DataFrame()

    metrics_dict: dict[str, dict[str, float]] = {}
    lazy_jobs: list[tuple[str, Any]] = []
    for var in variables:
        ds_t_var = ds_target[[var]]
        ds_p_var = ds_prediction[[var]]
        raw_results = _compute_ets_raw(ds_t_var, ds_p_var, thresholds)
        if var not in raw_results:
            continue

        c.print(f"Preparing ETS scores variable={var}...")
        lazy_jobs.append((var, raw_results[var]))

    if not lazy_jobs:
        return pd.DataFrame()

    c.print(f"Computing ETS scores for {len(lazy_jobs)} variable(s) in one dask graph...")
    computed = dask.compute(*[lazy for _, lazy in lazy_jobs])

    for (var, _lazy), ets_score in zip(lazy_jobs, computed, strict=False):
        if ets_score is None:
            continue
        metrics_dict[var] = {}
        ets_values = ets_score.values
        for i, threshold in enumerate(thresholds):
            metrics_dict[var][f"ETS {threshold}%"] = float(ets_values[i].item())

    return pd.DataFrame.from_dict(metrics_dict, orient="index")


def _calculate_ets_all_leads_batched(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    thresholds: list[int],
) -> list[dict[str, Any]]:
    """Compute ETS for every lead time in a single Dask graph.

    Instead of looping over lead times and triggering one dask.compute per lead,
    we preserve the 'lead_time' dimension so that quantile thresholds and ETS
    scores are computed for *all* leads simultaneously.  The result is equivalent
    to the per-lead loop but only materialises the graph once.

    Returns a list of row-dicts ready to be appended to the ``rows`` accumulator
    used in ``run()``:  [{"lead_time_hours": ..., "variable": ..., "ETS X%": ...}, ...]
    """
    leads_arr = ds_prediction["lead_time"].values
    is_td = np.issubdtype(np.asarray(leads_arr).dtype, np.timedelta64)
    lead_hours = [
        int(lt / np.timedelta64(1, "h")) if is_td else int(i) for i, lt in enumerate(leads_arr)
    ]
    n_leads = len(lead_hours)
    if n_leads == 0:
        return []

    variables = [str(v) for v in ds_target.data_vars if v in ds_prediction.data_vars]
    if not variables:
        return []

    lazy_jobs: list[tuple[str, Any]] = []
    for var in variables:
        raw = _compute_ets_raw(
            ds_target[[var]],
            ds_prediction[[var]],
            thresholds,
            preserve_dims=["lead_time"],
        )
        if var in raw:
            c.print(f"Preparing ETS scores (per lead) variable={var}...")
            lazy_jobs.append((var, raw[var]))

    if not lazy_jobs:
        return []

    c.print(f"Computing ETS scores per lead for {len(lazy_jobs)} variable(s) in one dask graph...")
    computed = dask.compute(*[lazy for _, lazy in lazy_jobs])

    rows: list[dict[str, Any]] = []
    for (var, _), ets_score in zip(lazy_jobs, computed, strict=False):
        if ets_score is None:
            continue
        # ets_score is an xr.DataArray with dims (quantile, lead_time).
        # Use named isel to avoid any ambiguity about axis order.
        for li, hours in enumerate(lead_hours):
            flat: dict[str, Any] = {"lead_time_hours": float(hours), "variable": str(var)}
            for ti, threshold in enumerate(thresholds):
                val = float(ets_score.isel(quantile=ti, lead_time=li, drop=True).values)
                flat[f"ETS {threshold}%"] = val
            rows.append(flat)

    return rows


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

    # Vectorized computation preserving 'level' - Optimized to batch per level
    # raw_results = _compute_ets_raw(
    #    ds_target[variables_3d], ds_prediction[variables_3d], thresholds, preserve_dims=["level"]
    # )

    grouped: dict[tuple[str, Any], list[tuple[int, int, Any]]] = {}
    split_specs = build_variable_level_lead_splits(
        ds_target[variables_3d],
        variables=[str(v) for v in variables_3d],
        split_level=True,
    )

    specs_by_var: dict[str, list[dict[str, Any]]] = {}
    for split_spec in split_specs:
        var_name = str(split_spec["variable"])
        specs_by_var.setdefault(var_name, []).append(split_spec)

    for var, var_specs in specs_by_var.items():
        jobs: list[dict[str, Any]] = []
        for split_spec in var_specs:
            lvl = split_spec["level"]
            init_slice = split_spec.get("init_slice", slice(None))

            ds_t_slice = xr.Dataset(
                {
                    var: apply_split_to_dataarray(
                        ds_target[var],
                        level=lvl,
                        lead_slice=slice(None),
                        init_slice=init_slice,
                    ),
                }
            )
            ds_p_slice = xr.Dataset(
                {
                    var: apply_split_to_dataarray(
                        ds_prediction[var],
                        level=lvl,
                        lead_slice=slice(None),
                        init_slice=init_slice,
                    ),
                }
            )

            raw_res_dict = _compute_ets_raw(ds_t_slice, ds_p_slice, thresholds)
            if var in raw_res_dict:
                jobs.append(
                    {
                        "var": var,
                        "level": lvl,
                        "init_start": int(split_spec.get("init_start", 0)),
                        "init_len": int(split_spec.get("init_len", 1)),
                        "lazy": raw_res_dict[var],
                    }
                )

        if not jobs:
            continue

        c.print(f"Computing ETS per level variable={var}")
        results = list(dask.compute(*[j["lazy"] for j in jobs]))
        for idx, res in enumerate(results):
            jobs[idx]["res"] = res

        for job in jobs:
            lvl = job["level"]
            if job.get("res") is None:
                continue

            # The lazy item was raw_res_dict[var] for all jobs in var_specs?
            # Wait, raw_res_dict comes from _compute_ets_raw call inside loop.
            # yes, raw_res_dict = _compute_ets_raw(ds_t_slice, ds_p_slice, thresholds)
            # So each job has a unique lazy object.

            key = (str(var), lvl)
            grouped.setdefault(key, []).append(
                (
                    int(job.get("init_start", 0)),
                    int(job.get("init_len", 1)),
                    job["res"],
                )
            )
    if not grouped:
        return None

    dfs = []
    for (var, lvl), parts in grouped.items():
        parts_sorted = sorted(parts, key=lambda item: item[0])
        weighted_sum = None
        total_weight = 0
        for _init_start, init_len, ets_score in parts_sorted:
            weight = max(1, int(init_len))
            weighted_part = ets_score * weight
            weighted_sum = weighted_part if weighted_sum is None else (weighted_sum + weighted_part)
            total_weight += weight

        if weighted_sum is None:
            continue

        ets_merged = weighted_sum / max(1, total_weight)
        row = {"variable": var, "level": int(lvl) if hasattr(lvl, "item") else lvl}
        vals = ets_merged.values
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


def _compute_all_ets(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    thresholds: list[int],
    do_per_lead: bool = False,
    do_per_level: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame | None]:
    """Compute overall ETS, per-lead ETS, and per-level ETS in a single dask.compute call.

    Compared to calling the three individual helpers separately this function:
    - Builds **all** lazy computation graphs before triggering any materialisation,
      allowing Dask to fuse and deduplicate the underlying data reads.
    - Computes quantiles for per-level ETS **once per variable** (preserving the
      ``level`` dimension) instead of once per (variable, level) slice, eliminating
      O(n_levels) redundant full-array scans per 3-D variable.

    Returns
    -------
    overall_df
        ETS scores aggregated over all dimensions, indexed by variable.
    per_lead_rows
        One row-dict per (variable, lead_time).  Empty when *do_per_lead* is False
        or the dataset has no ``lead_time`` dimension.
    per_level_df
        ETS scores per pressure level, or None when *do_per_level* is False / no 3-D vars.
    """
    q_values = [t / 100.0 for t in thresholds]
    all_variables = [str(v) for v in ds_target.data_vars if v in ds_prediction.data_vars]
    if not all_variables:
        return pd.DataFrame(), [], None

    variables_3d = [v for v in all_variables if "level" in ds_target[v].dims]
    has_lead = (
        "lead_time" in ds_prediction.dims and int(ds_prediction.sizes.get("lead_time", 0)) > 1
    )
    has_level = "level" in ds_target.dims and bool(variables_3d)

    # Each entry: (tag, lazy_DataArray)
    # tag is a tuple used to identify the result after dask.compute.
    lazy_tasks: list[tuple[Any, Any]] = []

    # ── Pass 1: overall ETS (reduce ALL dimensions) ───────────────────────────
    for var in all_variables:
        da_t = ds_target[var]
        da_p = ds_prediction[var]
        quantiles = compute_quantile_preserving(da_t, q_values, None)
        reduce_dims = list(da_t.dims)
        obs_events = da_t >= quantiles
        fcst_events = da_p >= quantiles
        bcm = BinaryContingencyManager(fcst_events=fcst_events, obs_events=obs_events)
        bcm = bcm.transform(reduce_dims=reduce_dims)
        c.print(f"Preparing ETS scores (overall) variable={var}...")
        lazy_tasks.append((("overall", var), bcm.equitable_threat_score()))

    # ── Pass 2: per-lead ETS (preserve lead_time during quantile / reduction) ─
    if do_per_lead and has_lead:
        for var in all_variables:
            da_t = ds_target[var]
            da_p = ds_prediction[var]
            # Quantiles have shape (quantile, lead_time); reduce over everything else.
            quantiles = compute_quantile_preserving(da_t, q_values, ["lead_time"])
            reduce_dims = [d for d in da_t.dims if d != "lead_time"]
            obs_events = da_t >= quantiles
            fcst_events = da_p >= quantiles
            bcm = BinaryContingencyManager(fcst_events=fcst_events, obs_events=obs_events)
            bcm = bcm.transform(reduce_dims=reduce_dims)
            c.print(f"Preparing ETS scores (per-lead) variable={var}...")
            lazy_tasks.append((("per_lead", var), bcm.equitable_threat_score()))

    # ── Pass 3: per-level ETS (quantiles computed once per var, level preserved)
    if do_per_level and has_level:
        for var in variables_3d:
            da_t = ds_target[var]
            da_p = ds_prediction[var]
            # Single quantile computation across all levels simultaneously.
            # Result shape: (quantile, level) — eliminates one full-data scan per level.
            quantiles_by_level = compute_quantile_preserving(da_t, q_values, ["level"])
            for lvl_raw in da_t["level"].values:
                lvl_val: Any = lvl_raw.item() if hasattr(lvl_raw, "item") else lvl_raw
                da_t_lvl = da_t.sel(level=lvl_val, drop=True)
                da_p_lvl = da_p.sel(level=lvl_val, drop=True)
                # q_lvl has shape (quantile,) — broadcasts over spatial/time dims.
                q_lvl = quantiles_by_level.sel(level=lvl_val, drop=True)
                obs_events_lvl = da_t_lvl >= q_lvl
                fcst_events_lvl = da_p_lvl >= q_lvl
                reduce_dims = list(da_t_lvl.dims)
                bcm = BinaryContingencyManager(
                    fcst_events=fcst_events_lvl, obs_events=obs_events_lvl
                )
                bcm = bcm.transform(reduce_dims=reduce_dims)
                c.print(f"Preparing ETS scores (per-level) variable={var} level={lvl_val}...")
                lazy_tasks.append((("per_level", var, lvl_val), bcm.equitable_threat_score()))

    if not lazy_tasks:
        return pd.DataFrame(), [], None

    # ── Single dask.compute for ALL passes ────────────────────────────────────
    n_overall = sum(1 for t, _ in lazy_tasks if t[0] == "overall")
    n_lead = sum(1 for t, _ in lazy_tasks if t[0] == "per_lead")
    n_level = sum(1 for t, _ in lazy_tasks if t[0] == "per_level")
    c.print(
        f"Computing all ETS passes ({len(lazy_tasks)} tasks: "
        f"{n_overall} overall, {n_lead} per-lead, {n_level} per-level) "
        f"in one dask graph..."
    )
    tags = [tag for tag, _ in lazy_tasks]
    lazies = [lazy for _, lazy in lazy_tasks]
    computed_raw = dask.compute(*lazies)
    results: dict[Any, Any] = dict(zip(tags, computed_raw, strict=False))

    # ── Unpack: overall_df ────────────────────────────────────────────────────
    metrics_dict: dict[str, dict[str, float]] = {}
    for var in all_variables:
        ets_score = results.get(("overall", var))
        if ets_score is None:
            continue
        metrics_dict[var] = {}
        vals = ets_score.values
        for i, threshold in enumerate(thresholds):
            metrics_dict[var][f"ETS {threshold}%"] = float(vals[i].item())
    overall_df = pd.DataFrame.from_dict(metrics_dict, orient="index")

    # ── Unpack: per_lead_rows ─────────────────────────────────────────────────
    per_lead_rows: list[dict[str, Any]] = []
    if do_per_lead and has_lead:
        leads_arr = ds_prediction["lead_time"].values
        is_td = np.issubdtype(np.asarray(leads_arr).dtype, np.timedelta64)
        lead_hours_list = [
            int(lt / np.timedelta64(1, "h")) if is_td else int(i) for i, lt in enumerate(leads_arr)
        ]
        for var in all_variables:
            ets_score = results.get(("per_lead", var))
            if ets_score is None:
                continue
            for li, hours in enumerate(lead_hours_list):
                flat: dict[str, Any] = {"lead_time_hours": float(hours), "variable": str(var)}
                for ti, threshold in enumerate(thresholds):
                    flat[f"ETS {threshold}%"] = float(
                        ets_score.isel(quantile=ti, lead_time=li, drop=True).values
                    )
                per_lead_rows.append(flat)

    # ── Unpack: per_level_df ──────────────────────────────────────────────────
    per_level_df: pd.DataFrame | None = None
    if do_per_level and has_level:
        level_rows: list[dict[str, Any]] = []
        for var in variables_3d:
            for lvl_raw in ds_target[var]["level"].values:
                lvl_val = lvl_raw.item() if hasattr(lvl_raw, "item") else lvl_raw
                ets_score = results.get(("per_level", var, lvl_val))
                if ets_score is None:
                    continue
                row: dict[str, Any] = {
                    "variable": var,
                    "level": int(lvl_val) if hasattr(lvl_val, "__int__") else lvl_val,
                }
                vals = ets_score.values
                for i, threshold in enumerate(thresholds):
                    row[f"ETS {threshold}%"] = float(
                        vals[i].item() if hasattr(vals[i], "item") else vals[i]
                    )
                level_rows.append(row)
        if level_rows:
            per_level_df = pd.DataFrame(level_rows)
            per_level_df = per_level_df.set_index("variable", drop=False)

    return overall_df, per_lead_rows, per_level_df


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
    ets_cfg = (metrics_cfg or {}).get("ets", {})
    mode = str((plotting_cfg or {}).get("output_mode", "plot")).lower()
    save_fig = mode in ("plot", "both")
    save_npz = mode in ("npz", "both")
    thresholds = ets_cfg.get("thresholds", [50, 60, 70, 80, 90])

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
        save_metric_by_lead_tables,
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

    # Determine what passes are needed up-front so everything can be computed
    # in a single dask.compute instead of three separate graph materialisations.
    _multi_lead = (
        ("lead_time" in ds_prediction.dims)
        and int(ds_prediction.sizes.get("lead_time", 0)) > 1
        and out_root is not None
        and members_indices is None
    )
    _do_per_level = report_per_level and out_root is not None and members_indices is None

    if members_indices is None:
        df, _per_lead_rows, _per_level_df = _compute_all_ets(
            ds_target,
            ds_prediction,
            thresholds,
            do_per_lead=_multi_lead,
            do_per_level=_do_per_level,
        )
    else:
        df = _calculate_ets_for_thresholds(ds_target, ds_prediction, thresholds)
        _per_lead_rows = []
        _per_level_df = None

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
            # Optional per-lead wide CSV when multi-lead policy provided.
            # Results were already computed by _compute_all_ets above.
            if _multi_lead:
                rows = _per_lead_rows
                if rows:
                    long_df = pd.DataFrame(rows)
                    long_df, wide_df = save_metric_by_lead_tables(
                        long_df=long_df,
                        section_output=section_output,
                        metric="ets_metrics",
                        init_time_range=init_range,
                        lead_time_range=lead_range,
                        ensemble=ens_token,
                        module="ets",
                    )

                    # Backward-compatible legacy filename used by tests/intercomparison.
                    out_wide = section_output / "ets_metrics_by_lead_wide.csv"
                    save_dataframe(wide_df, out_wide, index=False, module="ets")
                    # Optional: line plot of thresholds vs lead_time per variable
                    # Default to True so ETS thresholds per lead are visualized
                    do_plot = bool(ets_cfg.get("line_plot", True)) and (save_fig or save_npz)
                    if do_plot:
                        from ..helpers import build_output_filename

                        hours = wide_df["lead_time_hours"].values
                        hours_arr = np.asarray(hours, dtype=float)
                        # For each variable present in columns, collect threshold series
                        var_cols = [col for col in wide_df.columns if col != "lead_time_hours"]
                        # Expect columns like "<var>_ETS 50%"
                        by_var: dict[str, list[tuple[str, str]]] = {}
                        for col in var_cols:
                            if "_ETS " in col and col.endswith("%"):
                                v, rest = col.split("_ETS ", 1)
                                by_var.setdefault(v, []).append((col, rest))
                        for v, pairs in by_var.items():
                            pairs_sorted = sorted(pairs, key=lambda kv: int(kv[1].rstrip("%")))

                            # Use variable-specific colormap to distinguish thresholds
                            cmap_res = get_colormap_for_variable(v)
                            cmap = plt.get_cmap(cmap_res) if isinstance(cmap_res, str) else cmap_res

                            # Sample colors (avoiding very light start for sequential maps)
                            n_thresh = len(pairs_sorted)
                            colors = [cmap(i) for i in np.linspace(0.4, 1.0, n_thresh)]

                            if save_fig:
                                fig, ax = plt.subplots(figsize=(10, 6))
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
                                plt.close(fig)

                            if save_npz:
                                from numpy import savez as _savez

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
                                    rows_csv.append(
                                        {
                                            "variable": v,
                                            "lead_time_hours": float(h),
                                            "threshold": tlabel,
                                            "ETS": float(val),
                                        }
                                    )
                            save_dataframe(
                                pd.DataFrame(rows_csv), out_csv, index=False, module="ets"
                            )
            if _do_per_level:
                # Results were already computed by _compute_all_ets above.
                per_level_df = _per_level_df
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

                # Fuse overall + per-level into a single dask.compute for this member.
                df_m, _, per_level_m = _compute_all_ets(
                    ds_tgt_m,
                    ds_pred_m,
                    thresholds,
                    do_per_lead=False,
                    do_per_level=report_per_level,
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

                if report_per_level and per_level_m is not None:
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
