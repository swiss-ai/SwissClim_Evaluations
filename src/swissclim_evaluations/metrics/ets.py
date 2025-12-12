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


def _compute_ets_raw(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    thresholds: list[int],
    preserve_dims: list[str] | None = None,
) -> dict[str, xr.DataArray]:
    variables = list(ds_target.data_vars)
    lazy_results = {}
    q_values = [t / 100.0 for t in thresholds]

    for var in variables:
        da_target = ds_target[var]
        da_prediction = ds_prediction[var]

        # Determine dimensions to reduce for quantile calculation
        reduce_dims = list(da_target.dims)
        if preserve_dims:
            reduce_dims = [d for d in reduce_dims if d not in preserve_dims]

        # Compute quantiles lazily.
        quantiles = da_target.quantile(q_values, dim=reduce_dims, skipna=True)

        # Create events. Broadcasting will add 'quantile' dimension.
        obs_events = da_target >= quantiles
        fcst_events = da_prediction >= quantiles

        bcm = BinaryContingencyManager(fcst_events=fcst_events, obs_events=obs_events)
        bcm = bcm.transform(reduce_dims=reduce_dims)

        ets_score = bcm.equitable_threat_score()
        lazy_results[var] = ets_score

    # Compute all at once
    if not lazy_results:
        return {}

    keys = list(lazy_results.keys())
    values = list(lazy_results.values())
    computed_values = dask.compute(*values)

    return dict(zip(keys, computed_values, strict=False))


def _calculate_ets_for_thresholds(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    thresholds: list[int],
) -> pd.DataFrame:
    raw_results = _compute_ets_raw(ds_target, ds_prediction, thresholds)
    metrics_dict: dict[str, dict[str, float]] = {}

    for var, ets_score in raw_results.items():
        metrics_dict[var] = {}
        ets_values = ets_score.values
        for i, threshold in enumerate(thresholds):
            metrics_dict[var][f"ETS {threshold}%"] = float(ets_values[i].item())

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

    # Vectorized computation preserving 'level'
    raw_results = _compute_ets_raw(
        ds_target[variables_3d], ds_prediction[variables_3d], thresholds, preserve_dims=["level"]
    )

    dfs = []
    for var, ets_score in raw_results.items():
        # ets_score: (quantile, level)
        levels = ets_score.level.values
        for lvl in levels:
            row = {"variable": var, "level": int(lvl)}
            # Select this level
            ets_lvl = ets_score.sel(level=lvl).values
            for i, threshold in enumerate(thresholds):
                row[f"ETS {threshold}%"] = float(ets_lvl[i].item())
            dfs.append(row)

    if not dfs:
        return None

    df = pd.DataFrame(dfs)
    # Set index to variable to match original structure (though original had repeated index)
    df = df.set_index("variable", drop=False)
    return df


def run(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    out_root: Path | None = None,
    metrics_cfg: dict[str, Any] | None = None,
    ensemble_mode: str | None = None,
    lead_policy: Any | None = None,
) -> None:
    ets_cfg = (metrics_cfg or {}).get("ets", {})
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
        resolve_ensemble_mode,
        save_dataframe,
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

    df = _calculate_ets_for_thresholds(ds_target, ds_prediction, thresholds)

    print("Equitable Threat Score (targets vs predictions) — first 5 rows:")
    print(df.head())

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
            save_dataframe(df, out_csv)
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
                    df_i = _calculate_ets_for_thresholds(ds_t_i, ds_p_i, thresholds)
                    # Flatten columns with variable names to create a wide row
                    flat: dict[str, float] = {"lead_time_hours": float(hours)}
                    for var, row in df_i.iterrows():
                        for col, val in row.items():
                            flat[f"{var}_{col}"] = float(val)
                    rows.append(flat)
                if rows:
                    wide_df = pd.DataFrame(rows)
                    out_wide = section_output / "ets_metrics_by_lead_wide.csv"
                    save_dataframe(wide_df, out_wide, index=False)
                    # Optional: line plot of thresholds vs lead_time per variable
                    # Default to True so ETS thresholds per lead are visualized
                    do_plot = bool(ets_cfg.get("line_plot", True))
                    if do_plot:
                        from ..helpers import build_output_filename

                        hours = wide_df["lead_time_hours"].values
                        # For each variable present in columns, collect threshold series
                        var_cols = [c for c in wide_df.columns if c != "lead_time_hours"]
                        # Expect columns like "<var>_ETS 50%"
                        by_var: dict[str, list[tuple[str, str]]] = {}
                        for c in var_cols:
                            if "_ETS " in c and c.endswith("%"):
                                v, rest = c.split("_ETS ", 1)
                                by_var.setdefault(v, []).append((c, rest))
                        for v, pairs in by_var.items():
                            fig, ax = plt.subplots(figsize=(7, 3))
                            pairs_sorted = sorted(pairs, key=lambda kv: int(kv[1].rstrip("%")))
                            for col, tlabel in pairs_sorted:
                                ax.plot(
                                    hours, wide_df[col].values, marker="o", label=f"ETS {tlabel}"
                                )
                            ax.set_xlabel("lead_time (h)")
                            ax.set_ylabel("ETS")
                            ax.set_title(f"ETS thresholds vs lead_time — {v}")
                            ax.legend(ncols=min(3, len(pairs_sorted)), fontsize=8)
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
                            plt.savefig(out_png, bbox_inches="tight", dpi=150)
                            plt.close(fig)
                            # Save NPZ and CSV for line plot values
                            if not out_png.exists():
                                print(f"[ets] ERROR: File {out_png} was NOT created!")
                            else:
                                print(f"[ets] SUCCESS: File {out_png} created.")
                            # Save NPZ and CSV for line plot values
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
                            pd.DataFrame(rows_csv).to_csv(out_csv, index=False)
                            print(f"[ets] saved {out_png}")
                            print(f"[ets] saved {out_npz}")
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
                save_dataframe(df_m, out_csv_m, index_label="variable")
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
