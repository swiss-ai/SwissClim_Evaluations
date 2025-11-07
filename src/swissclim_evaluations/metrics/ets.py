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
            basic_cm = bcm.transform(reduce_dims="all")
            ets_score = basic_cm.equitable_threat_score()
            metrics_dict[var][f"ETS {threshold}%"] = float(ets_score.values)

    return pd.DataFrame.from_dict(metrics_dict, orient="index")


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
    reduce_ens_mean = True
    try:
        rem = ets_cfg.get("reduce_ensemble_mean")
        if rem is not None:
            reduce_ens_mean = bool(rem)
    except Exception:
        reduce_ens_mean = True
    aggregate_members_mean = bool(ets_cfg.get("aggregate_members_mean", True))

    had_ensemble_dim = ("ensemble" in ds_prediction.dims) or ("ensemble" in ds_target.dims)
    from ..helpers import ensemble_mode_to_token, resolve_ensemble_mode

    resolved_mode = resolve_ensemble_mode("ets", ensemble_mode, ds_target, ds_prediction)
    has_ens = "ensemble" in ds_prediction.dims
    if resolved_mode == "prob":
        raise ValueError("ensemble_mode=prob invalid for ETS metrics")
    members_indices: list[int] | None = None
    if resolved_mode == "members" and has_ens:
        members_indices = list(range(int(ds_prediction.sizes["ensemble"])))
    if resolved_mode == "none" and has_ens:
        resolved_mode = "mean"
    if resolved_mode == "mean" and has_ens and reduce_ens_mean:
        if "ensemble" in ds_prediction.dims:
            ds_prediction = ds_prediction.mean(dim="ensemble", keep_attrs=True)
        if "ensemble" in ds_target.dims:
            ds_target = ds_target.mean(dim="ensemble", keep_attrs=True)

    df = _calculate_ets_for_thresholds(ds_target, ds_prediction, thresholds)

    try:
        print("Equitable Threat Score (targets vs predictions) — first 5 rows:")
        print(df.head())
    except Exception:
        pass

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
            df.to_csv(out_csv)
            print(f"[ets] saved {out_csv}")
            # Optional per-lead wide CSV when multi-lead policy provided
            try:
                multi_lead = (
                    (lead_policy is not None)
                    and ("lead_time" in ds_prediction.dims)
                    and int(ds_prediction.sizes.get("lead_time", 0)) > 1
                    and getattr(lead_policy, "mode", "first") != "first"
                )
            except Exception:
                multi_lead = False
            if multi_lead:
                rows = []
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
                    wide_df.to_csv(out_wide, index=False)
                    print(f"[ets] saved {out_wide}")
                    # Optional: line plot of thresholds vs lead_time per variable
                    try:
                        # Default to True so ETS thresholds per lead are visualized
                        do_plot = bool(ets_cfg.get("line_plot", True))
                    except Exception:
                        do_plot = True
                    if do_plot:
                        import matplotlib.pyplot as _plt

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
                            fig, ax = _plt.subplots(figsize=(7, 3))
                            pairs_sorted = sorted(pairs, key=lambda kv: int(kv[1].rstrip("%")))
                            for col, tlabel in pairs_sorted:
                                ax.plot(
                                    hours, wide_df[col].values, marker="o", label=f"ETS {tlabel}"
                                )
                            ax.set_xlabel("lead_time (h)")
                            ax.set_ylabel("ETS")
                            ax.set_title(f"ETS thresholds vs lead_time — {v}")
                            ax.legend(ncols=min(3, len(pairs_sorted)), fontsize=8)
                            out_png = section_output / f"ets_line_{v}.png"
                            _plt.tight_layout()
                            _plt.savefig(out_png, bbox_inches="tight", dpi=150)
                            _plt.close(fig)
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
                df_m.to_csv(out_csv_m)
                print(f"[ets] saved {out_csv_m}")
                per_member_dfs.append(df_m)
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
                    pooled_df.to_csv(out_pool)
                    print(f"[ets] saved {out_pool}")
