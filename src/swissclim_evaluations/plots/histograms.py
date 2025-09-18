from __future__ import annotations

from pathlib import Path
from typing import Any

import dask.array as dsa
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..lead_time_policy import LeadTimePolicy


def _lat_bands() -> tuple[np.ndarray, int, int]:
    lat_bins = np.arange(-90, 91, 10)
    n_bands = len(lat_bins) - 1
    n_rows = n_bands // 2
    return lat_bins, n_bands, n_rows


def run(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    out_root: Path,
    plotting_cfg: dict[str, Any],
    lead_policy: LeadTimePolicy | None = None,
) -> None:
    mode = str(plotting_cfg.get("output_mode", "plot")).lower()
    save_fig = mode in ("plot", "both")
    save_npz = mode in ("npz", "both")
    dpi = int(plotting_cfg.get("dpi", 48))
    # Optional subsampling  to avoid loading full arrays.
    max_samples = plotting_cfg.get("histogram_max_samples", None)
    try:
        max_samples = int(max_samples) if max_samples is not None else None
        if max_samples <= 0:
            max_samples = None
    except Exception:
        max_samples = None
    base_seed = int(plotting_cfg.get("random_seed", 42))
    section_output = out_root / "histograms"

    # Select only genuine 2D variables (no 'level' dimension)
    variables_2d = [
        v for v in ds_target.data_vars if "level" not in ds_target[v].dims
    ]
    if not variables_2d:
        print("[histograms] No 2D variables found – skipping.")
        return
    print(f"[histograms] Processing {len(variables_2d)} 2D variables.")
    lat_bins, n_bands, n_rows = _lat_bands()

    multi_lead = (
        lead_policy is not None
        and "lead_time" in ds_prediction.dims
        and int(ds_prediction.lead_time.size) > 1
        and lead_policy.mode != "first"
    )
    if multi_lead:
        hours = (
            ds_prediction["lead_time"].values // np.timedelta64(1, "h")
        ).astype(int)
        panel_hours = lead_policy.select_panel_hours(list(map(int, hours)))
        hour_to_idx = {int(h): i for i, h in enumerate(hours)}
    else:
        panel_hours = [None]
        hour_to_idx = {}

    for i, variable_name in enumerate(variables_2d):
        print(f"[histograms] variable: {variable_name}")
        for lead_h in panel_hours:
            fig, axs = plt.subplots(
                n_rows, 2, figsize=(16, 3 * n_rows), dpi=dpi
            )
            lead_suffix = f"_lead{lead_h}" if lead_h is not None else ""
            # Collect combined NPZ data across all bands
            combined = {
                "neg_counts": [],
                "neg_bins": [],
                "pos_counts": [],
                "pos_bins": [],
                "neg_lat_min": [],
                "neg_lat_max": [],
                "pos_lat_min": [],
                "pos_lat_max": [],
            }

            # Helper functions are defined inside the loop scope so they are available each iteration
            def _choose_edges(
                da1: xr.DataArray, da2: xr.DataArray, bins: int = 1000
            ):
                try:
                    both = xr.concat([da1, da2], dim="_t")
                    q = both.quantile([0.001, 0.999], skipna=True).compute()
                    vmin = float(q.isel(quantile=0).item())
                    vmax = float(q.isel(quantile=1).item())
                    if (
                        not np.isfinite(vmin)
                        or not np.isfinite(vmax)
                        or vmin == vmax
                    ):
                        vmin, vmax = -1.0, 1.0
                except Exception:
                    vmin, vmax = -1.0, 1.0
                return np.linspace(vmin, vmax, bins + 1)

            def _dask_hist(da: xr.DataArray, edges: np.ndarray):
                data = getattr(da, "data", da)
                darr = dsa.asarray(data)
                darr = darr.ravel()
                darr = darr[~dsa.isnan(darr)]
                counts = dsa.histogram(darr, bins=np.asarray(edges))[0]
                return counts

            def _subsample_values(
                da: xr.DataArray, k: int, seed: int
            ) -> np.ndarray:
                if k is None:
                    arr = np.asarray(da.compute().values).ravel()
                    return arr[np.isfinite(arr)]
                size = int(getattr(da, "size", 0) or 0)
                if size == 0:
                    return np.array([], dtype=float)
                if size <= k:
                    arr = np.asarray(da.compute().values).ravel()
                    return arr[np.isfinite(arr)]
                dims = list(da.dims)
                nd = max(1, len(dims))
                frac = (k / float(size)) ** (1.0 / nd)
                rng = np.random.default_rng(seed)
                indexers: dict[str, np.ndarray] = {}
                for d in dims:
                    n = int(da.sizes.get(d, 1))
                    take = max(1, int(np.ceil(frac * n)))
                    take = min(take, n)
                    idx = rng.choice(n, size=take, replace=False)
                    idx.sort()
                    indexers[d] = idx
                sub = da.isel(**indexers)
                arr = np.asarray(sub.compute().values).ravel()
                return arr[np.isfinite(arr)]

            # Negative latitudes (right column)
            for j in range(n_bands // 2):
                # Negative latitude band (southern hemisphere)
                lat_max = lat_bins[j]
                lat_min = lat_bins[j + 1]
                da_true = ds_target[variable_name].sel(
                    latitude=slice(lat_min, lat_max)
                )
                da_pred = ds_prediction[variable_name].sel(
                    latitude=slice(lat_min, lat_max)
                )
                if max_samples is not None:
                    ds_seed = base_seed + (i + 1) * 1000 + (j + 1) * 10 + 1
                    ml_seed = ds_seed + 1
                    ds_sample = _subsample_values(da_true, max_samples, ds_seed)
                    ml_sample = _subsample_values(da_pred, max_samples, ml_seed)
                    if ds_sample.size == 0 or ml_sample.size == 0:
                        axs[j, 1].set_title(
                            f"Lat {lat_min}° to {lat_max}° (No data)"
                        )
                        continue
                    try:
                        both = np.concatenate([ds_sample, ml_sample])
                        qlow, qhigh = np.quantile(both, [0.001, 0.999])
                        if (
                            not np.isfinite(qlow)
                            or not np.isfinite(qhigh)
                            or qlow == qhigh
                        ):
                            qlow, qhigh = -1.0, 1.0
                    except Exception:
                        qlow, qhigh = -1.0, 1.0
                    edges = np.linspace(qlow, qhigh, 1001)
                    dsa_ds = dsa.from_array(
                        ds_sample, chunks=ds_sample.shape[0] // 4 or 1
                    )
                    dsa_ml = dsa.from_array(
                        ml_sample, chunks=ml_sample.shape[0] // 4 or 1
                    )
                    counts_ds = dsa.histogram(dsa_ds, bins=edges)[0].compute()
                    counts_ml = dsa.histogram(dsa_ml, bins=edges)[0].compute()
                else:
                    edges = _choose_edges(da_true, da_pred, bins=1000)
                    counts_ds = (
                        _dask_hist(da_true, edges).compute().astype(float)
                    )
                    counts_ml = (
                        _dask_hist(da_pred, edges).compute().astype(float)
                    )
                width = np.diff(edges)
                bin_area = (
                    counts_ds.sum() * width.mean()
                    if counts_ds.sum() > 0
                    else 1.0
                )
                counts_ds = counts_ds / bin_area
                bin_area_ml = (
                    counts_ml.sum() * width.mean()
                    if counts_ml.sum() > 0
                    else 1.0
                )
                counts_ml = counts_ml / bin_area_ml
                axs[j, 1].bar(
                    edges[:-1],
                    counts_ds,
                    width=width,
                    align="edge",
                    alpha=0.5,
                    color="skyblue",
                    label="Ground Truth",
                )
                axs[j, 1].bar(
                    edges[:-1],
                    counts_ml,
                    width=width,
                    align="edge",
                    alpha=0.5,
                    color="salmon",
                    label="Model Prediction",
                )
                axs[j, 1].set_title(f"Lat {lat_min}° to {lat_max}°")
                axs[j, 1].legend(loc="upper right")
                if save_npz:
                    combined["neg_counts"].append((counts_ds, counts_ml))
                    combined["neg_bins"].append(edges)
                    combined["neg_lat_min"].append(float(lat_min))
                    combined["neg_lat_max"].append(float(lat_max))

            # Positive latitudes (left column)
            for j in range(n_bands // 2):
                # Positive latitude band (northern hemisphere)
                idx = -(j + 1)
                lat_max = lat_bins[idx - 1]
                lat_min = lat_bins[idx]
                da_true = ds_target[variable_name].sel(
                    latitude=slice(lat_min, lat_max)
                )
                da_pred = ds_prediction[variable_name].sel(
                    latitude=slice(lat_min, lat_max)
                )
                if max_samples is not None:
                    ds_seed = base_seed + (i + 1) * 1000 + (j + 1) * 10 + 2
                    ml_seed = ds_seed + 1
                    ds_sample = _subsample_values(da_true, max_samples, ds_seed)
                    ml_sample = _subsample_values(da_pred, max_samples, ml_seed)
                    if ds_sample.size == 0 or ml_sample.size == 0:
                        axs[j, 0].set_title(
                            f"Lat {lat_min}° to {lat_max}° (No data)"
                        )
                        continue
                    try:
                        both = np.concatenate([ds_sample, ml_sample])
                        qlow, qhigh = np.quantile(both, [0.001, 0.999])
                        if (
                            not np.isfinite(qlow)
                            or not np.isfinite(qhigh)
                            or qlow == qhigh
                        ):
                            qlow, qhigh = -1.0, 1.0
                    except Exception:
                        qlow, qhigh = -1.0, 1.0
                    edges = np.linspace(qlow, qhigh, 1001)
                    dsa_ds = dsa.from_array(
                        ds_sample, chunks=ds_sample.shape[0] // 4 or 1
                    )
                    dsa_ml = dsa.from_array(
                        ml_sample, chunks=ml_sample.shape[0] // 4 or 1
                    )
                    counts_ds = dsa.histogram(dsa_ds, bins=edges)[0].compute()
                    counts_ml = dsa.histogram(dsa_ml, bins=edges)[0].compute()
                else:
                    edges = _choose_edges(da_true, da_pred, bins=1000)
                    counts_ds = (
                        _dask_hist(da_true, edges).compute().astype(float)
                    )
                    counts_ml = (
                        _dask_hist(da_pred, edges).compute().astype(float)
                    )
                width = np.diff(edges)
                bin_area = (
                    counts_ds.sum() * width.mean()
                    if counts_ds.sum() > 0
                    else 1.0
                )
                counts_ds = counts_ds / bin_area
                bin_area_ml = (
                    counts_ml.sum() * width.mean()
                    if counts_ml.sum() > 0
                    else 1.0
                )
                counts_ml = counts_ml / bin_area_ml
                axs[j, 0].bar(
                    edges[:-1],
                    counts_ds,
                    width=width,
                    align="edge",
                    alpha=0.5,
                    color="skyblue",
                    label="Ground Truth",
                )
                axs[j, 0].bar(
                    edges[:-1],
                    counts_ml,
                    width=width,
                    align="edge",
                    alpha=0.5,
                    color="salmon",
                    label="Model Prediction",
                )
                axs[j, 0].set_title(f"Lat {lat_min}° to {lat_max}°")
                axs[j, 0].legend(loc="upper right")
                if save_npz:
                    combined["pos_counts"].append((counts_ds, counts_ml))
                    combined["pos_bins"].append(edges)
                    combined["pos_lat_min"].append(float(lat_min))
                    combined["pos_lat_max"].append(float(lat_max))

            units = ds_target[variable_name].attrs.get("units", "")
            plt.suptitle(
                f"Distribution of {variable_name} ({units}) by latitude bands{(' - lead ' + str(lead_h) + 'h') if lead_h is not None else ''}",
                y=1.02,
            )
            plt.tight_layout()

            if save_fig:
                section_output.mkdir(parents=True, exist_ok=True)
                out_png = (
                    section_output
                    / f"{variable_name}_sfc_latbands{lead_suffix}.png"
                )
                plt.savefig(out_png, bbox_inches="tight", dpi=200)
                print(f"[histograms] saved {out_png}")
            if save_npz:

                def _stack_counts_bins(pairs):
                    counts = [p[0] for p in pairs]
                    bins = [p[1] for p in pairs]
                    return np.stack(counts, axis=0), np.stack(bins, axis=0)

                neg_counts, neg_bins_arr = (
                    _stack_counts_bins(combined["neg_counts"])
                    if combined["neg_counts"]
                    else (np.empty((0,)), np.empty((0,)))
                )
                pos_counts, pos_bins_arr = (
                    _stack_counts_bins(combined["pos_counts"])
                    if combined["pos_counts"]
                    else (np.empty((0,)), np.empty((0,)))
                )
                section_output.mkdir(parents=True, exist_ok=True)
                out_npz = (
                    section_output
                    / f"{variable_name}_sfc_latbands_combined{lead_suffix}.npz"
                )
                np.savez(
                    out_npz,
                    neg_counts=np.array(combined["neg_counts"], dtype=object),
                    neg_bins=np.array(combined["neg_bins"], dtype=object),
                    pos_counts=np.array(combined["pos_counts"], dtype=object),
                    pos_bins=np.array(combined["pos_bins"], dtype=object),
                    neg_lat_min=np.array(combined["neg_lat_min"]),
                    neg_lat_max=np.array(combined["neg_lat_max"]),
                    pos_lat_min=np.array(combined["pos_lat_min"]),
                    pos_lat_max=np.array(combined["pos_lat_max"]),
                    allow_pickle=True,
                )
                print(f"[histograms] saved {out_npz}")
            plt.close(fig)
