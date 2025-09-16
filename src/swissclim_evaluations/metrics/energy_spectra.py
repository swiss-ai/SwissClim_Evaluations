from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def calculate_energy_spectra(
    data: xr.DataArray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute zonal energy spectra vs. longitudinal wavenumber.

    Steps:
    - Average over optional ensemble and lead_time
    - Select dims into (latitude, longitude[, time]) order
    - Interpolate NaNs along longitude
    - rFFT along longitude, magnitude^2
    - Latitude-weighted average (cos(lat)) and time average
    Returns (wavenumber, spectrum, k_effective) where k_effective≈n_lon/4.
    """
    # Reduce optional ensemble/lead_time dimensions if present
    if "ensemble" in data.dims:
        data = data.mean(dim="ensemble")
    if "lead_time" in data.dims:
        data = data.mean(dim="lead_time")

    # Drop/squeeze a singleton level dimension so transpose works
    if "level" in data.dims and int(data.sizes.get("level", 0)) == 1:
        data = data.isel(level=0, drop=True)

    # Choose a time-like dimension if available
    time_dim: str | None
    if "init_time" in data.dims:
        time_dim = "init_time"
    elif "time" in data.dims:
        time_dim = "time"
    else:
        time_dim = None

    # Reorder dims to [latitude, longitude, time?]
    order = [d for d in ("latitude", "longitude") if d in data.dims]
    if time_dim:
        order.append(time_dim)
    var_data = data.transpose(*order)

    # Rechunk longitude to single chunk before interpolate (dask safety)
    try:
        if hasattr(var_data.data, "chunks"):
            var_data = var_data.chunk({"longitude": -1})
    except Exception:
        pass

    # Fill NaNs along longitude to avoid FFT propagation of NaNs
    try:
        if "longitude" in var_data.dims:
            var_data = var_data.interpolate_na(dim="longitude")
    except Exception:
        pass

    # Prepare arrays
    n_lon = int(var_data.sizes.get("longitude", 0))
    if n_lon == 0:
        return np.array([]), np.array([]), 0.0

    arr = var_data.values  # shape: (n_lat, n_lon[, n_time])
    if arr.ndim == 2:
        arr = arr[:, :, None]  # add time axis

    n_lat, _, n_time = arr.shape

    # rFFT along longitude (axis=1)
    fft = np.fft.rfft(arr, axis=1)
    power = np.abs(fft) ** 2  # (n_lat, n_k, n_time)

    # Latitude weighting (cosine of radians)
    lat_vals = var_data.coords.get("latitude", None)
    if lat_vals is not None:
        cosw = np.cos(np.deg2rad(np.asarray(lat_vals)))
        cosw = np.clip(cosw, 1e-6, None)
        cosw = cosw.reshape((n_lat, 1, 1))
        power_w = power * cosw
        lat_weight = cosw
    else:
        power_w = power
        lat_weight = 1.0

    # Average over latitude and time
    power_lat_mean = power_w.sum(axis=0) / (
        lat_weight.sum(axis=0) if isinstance(lat_weight, np.ndarray) else n_lat
    )
    spectrum = power_lat_mean.mean(axis=-1)  # (n_k,)

    # Wavenumber in cycles around Earth
    n_k = spectrum.shape[0]
    wavenumber = np.arange(n_k, dtype=float)

    # Simple effective resolution proxy
    k_effective = float(n_lon / 4.0)

    # Drop first couple bins to avoid DC/very-low-k artifacts when plotting; caller may slice
    return wavenumber, spectrum, k_effective


def calculate_log_spectral_distance(
    spectrum1: np.ndarray, spectrum2: np.ndarray
) -> float:
    eps = 1e-10
    log_spec1 = np.log10(spectrum1 + eps)
    log_spec2 = np.log10(spectrum2 + eps)
    return float(np.sqrt(np.mean((log_spec1 - log_spec2) ** 2)))


def _plot_energy_spectra(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    var: str,
    level: int | None,
    out_path: Path | None,
    dpi: int,
    save_plot_data: bool = False,
    save_figure: bool = True,
) -> float:
    # Select variable and optional level
    if level is not None:
        da_target = ds_target[var].sel(level=level)
        da_prediction = ds_prediction[var].sel(level=level)
    else:
        da_target = ds_target[var]
        da_prediction = ds_prediction[var]

    wavenumber_ds, spectrum_ds, k_eff = calculate_energy_spectra(da_target)
    wavenumber_ml, spectrum_ml, _ = calculate_energy_spectra(da_prediction)

    # Align to common length if necessary (should be identical)
    n = min(spectrum_ds.shape[0], spectrum_ml.shape[0])
    wavenumber_ds = wavenumber_ds[:n]
    wavenumber_ml = wavenumber_ml[:n]
    spectrum_ds = spectrum_ds[:n]
    spectrum_ml = spectrum_ml[:n]

    lsd = calculate_log_spectral_distance(spectrum_ds, spectrum_ml)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi * 2)
    ax.loglog(
        wavenumber_ds[2:-2],
        spectrum_ds[2:-2],
        color="skyblue",
        label="Ground Truth",
    )
    ax.loglog(
        wavenumber_ml[2:-2],
        spectrum_ml[2:-2],
        color="salmon",
        label="Model Prediction",
    )

    if n > 0:
        k_min = max(1e-6, float(min(wavenumber_ds[2], wavenumber_ml[2])))
        k_max = float(max(wavenumber_ds[-3], wavenumber_ml[-3]))
        k_range = np.logspace(np.log10(k_min), np.log10(k_max), 10)
        y_max = float(max(spectrum_ds.max(), spectrum_ml.max()))
        ax.loglog(
            k_range, y_max * k_range ** (-3), "--", alpha=0.3, label="k⁻³ (ref)"
        )
        ax.loglog(
            k_range,
            y_max * k_range ** (-5 / 3),
            ":",
            alpha=0.5,
            label="k⁻⁵ᐟ³ (ref)",
        )
        ax.axvline(
            k_eff,
            linestyle="--",
            color="gray",
            alpha=0.5,
            label="k_eff≈n_lon/4",
        )

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.02,
        0.98,
        f"LSD = {lsd:.4f}",
        transform=ax.transAxes,
        va="top",
        bbox=props,
    )

    ax.set_xlabel("Longitudinal Wavenumber (cycles/Earth)")
    ax.set_ylabel("Energy Density")
    title = f"Energy Spectra Comparison for {var}" + (
        f" at Level {level}" if level is not None else ""
    )
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()

    if save_figure and out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        print(f"[energy_spectra] saved {out_path}")
    if save_plot_data:
        npz_target = (
            out_path.with_suffix(".npz")
            if out_path is not None
            else Path("spectra_temp.npz")
        )
        np.savez(
            npz_target,
            wavenumber_ds=wavenumber_ds,
            spectrum_ds=spectrum_ds,
            wavenumber_ml=wavenumber_ml,
            spectrum_ml=spectrum_ml,
            k_effective=k_eff,
            lsd=lsd,
            level=level if level is not None else -1,
            variable=var,
        )
        print(f"[energy_spectra] saved {npz_target}")
    plt.close(fig)
    return lsd


def run(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    out_root: Path,
    plotting_cfg: dict[str, Any],
    select_cfg: dict[str, Any],
) -> None:
    mode = str(plotting_cfg.get("output_mode", "plot")).lower()
    dpi = int(plotting_cfg.get("dpi", 48))
    # Always export NPZ data; figures are controlled by output_mode
    save_plot_data = True
    save_figure = mode in ("plot", "both")
    section_output = out_root / "energy_spectra"

    if "level" in ds_target.dims and int(ds_target.level.size) > 1:
        variables_3d = [
            v for v in ds_target.data_vars if "level" in ds_target[v].dims
        ]
        variables_2d = [v for v in ds_target.data_vars if v not in variables_3d]
        levels = select_cfg.get("levels") or list(ds_target.level.values)
    else:
        variables_3d = []
        variables_2d = list(ds_target.data_vars)
        levels = []

    # 2D variables (no level); write a simple CSV summarizing LSD for completeness
    lsd_rows = []
    for var in variables_2d:
        print(f"[energy_spectra] 2D variable: {var}")
        lsd = _plot_energy_spectra(
            ds_target,
            ds_prediction,
            var,
            None,
            (section_output / f"{var}_sfc_energy.png") if save_figure else None,
            dpi,
            save_plot_data,
            save_figure,
        )
        lsd_rows.append({"variable": var, "lsd": lsd})

    if lsd_rows:
        section_output.mkdir(parents=True, exist_ok=True)
        df2d = pd.DataFrame(lsd_rows).set_index("variable")
        out_csv_2d = section_output / "lsd_2d_metrics.csv"
        df2d.to_csv(out_csv_2d)
        print(f"[energy_spectra] saved {out_csv_2d}")

    # 3D variables and LSD table
    lsd_data: dict[str, list[float]] = {var: [] for var in variables_3d}
    for var in variables_3d:
        print(f"[energy_spectra] 3D variable: {var}")
        for level in levels:
            lsd = _plot_energy_spectra(
                ds_target,
                ds_prediction,
                var,
                int(level),
                (section_output / f"{var}_{level}hPa_energy.png")
                if save_figure
                else None,
                dpi,
                save_plot_data,
                save_figure,
            )
            lsd_data[var].append(lsd)

    if variables_3d:
        df = pd.DataFrame(lsd_data, index=levels)
        df.index.name = "Height Level"
        section_output.mkdir(parents=True, exist_ok=True)
        out_csv = section_output / "lsd_metrics.csv"
        df.to_csv(out_csv)
        print(f"[energy_spectra] saved {out_csv}")
