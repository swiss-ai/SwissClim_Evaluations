from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def calculate_energy_spectra(data: xr.DataArray):
    EARTH_RADIUS_KM = 6371.0
    wavenumber_list = []

    # Reduce optional ensemble dimension if present
    if "ensemble" in data.dims:
        data = data.mean(dim="ensemble")

    # Prefer init_time over time for temporal axis handling
    if "init_time" in data.dims:
        # If lead_time is present, collapse it (mean) so we can have a single temporal axis
        if "lead_time" in data.dims:
            data = data.mean(dim="lead_time")
        var_data = data.transpose("latitude", "longitude", "init_time")
        time_axis = -1
    elif "time" in data.dims:
        var_data = data.transpose("latitude", "longitude", "time")
        time_axis = -1
    else:
        var_data = data.transpose("latitude", "longitude")
        time_axis = None
    var_data = var_data.interpolate_na(dim="longitude")
    n_lon = var_data.longitude.size

    latitudes = np.deg2rad(var_data.latitude.values)
    cos_latitudes = np.cos(latitudes)
    circumference = 2 * np.pi * EARTH_RADIUS_KM * cos_latitudes
    dx = circumference / n_lon
    fft = np.fft.rfft(var_data, axis=1)

    for dx_item in dx:
        wavenumber = np.fft.rfftfreq(n_lon, d=dx_item) * n_lon * dx_item
        wavenumber_list.append(wavenumber)
    average_wavenumber = np.average(np.array(wavenumber_list), axis=0)

    power_spectrum = np.abs(fft) ** 2
    if time_axis is not None:
        power_spectrum_mean = np.average(
            power_spectrum.mean(axis=time_axis), axis=0, weights=cos_latitudes
        )
    else:
        power_spectrum_mean = np.average(
            power_spectrum, axis=0, weights=cos_latitudes
        )

    effective_resolution = 1 / (4 * dx.mean()) * (2 * np.pi * EARTH_RADIUS_KM)
    return (
        average_wavenumber[2:-2],
        power_spectrum_mean[2:-2],
        effective_resolution,
    )


def calculate_log_spectral_distance(
    spectrum1: np.ndarray, spectrum2: np.ndarray
) -> float:
    eps = 1e-10
    log_spec1 = np.log10(spectrum1 + eps)
    log_spec2 = np.log10(spectrum2 + eps)
    return float(np.sqrt(np.mean((log_spec1 - log_spec2) ** 2)))


def _plot_energy_spectra(
    ds: xr.Dataset,
    ds_ml: xr.Dataset,
    var: str,
    level: int | None,
    out_path: Path | None,
    dpi: int,
    save_plot_data: bool = False,
    save_figure: bool = True,
) -> float:
    if level is not None:
        var_data = ds[var].sel(level=level)
        var_data_ml = ds_ml[var].sel(level=level)
    else:
        var_data = ds[var]
        var_data_ml = ds_ml[var]

    wavenumber_ds, spectrum_ds, effective_resolution = calculate_energy_spectra(
        var_data
    )
    wavenumber_ml, spectrum_ml, _ = calculate_energy_spectra(var_data_ml)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(wavenumber_ds, spectrum_ds, color="skyblue", label="Ground Truth")
    ax.loglog(
        wavenumber_ml,
        spectrum_ml,
        color="salmon",
        label="Model Prediction",
    )

    k_min = min(wavenumber_ds.min(), wavenumber_ml.min())
    k_max = max(wavenumber_ds.max(), wavenumber_ml.max())
    k_range = np.logspace(np.log10(k_min + 1e-6), np.log10(k_max), 10)
    y_max = max(spectrum_ds.max(), spectrum_ml.max())
    ax.loglog(
        k_range,
        y_max * k_range ** (-3),
        "--",
        alpha=0.5,
        label="k⁻³",
        color="white",
    )
    ax.loglog(
        k_range,
        y_max * k_range ** (-5 / 3),
        ":",
        alpha=0.5,
        label="k⁻⁵ᐟ³",
        color="white",
    )
    ax.axvline(
        effective_resolution,
        color="goldenrod",
        linestyle="--",
        label="Effective Model Resolution",
    )

    lsd = calculate_log_spectral_distance(spectrum_ds, spectrum_ml)
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.5,
        0.05,
        f"LSD = {lsd:.4f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=props,
    )

    ax.set_xlabel("Wavenumber (num cycles around the Earth)")
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
        # Save underlying spectra and wavenumber arrays
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
            effective_resolution=effective_resolution,
            lsd=lsd,
            level=level if level is not None else -1,
            variable=var,
        )
        print(f"[energy_spectra] saved {npz_target}")
    plt.close(fig)
    return lsd


def run(
    ds: xr.Dataset,
    ds_ml: xr.Dataset,
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

    variables_2d = [v for v in ds.data_vars if "level" not in ds[v].dims]
    variables_3d = [v for v in ds.data_vars if "level" in ds[v].dims]
    levels = select_cfg.get("levels") or list(ds.coords.get("level", []))

    # 2D variables
    for var in variables_2d:
        print(f"[energy_spectra] 2D variable: {var}")
        _plot_energy_spectra(
            ds,
            ds_ml,
            var,
            None,
            (section_output / f"{var}_sfc_energy.png") if save_figure else None,
            dpi,
            save_plot_data,
            save_figure,
        )

    # 3D variables and LSD table
    lsd_data: dict[str, list[float]] = {var: [] for var in variables_3d}
    for var in variables_3d:
        print(f"[energy_spectra] 3D variable: {var}")
        for level in levels:
            lsd = _plot_energy_spectra(
                ds,
                ds_ml,
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
