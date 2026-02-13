from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D
from scores.functions import create_latitude_weights

from .. import console as c
from ..dask_utils import compute_jobs, resolve_dynamic_chunk_size
from ..helpers import (
    COLOR_GROUND_TRUTH,
    COLOR_MODEL_PREDICTION,
    build_output_filename,
    ensemble_mode_to_token,
    extract_date_from_dataset,
    format_level_label,
    format_variable_name,
    get_variable_units,
    resolve_ensemble_mode,
    save_data,
    save_dataframe,
    save_figure as save_fig_helper,
)

EARTH_RADIUS_KM = 6371.0
EARTH_CIRCUMFERENCE_KM = 2 * np.pi * EARTH_RADIUS_KM

# Standard wavebands in km (wavelength ranges)
WAVE_BANDS: list[dict[str, float | str]] = [
    {"name": "planetary", "min_km": 5000.0, "max_km": 20000.0},
    {"name": "synoptic", "min_km": 1000.0, "max_km": 5000.0},
    {"name": "upper_mesoscale", "min_km": 250.0, "max_km": 1000.0},
    {"name": "lower_mesoscale", "min_km": 10.0, "max_km": 250.0},
]


def calculate_energy_spectra(
    da_var: xr.DataArray,
    average_dims: Sequence[str] | None = None,
    weights: xr.DataArray | None = None,
) -> xr.DataArray:
    """Compute zonal energy spectra retaining time structure.

    Notes on spectral coordinates
    -----------------------------
    - wavenumber: cycles per km (NOT angular; no 2π factor). Computed using
      Earth circumference in km.
    - wavelength: km (1 / wavenumber).
    - wavenumber_m: cycles per m (wavenumber / 1000).
    Returned power units: (original units)^2 after latitude weighting. Averaging
    strategy:
        If ``average_dims`` is provided we now (v2) compute the power spectrum
        FIRST for each member along those dimensions and *then* average the
        (latitude–weighted) power spectra. This implements the mean( spectrum(x)
        ) instead of spectrum( mean(x) ), which avoids loss of variance prior to
        quadratic power computation.
    Time / lead dimensions are never implicitly averaged here.
    """

    if "longitude" not in da_var.dims:
        raise ValueError("longitude dimension required for energy spectra")

    n_lon = da_var.sizes["longitude"]
    if n_lon < 4:
        raise ValueError("Need at least 4 longitudes for spectral analysis")

    # Dask-parallelized rFFT along longitude using apply_ufunc
    da_fft = xr.apply_ufunc(
        np.fft.rfft,
        da_var,
        input_core_dims=[["longitude"]],
        output_core_dims=[["wavenumber"]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[np.complex128],
        dask_gufunc_kwargs={"output_sizes": {"wavenumber": n_lon // 2 + 1}},
    )

    # physical spacing along longitude
    dx_km = EARTH_CIRCUMFERENCE_KM / n_lon
    da_fft["wavenumber"] = ("wavenumber", np.fft.rfftfreq(n_lon, d=dx_km))
    da_fft["wavenumber"].attrs.update(
        {
            "units": "cycles km^-1",
            "long_name": "zonal wavenumber (cycles per km)",
            "note": "Not angular wavenumber; no 2π factor",
        }
    )

    # Drop the zero wavenumber (mean) component for log scaling clarity
    da_fft = da_fft.isel(wavenumber=slice(1, None))
    # Assign wavelength coordinate using raw numpy values to avoid ambiguity errors
    da_fft["wavelength"] = ("wavenumber", 1.0 / da_fft["wavenumber"].values)
    da_fft["wavelength"].attrs.update(
        {
            "units": "km",
            "long_name": "zonal wavelength",
        }
    )
    da_fft["wavenumber_m"] = (
        "wavenumber",
        da_fft["wavenumber"].values / 1000.0,
    )
    da_fft["wavenumber_m"].attrs.update(
        {
            "units": "cycles m^-1",
            "long_name": "zonal wavenumber (cycles per meter)",
            "note": "wavenumber / 1000",
        }
    )

    # Power spectrum
    da_power = (da_fft * np.conjugate(da_fft)).real
    # Add units for power spectrum if input had units
    in_units = get_variable_units(da_var, str(da_var.name))
    if in_units:
        da_power.attrs["units"] = f"{in_units}^2"
    da_power.attrs["long_name"] = "Latitude-weighted zonal power spectrum"

    # Latitude weighting (cos φ) – retains any non-latitude dims (e.g. ensemble)
    if "latitude" not in da_power.coords:
        raise ValueError(
            "calculate_energy_spectra requires a 'latitude' coordinate in the input DataArray."
        )

    if weights is None:
        weights = create_latitude_weights(da_power["latitude"])
        # Fix for floating point errors giving slightly (~-10^-8) negative weights at poles
        weights = weights.clip(min=0.0)

    da_power = da_power.weighted(weights).mean(dim="latitude")

    # Post-spectrum averaging over requested dims (e.g., ensemble)
    if average_dims:
        post_avg_dims = [d for d in average_dims if d in da_power.dims]
        if post_avg_dims:
            da_power = da_power.mean(dim=post_avg_dims)
    return da_power


def calculate_log_spectral_distance(spectrum1: np.ndarray, spectrum2: np.ndarray) -> float:
    eps = 1e-10
    log_spec1 = np.log10(spectrum1 + eps)
    log_spec2 = np.log10(spectrum2 + eps)
    return float(np.sqrt(np.mean((log_spec1 - log_spec2) ** 2)))


def _compute_lsd_da(spec_target: xr.DataArray, spec_pred: xr.DataArray) -> xr.DataArray:
    """Vectorized Log Spectral Distance along wavenumber (no averaging over time dims)."""
    eps = 1e-10
    log_t = np.log10(spec_target + eps)
    log_p = np.log10(spec_pred + eps)
    diff2 = (log_t - log_p) ** 2
    lsd_da = np.sqrt(diff2.mean(dim="wavenumber"))
    lsd_da.name = "lsd"
    return lsd_da


def _compute_banded_lsd_da(
    spec_target: xr.DataArray,
    spec_pred: xr.DataArray,
    wave_bands: list[dict] | None = None,
) -> xr.DataArray:
    """Compute LSD per waveband, returning a DataArray with a new 'band' dimension.

    The input spectra must include a 'wavenumber' coordinate in cycles/km. We
    convert wavelength bands [min_km, max_km] to wavenumber ranges and compute
    LSD over the subset of wavenumbers within each band. If a band has no
    overlapping spectral points (e.g., due to grid/Nyquist limits), its LSD
    will be NaN.
    """
    bands = wave_bands or WAVE_BANDS
    # Determine available wavenumber range
    kvals = spec_target["wavenumber"].values
    # Build per-band LSD DataArrays
    lsd_list: list[xr.DataArray] = []
    band_names: list[str] = []
    for band in bands:
        name = str(band["name"])  # e.g., 'planetary'
        wl_min = float(band["min_km"])  # km (lower wavelength bound)
        wl_max = float(band["max_km"])  # km (upper wavelength bound)
        # Convert to wavenumber range (cycles/km)
        # Larger wavelengths -> smaller wavenumber
        k_low = 1.0 / wl_max  # inclusive lower bound in k-space
        k_high = 1.0 / wl_min  # inclusive upper bound in k-space
        mask = (kvals >= k_low) & (kvals <= k_high)
        if not np.any(mask):
            # Create a NaN DA with proper dims (time-like dims retained).
            # Select a single wavenumber index to drop the wavenumber dimension.
            template = spec_target.isel(wavenumber=0)
            lsd_empty = template.astype(float) * np.nan
            lsd_empty = lsd_empty.rename("lsd")
            lsd_list.append(lsd_empty)
            band_names.append(name)
            continue
        st = spec_target.isel(wavenumber=np.where(mask)[0])
        sp = spec_pred.isel(wavenumber=np.where(mask)[0])
        lsd_band = _compute_lsd_da(st, sp)
        lsd_band = lsd_band.rename("lsd")
        lsd_list.append(lsd_band)
        band_names.append(name)

    # Stack into a new 'band' dimension
    lsd_banded = xr.concat(lsd_list, dim="band")
    lsd_banded = lsd_banded.assign_coords(band=("band", band_names))
    return lsd_banded


def _compute_spectra_pair(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    var: str,
    level: int | None = None,
    *,
    reduce_ensemble: bool = True,
    weights: xr.DataArray | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute and align spectra for target and prediction once.

    Returns (spectrum_target, spectrum_prediction) with identical coordinates.
    """
    if level is not None:
        da_target = ds_target[var].sel(level=level, drop=True)
        da_prediction = ds_prediction[var].sel(level=level, drop=True)
    else:
        da_target = ds_target[var]
        da_prediction = ds_prediction[var]

    spec_t = calculate_energy_spectra(
        da_target,
        average_dims=(
            ["ensemble"] if (reduce_ensemble and ("ensemble" in da_target.dims)) else None
        ),
        weights=weights,
    )
    spec_p = calculate_energy_spectra(
        da_prediction,
        average_dims=(
            ["ensemble"] if (reduce_ensemble and ("ensemble" in da_prediction.dims)) else None
        ),
        weights=weights,
    )
    spec_t, spec_p = xr.align(spec_t, spec_p, join="inner")
    return spec_t, spec_p


def _plot_single_spectrum(
    wavenumber: np.ndarray,
    arr_target: np.ndarray,
    arr_pred: np.ndarray,
    lsd_val: float,
    var: str,
    level: int | None,
    init_label: str,
    lead_label: str,
    out_path: Path | None,
    dpi: int,
    save_plot_data: bool,
    save_figure: bool,
    date_str: str = "",
):
    """Create one spectrum comparison figure & optional NPZ."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi * 2)
    ax.loglog(wavenumber, arr_target, color=COLOR_GROUND_TRUTH, label="Target")
    ax.loglog(wavenumber, arr_pred, color=COLOR_MODEL_PREDICTION, label="Prediction")

    props = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5}  # dict literal (ruff C408)
    ax.text(
        0.5,
        0.05,
        f"LSD = {lsd_val:.4f}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        bbox=props,
    )
    ax.set_xlabel("Zonal Wavenumber (cycles/km)")
    ax.set_ylabel("Energy Density (weighted)")
    # --- Top axis wavelength (km) -------------------------------------------------
    # Select a physically-informed set of wavelength candidates (km) → convert
    # to wavenumber positions (cycles/km) and keep those that fall inside the
    # plotted wavenumber span. This yields stable labels (e.g., 40k, 20k, 10k,
    # 5k, 2k, 1k, 500, 200, 100, 50, 20, 10, 5, 2, 1 ...).
    k_min = float(np.nanmin(wavenumber[wavenumber > 0]))
    k_max = float(np.nanmax(wavenumber))
    if k_min <= 0 or not np.isfinite(k_min):  # safety
        return
    ax.set_xlim(k_min, k_max)

    # Add golden dotted line at 4*dx cutoff (k_max / 2)
    k_cutoff = k_max / 2.0
    ax.axvline(k_cutoff, color="gold", linestyle=":", linewidth=2, alpha=0.8, label="4dx Cutoff")

    add_wavelength_axis(ax, k_min, k_max)

    lev_part = format_level_label(level)
    # Simplify title to just show init date if available, matching other plots
    if date_str:
        date_suffix = date_str
    else:
        date_suffix = ""
        if init_label and init_label != "noInit":
            date_suffix = f" ({init_label}"
            if lead_label and lead_label != "noLead":
                date_suffix += f" +{lead_label}"
            date_suffix += ")"

    ax.set_title(
        f"Energy Spectra — {format_variable_name(var)}{lev_part}{date_suffix}",
        pad=24,
        fontsize=10,
    )
    ax.legend(fontsize=10)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    if save_figure and out_path is not None:
        save_fig_helper(fig, out_path, module="energy_spectra")

        # Generate Ratio Plot
        # We derive the filename by replacing the metric name
        if "energy_spectrum" in out_path.name:
            ratio_name = out_path.name.replace("energy_spectrum", "energy_ratio")
            ratio_path = out_path.parent / ratio_name

            fig_r, ax_r = plt.subplots(figsize=(10, 6), dpi=dpi * 2)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = arr_pred / arr_target

            # Apply 4dx cutoff (k_max / 2) similar to intercompare
            k_max_model = np.nanmax(wavenumber)
            if np.isfinite(k_max_model) and k_max_model > 0:
                k_cutoff = k_max_model / 2.0
                mask = wavenumber <= k_cutoff
                ax_r.semilogx(wavenumber[mask], ratio[mask], color=COLOR_MODEL_PREDICTION, lw=1.5)
            else:
                ax_r.semilogx(wavenumber, ratio, color=COLOR_MODEL_PREDICTION, lw=1.5)

            ax_r.axhline(1.0, color="gray", linestyle="--", alpha=0.7)

            ax_r.set_xlabel("Zonal Wavenumber (cycles/km)")
            ax_r.set_ylabel("Ratio (Prediction / Target)")
            ax_r.set_xlim(k_min, k_max)
            # Auto-scale y but keep 1.0 centered or visible
            # ax_r.set_ylim(0.5, 2.0) # Optional: fixed range

            add_wavelength_axis(ax_r, k_min, k_max)

            ax_r.set_title(
                f"Energy Ratio — {format_variable_name(var)}{lev_part}{date_suffix}",
                pad=24,
                fontsize=10,
            )
            ax_r.grid(True, which="both", ls="--", alpha=0.5)
            plt.tight_layout()
            save_fig_helper(fig_r, ratio_path, module="energy_spectra")
            plt.close(fig_r)

    if save_plot_data and out_path is not None:
        np_path = out_path.with_suffix(".npz")
        save_data(
            np_path,
            wavenumber=wavenumber,
            spectrum_target=arr_target,
            spectrum_prediction=arr_pred,
            lsd=lsd_val,
            variable=var,
            level=-1 if level is None else level,
            init_time=init_label,
            lead_time=lead_label,
            module="energy_spectra",
        )
    plt.close(fig)


def _plot_energy_spectra(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    var: str,
    level: int | None,
    out_path: Path | None,
    dpi: int,
    save_plot_data: bool = False,
    save_figure: bool = True,
    override_ensemble_token: str | None = None,
    chunk_size_cfg: int | str | None = None,
    weights: xr.DataArray | None = None,
) -> xr.DataArray:
    """Generate ONE spectrum & LSD per (init_time, lead_time) combination (no temporal averaging).

    Returns
    -------
    xr.DataArray
        LSD values with remaining time-like dims (init_time, lead_time, ...).
    """
    spectrum_target, spectrum_pred = _compute_spectra_pair(
        ds_target, ds_prediction, var, level, reduce_ensemble=True, weights=weights
    )

    # Compute LSD per time slice (vectorized)
    lsd_da = _compute_lsd_da(spectrum_target, spectrum_pred)
    # Ensemble token: mark ensmean if we averaged an ensemble dimension
    # Infer ensemble token: if either ds had ensemble originally, we reduced over it
    had_ensemble = ("ensemble" in getattr(ds_target.get(var), "dims", {})) or (
        "ensemble" in getattr(ds_prediction.get(var), "dims", {})
    )
    ens_token = override_ensemble_token or ("mean" if had_ensemble else None)

    # Determine time-like dims (exclude wavenumber)
    time_dims = [d for d in spectrum_target.dims if d != "wavenumber"]
    # Create per-time outputs

    if not time_dims:  # no time dims → single spectrum
        wn = spectrum_target["wavenumber"].values
        arr_t = spectrum_target.values
        arr_p = spectrum_pred.values
        init_label = "none"
        if "init_time" in spectrum_target.coords:
            try:
                val = spectrum_target["init_time"].values
                init_np = np.datetime64(val).astype("datetime64[h]")
                init_label = np.datetime_as_string(init_np, unit="h").replace(":", "")
            except Exception:
                pass

        lead_label = "none"
        if "lead_time" in spectrum_target.coords:
            try:
                val = spectrum_target["lead_time"].values
                hours = int(np.timedelta64(val) / np.timedelta64(1, "h"))
                lead_label = f"{hours:03d}h"
            except Exception:
                pass

        date_str = extract_date_from_dataset(spectrum_target)

        base_dir = out_path.parent if out_path else Path(".")  # fallback
        fname = build_output_filename(
            metric="energy_spectrum",
            variable=var,
            level=f"{level}hPa" if level is not None else "surface",
            qualifier=None,
            init_time_range=None,
            lead_time_range=None,
            ensemble=ens_token,
            ext="png",
        )
        _plot_single_spectrum(
            wn,
            arr_t,
            arr_p,
            float(lsd_da.values),
            var,
            level,
            init_label,
            lead_label,
            base_dir / fname if save_figure else None,
            dpi,
            save_plot_data,
            save_figure,
            date_str=date_str,
        )
        return lsd_da

    # Create stacked iterator
    stacked_target = spectrum_target.stack(__time__=time_dims)
    stacked_pred = spectrum_pred.stack(__time__=time_dims)
    stacked_lsd = lsd_da.stack(__time__=time_dims)
    coords_df = stacked_target.__time__.to_index()  # MultiIndex with labels

    # Output directory – flattened layout consistent with other modules.
    section_output = out_path.parent if out_path else Path(".")
    section_output.mkdir(parents=True, exist_ok=True)

    # Collect jobs
    jobs = []
    for idx, key in enumerate(coords_df):
        sel_kwargs = {str(dim): key[i] for i, dim in enumerate(time_dims)}

        # Robust init_time formatting (ensure numpy datetime64)
        init_raw = None
        if "init_time" in sel_kwargs:
            init_raw = sel_kwargs["init_time"]
        elif "init_time" in spectrum_target.coords:
            init_raw = spectrum_target["init_time"].values

        if init_raw is not None:
            init_label = "noinit"
            try:
                # Try direct conversion first (handles numpy scalars/arrays and Timestamps usually)
                init_np = np.datetime64(init_raw).astype("datetime64[h]")
                init_label = np.datetime_as_string(init_np, unit="h")
            except Exception:
                try:
                    # pandas Timestamp path or other objects
                    val = init_raw.item() if hasattr(init_raw, "item") else init_raw
                    if hasattr(val, "to_datetime64"):
                        init_np = np.datetime64(val.to_datetime64())
                        init_label = np.datetime_as_string(
                            init_np.astype("datetime64[h]"), unit="h"
                        )
                    else:
                        init_label = str(val)
                except Exception:
                    init_label = str(init_raw)
            # sanitize for filename
            init_label = init_label.replace(":", "")
        else:
            init_label = "noinit"

        # Robust lead_time (hours) formatting
        if "lead_time" in sel_kwargs:
            lt_raw = sel_kwargs["lead_time"]
            hours = 0
            try:
                lt_td = np.timedelta64(lt_raw)
                hours = int(lt_td / np.timedelta64(1, "h"))
            except Exception:
                try:
                    # If already numeric-like
                    hours = int(lt_raw)
                except Exception:
                    hours = 0

            # Skip generation if multiple lead times are present (except spectrograms which are
            # handled separately)
            if "lead_time" in ds_prediction.dims and ds_prediction.sizes["lead_time"] > 1:
                continue

            lead_label = f"{hours:03d}h"
        else:
            lead_label = "noLead"

        fname = build_output_filename(
            metric="energy_spectrum",
            variable=var,
            level=f"{level}hPa" if level is not None else None,
            qualifier=None,
            init_time_range=(init_label, init_label),
            lead_time_range=(lead_label, lead_label),
            ensemble=ens_token,
            ext="png",
        )
        # Provide a path if either figure OR plot data requested
        target_path = section_output / fname if (save_figure or save_plot_data) else None

        t_slice = stacked_target.isel(__time__=idx)
        date_str = extract_date_from_dataset(t_slice)

        jobs.append(
            {
                "idx": idx,
                "init_label": init_label,
                "lead_label": lead_label,
                "target_path": target_path,
                "t_lazy": t_slice,
                "p_lazy": stacked_pred.isel(__time__=idx),
                "lsd_lazy": stacked_lsd.isel(__time__=idx),
                "date_str": date_str,
            }
        )

    # Compute all
    n_points = int(spectrum_target.size + spectrum_pred.size)
    num_vars = 1
    dynamic_chunk = resolve_dynamic_chunk_size(
        {"chunk_size": chunk_size_cfg},
        n_points=n_points,
        num_vars=num_vars,
    )

    wn = spectrum_target["wavenumber"].values

    def _process_batch(batch_jobs: list[dict[str, Any]]):
        for job in batch_jobs:
            if "arr_t" not in job:
                continue

            arr_t = job["arr_t"]
            arr_p = job["arr_p"]
            lsd_val = float(job["lsd_val"])

            _plot_single_spectrum(
                wn,
                np.asarray(arr_t),
                np.asarray(arr_p),
                lsd_val,
                var,
                level,
                job["init_label"],
                job["lead_label"],
                job["target_path"],
                dpi,
                save_plot_data,
                save_figure,
                date_str=job.get("date_str", ""),
            )
            # Clear large arrays from memory
            job["arr_t"] = None
            job["arr_p"] = None

    compute_jobs(
        jobs,
        key_map={"t_lazy": "arr_t", "p_lazy": "arr_p", "lsd_lazy": "lsd_val"},
        chunk_size=dynamic_chunk,
        desc="Computing energy spectra",
        batch_callback=_process_batch,
    )

    return lsd_da  # shape: time dims only (no wavenumber)


def _plot_averaged_spectra(
    spec_t: xr.DataArray,
    spec_p: xr.DataArray,
    var: str,
    level: int | None,
    out_dir: Path,
    init_range: tuple[str, str] | None,
    lead_range: tuple[str, str] | None,
    ens_token: str | None,
    dpi: int,
    save_plot_data: bool,
    save_figure: bool,
) -> None:
    """Plot spectra averaged over init_time.

    This generates a single plot (per lead time) where the power spectra have been
    averaged over all initialization times. The filename will reflect the full
    initialization range (e.g. init2020010100-2020123123).
    """
    if "init_time" not in spec_t.dims or spec_t.sizes["init_time"] <= 1:
        return

    # Average over init_time
    # Note: docstring says "Time / lead dimensions are never implicitly averaged".
    # Here we explicitly average over init_time to get the mean spectrum.
    spec_t_avg = spec_t.mean(dim="init_time")
    spec_p_avg = spec_p.mean(dim="init_time")

    # Compute LSD of the averaged spectra
    lsd_avg_da = _compute_lsd_da(spec_t_avg, spec_p_avg)
    wn = spec_t_avg["wavenumber"].values

    # Determine iteration dims (likely lead_time)
    if "lead_time" in spec_t_avg.dims:
        leads = spec_t_avg["lead_time"].values
        for lt in leads:
            st = spec_t_avg.sel(lead_time=lt)
            sp = spec_p_avg.sel(lead_time=lt)
            lsd = float(lsd_avg_da.sel(lead_time=lt).values)

            # Format lead label
            try:
                hours = int(pd.Timedelta(lt).total_seconds() / 3600)
                lead_label = f"{hours:03d}h"
            except Exception:
                # Handle potential numpy/other types
                val = lt.item() if hasattr(lt, "item") else lt
                hours = int(val / 3600 / 1e9) if isinstance(val, int) else 0
                lead_label = f"{hours:03d}h"

            # Construct filename with full init range
            fname = build_output_filename(
                metric="energy_spectrum",
                variable=var,
                level=f"{level}hPa" if level is not None else None,
                qualifier=None,
                init_time_range=init_range,
                lead_time_range=(lead_label, lead_label),
                ensemble=ens_token,
                ext="png",
            )
            out_path = out_dir / fname
            init_label_str = f"{init_range[0]}-{init_range[1]}" if init_range else "mean"

            _plot_single_spectrum(
                wn,
                st.values,
                sp.values,
                lsd,
                var,
                level,
                init_label=init_label_str,
                lead_label=lead_label,
                out_path=out_path,
                dpi=dpi,
                save_plot_data=save_plot_data,
                save_figure=save_figure,
            )
    else:
        # No lead time dim (scalar or missing)
        lead_label = "noLead"
        fname = build_output_filename(
            metric="energy_spectrum",
            variable=var,
            level=f"{level}hPa" if level is not None else None,
            qualifier=None,
            init_time_range=init_range,
            lead_time_range=lead_range,  # Default lead range from dataset
            ensemble=ens_token,
            ext="png",
        )
        out_path = out_dir / fname
        init_label_str = f"{init_range[0]}-{init_range[1]}" if init_range else "mean"

        _plot_single_spectrum(
            wn,
            spec_t_avg.values,
            spec_p_avg.values,
            float(lsd_avg_da.values),
            var,
            level,
            init_label=init_label_str,
            lead_label=lead_label,
            out_path=out_path,
            dpi=dpi,
            save_plot_data=save_plot_data,
            save_figure=save_figure,
        )


def run(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    out_root: Path,
    plotting_cfg: dict[str, Any],
    select_cfg: dict[str, Any],
    ensemble_mode: str | None = None,
    cfg: dict[str, Any] | None = None,
    performance_cfg: dict[str, Any] | None = None,
) -> None:
    """Compute basic 2D energy spectra metrics and write an averaged LSD CSV.

    This minimal implementation restores a valid, lint-clean `run()` after a
    bad merge. It computes per-variable spectra for 2D variables (no 'level'),
    evaluates the Log Spectral Distance (LSD) across remaining time-like dims,
    and writes a single summary CSV using standardized filenames.
    """
    section_output = out_root / "energy_spectra"
    section_output.mkdir(parents=True, exist_ok=True)

    perf_cfg = performance_cfg or {}
    chunk_size_cfg = perf_cfg.get("chunk_size")

    # Helper to place x-ticks exactly at selected lead hours (downsample if many)
    def _apply_lead_ticks(ax: Any, hours: np.ndarray) -> None:
        hrs = np.asarray(hours).astype(int)
        n = hrs.size
        if n == 0:
            return
        if n <= 16:
            ticks = hrs
        else:
            stride = max(1, int(np.ceil(n / 12)))
            ticks = hrs[::stride]
            if ticks[-1] != hrs[-1]:
                ticks = np.concatenate([ticks, [hrs[-1]]])
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(t)) for t in ticks])

    # Extract config
    es_cfg = (cfg or {}).get("metrics", {}).get("energy_spectra", {})
    report_per_level = bool(es_cfg.get("report_per_level", True))

    # Preserve full datasets for metrics
    ds_target_full = ds_target
    ds_prediction_full = ds_prediction

    resolved_mode = resolve_ensemble_mode(
        "energy_spectra", ensemble_mode, ds_target_full, ds_prediction_full
    )
    has_ens = "ensemble" in ds_target_full.dims or "ensemble" in ds_prediction_full.dims
    ensemble_members: list[int | None] = [None]
    if resolved_mode == "mean" and has_ens:
        if "ensemble" in ds_target_full.dims:
            ds_target_full = ds_target_full.mean(dim="ensemble")
        if "ensemble" in ds_prediction_full.dims:
            ds_prediction_full = ds_prediction_full.mean(dim="ensemble")

        ens_token = ensemble_mode_to_token("mean")
    elif resolved_mode == "pooled" and has_ens:
        ens_token = ensemble_mode_to_token("pooled")
    elif resolved_mode == "members" and has_ens:
        # Preserve ensemble members; token will indicate members mode
        n_ens = ds_prediction_full.sizes.get("ensemble", ds_target_full.sizes.get("ensemble", 0))
        ensemble_members = list(range(n_ens))
        ens_token = None
    else:
        ens_token = None

    resolved = resolved_mode
    tgt = ds_target_full
    pred = ds_prediction_full

    # Helpers to extract time range tokens
    def _extract_init_range(ds: xr.Dataset) -> tuple[str, str] | None:
        if "init_time" not in ds.dims:
            return None
        vals = ds["init_time"].values
        if getattr(vals, "size", 0) == 0:
            return None
        start = np.datetime64(vals.min()).astype("datetime64[h]")
        end = np.datetime64(vals.max()).astype("datetime64[h]")

        def _fmt(x: np.datetime64) -> str:
            return (
                np.datetime_as_string(x, unit="h")
                .replace("-", "")
                .replace(":", "")
                .replace("T", "")
            )

        return (_fmt(start), _fmt(end))

    def _extract_lead_range(ds: xr.Dataset) -> tuple[str, str] | None:
        if "lead_time" not in ds.dims:
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

    init_range = _extract_init_range(pred)
    lead_range = _extract_lead_range(pred)

    # Detect multi-lead context: when more than one lead_time is present after selection
    is_multi_lead = ("lead_time" in pred.dims) and int(
        getattr(pred, "sizes", {}).get("lead_time", 0)
    ) > 1

    # Select 2D variables (exclude true 3D with 'level' dim)
    variables_2d = [v for v in tgt.data_vars if "level" not in tgt[v].dims]

    # Prepare plotting subset (single init_time) to avoid figure explosion
    time_index = 0
    plot_dt = (plotting_cfg or {}).get("plot_datetime")

    if "init_time" in tgt.dims and tgt.sizes["init_time"] > 0:
        if plot_dt is not None:
            try:
                target_dt = np.datetime64(plot_dt)
                matches = np.where(tgt.init_time.values == target_dt)[0]
                if matches.size > 0:
                    time_index = int(matches[0])
                else:
                    c.print(
                        f"[energy_spectra] Warning: plot_datetime {plot_dt} not found. "
                        "Using first init_time."
                    )
                    time_index = 0
            except Exception as e:
                c.print(
                    f"[energy_spectra] Warning: Error selecting plot_datetime {plot_dt}: {e}. "
                    "Using first init_time."
                )
                time_index = 0
        else:
            time_index = 0

    tgt_plot = tgt.isel(init_time=time_index) if "init_time" in tgt.dims else tgt
    pred_plot = pred.isel(init_time=time_index) if "init_time" in pred.dims else pred
    # Basic schema validation: require longitude for spectra
    has_lon_any = ("longitude" in tgt.dims) or any(
        ("longitude" in tgt[v].dims) for v in variables_2d
    )
    if not has_lon_any:
        raise ValueError("longitude dimension required for energy spectra")

    # Compute weights once
    weights = None
    if "latitude" in tgt.dims:
        weights = create_latitude_weights(tgt.latitude)
        weights = weights.clip(min=0.0)

    # In multi-lead mode, suppress non-spectrogram artifacts (CSV summaries, 1D exports)
    # Initialize holders used in both branches
    lsd_long_rows: list[dict[str, float | str]] = []
    lsd_banded_long_rows: list[dict[str, float | str]] = []

    mode = str((plotting_cfg or {}).get("output_mode", "plot")).lower()
    save_npz = mode in ("npz", "both")
    save_plot = mode in ("plot", "both")
    dpi = int((plotting_cfg or {}).get("dpi", 200))

    if not is_multi_lead:
        summary_rows: list[dict[str, Any]] = []
        lsd_lead_time_rows: list[dict[str, Any]] = []
        lsd_plot_rows: list[dict[str, Any]] = []
        lsd_banded_rows: list[dict[str, Any]] = []

        # Collect LSD-by-lead for optional line plots/CSVs
        for var in variables_2d:
            c.print(f"[energy_spectra] Processing variable: {var}")
            # Preserve ensemble for pooled/members modes
            # Only reduce when resolved == 'mean'
            spec_t, spec_p = _compute_spectra_pair(
                tgt, pred, str(var), None, reduce_ensemble=(resolved == "mean"), weights=weights
            )
            lsd_da = _compute_lsd_da(spec_t, spec_p)
            lsd_mean = float(lsd_da.mean().values)
            summary_rows.append({"variable": str(var), "lsd_mean": lsd_mean})

            # Per Lead Time (averaged over init_time)
            if "lead_time" in lsd_da.dims:
                red_dims = [d for d in lsd_da.dims if d != "lead_time"]
                lsd_lead = lsd_da.mean(dim=red_dims) if red_dims else lsd_da
                # Compute once
                lsd_lead = lsd_lead.compute()
                for t in lsd_lead["lead_time"].values:
                    val = float(lsd_lead.sel(lead_time=t).values)
                    hours = int(np.timedelta64(t) / np.timedelta64(1, "h"))
                    lsd_lead_time_rows.append(
                        {
                            "variable": str(var),
                            "lead_time": f"{hours:03d}h",
                            "lsd_mean": val,
                        }
                    )

            if save_plot or save_npz:
                _plot_averaged_spectra(
                    spec_t,
                    spec_p,
                    str(var),
                    None,
                    section_output,
                    init_range,
                    lead_range,
                    ens_token,
                    dpi,
                    save_plot_data=save_npz,
                    save_figure=save_plot,
                )

            # Plot Datetime specific LSD
            if "init_time" in lsd_da.dims:
                # Calculate LSD for the specific plot datetime only
                # Use time_index which was computed earlier for plot_dt
                lsd_plot_da = lsd_da.isel(init_time=time_index)
                lsd_plot_val = float(lsd_plot_da.mean().compute())

                # Get the actual datetime string
                t_val = lsd_da["init_time"].isel(init_time=time_index).values
                try:
                    t_str = np.datetime_as_string(t_val, unit="h")
                except Exception:
                    t_str = str(t_val)

                lsd_plot_rows.append(
                    {
                        "variable": str(var),
                        "init_time": t_str,
                        "lsd_mean": lsd_plot_val,
                    }
                )

            # Compute banded LSD metrics (2D)
            lsd_bands_da = _compute_banded_lsd_da(spec_t, spec_p)
            # Average over time dims
            red_dims = [d for d in lsd_bands_da.dims if d != "band"]
            lsd_bands_mean = lsd_bands_da.mean(dim=red_dims) if red_dims else lsd_bands_da

            for bname in lsd_bands_mean["band"].values:
                val = float(lsd_bands_mean.sel(band=bname).values)
                lsd_banded_rows.append(
                    {
                        "variable": str(var),
                        "band": str(bname),
                        "lsd_mean": val,
                    }
                )

        if summary_rows:
            df_summary = pd.DataFrame(summary_rows).set_index("variable")
            out_csv = section_output / build_output_filename(
                metric="energy_ratios",
                variable=None,
                level=None,
                qualifier="averaged",
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token,
                ext="csv",
            )
            df_summary.to_csv(out_csv)
            c.print(f"[energy_spectra] saved {out_csv}")

            if lsd_lead_time_rows:
                df_lead = pd.DataFrame(lsd_lead_time_rows)
                out_csv_lead = section_output / build_output_filename(
                    metric="energy_ratios",
                    variable=None,
                    level=None,
                    qualifier="lead_time",
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token,
                    ext="csv",
                )
                df_lead.to_csv(out_csv_lead, index=False)
                c.print(f"[energy_spectra] saved {out_csv_lead}")

            if lsd_banded_rows:
                df_banded = pd.DataFrame(lsd_banded_rows)
                out_csv_banded = section_output / build_output_filename(
                    metric="energy_ratios_bands",
                    variable=None,
                    level=None,
                    qualifier="averaged",
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token,
                    ext="csv",
                )
                df_banded.to_csv(out_csv_banded, index=False)
                c.print(f"[energy_spectra] saved {out_csv_banded}")

    # Generate plots and NPZ for 2D variables
    if save_plot or save_npz:
        for var in variables_2d:
            for ens_idx in ensemble_members:
                t_p = tgt_plot
                p_p = pred_plot
                curr_ens_token = ens_token

                if ens_idx is not None:
                    if "ensemble" in p_p.dims:
                        p_p = p_p.isel(ensemble=ens_idx)
                    if "ensemble" in t_p.dims:
                        if t_p.sizes["ensemble"] == 1:
                            t_p = t_p.isel(ensemble=0)
                        else:
                            t_p = t_p.isel(ensemble=ens_idx)
                    curr_ens_token = ensemble_mode_to_token("members", ens_idx)

                _plot_energy_spectra(
                    t_p,
                    p_p,
                    str(var),
                    None,
                    section_output / "placeholder",
                    dpi=dpi,
                    save_plot_data=save_npz,
                    save_figure=save_plot,
                    override_ensemble_token=curr_ens_token,
                    chunk_size_cfg=chunk_size_cfg,
                    weights=weights,
                )

    # 3D variables (per-level)
    variables_3d = []
    if "level" in tgt.dims:
        variables_3d = [v for v in tgt.data_vars if "level" in tgt[v].dims]

    if variables_3d:
        lsd_3d_rows: list[dict[str, Any]] = []
        lsd_banded_3d_rows: list[dict[str, Any]] = []
        lsd_3d_averaged_rows: list[dict[str, Any]] = []  # New for averaged
        lsd_3d_lead_time_rows: list[dict[str, Any]] = []
        lsd_3d_plot_rows: list[dict[str, Any]] = []

        levels = tgt["level"].values
        for var in variables_3d:
            # Compute spectra for ALL levels at once (vectorized)
            spec_t, spec_p = _compute_spectra_pair(
                tgt,
                pred,
                str(var),
                level=None,
                reduce_ensemble=(resolved == "mean"),
                weights=weights,
            )

            # 1. Global LSD per level
            lsd_da = _compute_lsd_da(spec_t, spec_p)
            # Average over all dims except level
            red_dims = [d for d in lsd_da.dims if d != "level"]
            lsd_means_per_level = lsd_da.mean(dim=red_dims).compute()

            # 2. Banded LSD per level
            lsd_bands_da = _compute_banded_lsd_da(spec_t, spec_p)
            red_dims_band = [d for d in lsd_bands_da.dims if d not in ["level", "band"]]
            lsd_bands_mean_per_level = (
                lsd_bands_da.mean(dim=red_dims_band).compute() if red_dims_band else lsd_bands_da
            )

            var_lsd_values = []
            for lvl in levels:
                val = float(lsd_means_per_level.sel(level=lvl).item())
                lsd_3d_rows.append({"variable": str(var), "level": int(lvl), "lsd_mean": val})
                var_lsd_values.append(val)

                for bname in lsd_bands_mean_per_level["band"].values:
                    val_band = float(lsd_bands_mean_per_level.sel(level=lvl, band=bname).item())
                    lsd_banded_3d_rows.append(
                        {
                            "variable": str(var),
                            "level": int(lvl),
                            "band": str(bname),
                            "lsd_mean": val_band,
                        }
                    )

            # Average over levels for the "averaged" summary
            lsd_3d_averaged_rows.append(
                {
                    "variable": str(var),
                    "lsd_mean": float(np.mean(var_lsd_values)),
                }
            )

            # Per Lead Time (averaged over init_time and level)
            if "lead_time" in lsd_da.dims:
                red_dims = [d for d in lsd_da.dims if d != "lead_time"]
                lsd_lead = lsd_da.mean(dim=red_dims) if red_dims else lsd_da
                lsd_lead = lsd_lead.compute()
                for t in lsd_lead["lead_time"].values:
                    val = float(lsd_lead.sel(lead_time=t).values)
                    hours = int(np.timedelta64(t) / np.timedelta64(1, "h"))
                    lsd_3d_lead_time_rows.append(
                        {
                            "variable": str(var),
                            "lead_time": f"{hours:03d}h",
                            "lsd_mean": val,
                        }
                    )

            if save_plot or save_npz:
                for lvl in levels:
                    _plot_averaged_spectra(
                        spec_t.sel(level=lvl),
                        spec_p.sel(level=lvl),
                        str(var),
                        int(lvl),
                        section_output,
                        init_range,
                        lead_range,
                        ens_token,
                        dpi,
                        save_plot_data=save_npz,
                        save_figure=save_plot,
                    )

            # Plot Datetime specific LSD
            if "init_time" in lsd_da.dims:
                # Calculate LSD for the specific plot datetime only
                lsd_plot_da = lsd_da.isel(init_time=time_index)
                # Average over other dims (like lead_time, level)
                lsd_plot_val = float(lsd_plot_da.mean().compute())

                t_val = lsd_da["init_time"].isel(init_time=time_index).values
                try:
                    t_str = np.datetime_as_string(t_val, unit="h")
                except Exception:
                    t_str = str(t_val)

                lsd_3d_plot_rows.append(
                    {
                        "variable": str(var),
                        "init_time": t_str,
                        "lsd_mean": lsd_plot_val,
                    }
                )

        if report_per_level and lsd_3d_rows:
            import pandas as _pd

            df_3d = _pd.DataFrame(lsd_3d_rows)
            out_csv_3d = section_output / build_output_filename(
                metric="energy_ratios_3d",
                variable=None,
                level=None,
                qualifier="per_level",
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token,
                ext="csv",
            )
            df_3d.to_csv(out_csv_3d, index=False)
            c.print(f"[energy_spectra] saved {out_csv_3d}")

        if report_per_level and lsd_banded_3d_rows:
            import pandas as _pd

            df_banded_3d = _pd.DataFrame(lsd_banded_3d_rows)
            out_csv_banded_3d = section_output / build_output_filename(
                metric="energy_ratios_bands_3d",
                variable=None,
                level=None,
                qualifier="per_level",
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token,
                ext="csv",
            )
            save_dataframe(df_banded_3d, out_csv_banded_3d, index=False, module="energy_spectra")

        # Always save averaged 3D metrics
        if lsd_3d_averaged_rows:
            import pandas as _pd

            df_3d_avg = _pd.DataFrame(lsd_3d_averaged_rows)
            out_csv_3d_avg = section_output / build_output_filename(
                metric="energy_ratios_3d",
                variable=None,
                level=None,
                qualifier="averaged",
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token,
                ext="csv",
            )
            save_dataframe(df_3d_avg, out_csv_3d_avg, index=False, module="energy_spectra")

        if lsd_3d_lead_time_rows:
            import pandas as _pd

            df_3d_lead = _pd.DataFrame(lsd_3d_lead_time_rows)
            out_csv_3d_lead = section_output / build_output_filename(
                metric="energy_ratios_3d",
                variable=None,
                level=None,
                qualifier="lead_time",
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token,
                ext="csv",
            )
            save_dataframe(df_3d_lead, out_csv_3d_lead, index=False, module="energy_spectra")

        # Generate plots and NPZ for 3D variables per level
        if save_plot or save_npz:
            for var in variables_3d:
                for lvl in levels:
                    for ens_idx in ensemble_members:
                        t_p = tgt_plot
                        p_p = pred_plot
                        curr_ens_token = ens_token

                        if ens_idx is not None:
                            if "ensemble" in p_p.dims:
                                p_p = p_p.isel(ensemble=ens_idx)
                            if "ensemble" in t_p.dims:
                                if t_p.sizes["ensemble"] == 1:
                                    t_p = t_p.isel(ensemble=0)
                                else:
                                    t_p = t_p.isel(ensemble=ens_idx)
                            curr_ens_token = ensemble_mode_to_token("members", ens_idx)

                        _plot_energy_spectra(
                            t_p,
                            p_p,
                            str(var),
                            int(lvl),
                            section_output / "placeholder",
                            dpi=dpi,
                            save_plot_data=save_npz,
                            save_figure=save_plot,
                            override_ensemble_token=curr_ens_token,
                            weights=weights,
                        )

    # Optional: per-member NPZ spectrum exports when requested
    mode = str((plotting_cfg or {}).get("output_mode", "plot")).lower()
    save_npz = mode in ("npz", "both")
    if (
        (not is_multi_lead)
        and save_npz
        and (resolved == "members")
        and ("ensemble" in pred_plot.dims)
    ):
        dpi = int((plotting_cfg or {}).get("dpi", 48))

        for var in variables_2d:
            for mi in range(int(pred_plot.sizes["ensemble"])):
                token = ensemble_mode_to_token("members", mi)
                ds_t_m = tgt_plot.isel(ensemble=mi) if "ensemble" in tgt_plot.dims else tgt_plot
                ds_p_m = pred_plot.isel(ensemble=mi) if "ensemble" in pred_plot.dims else pred_plot
                _plot_energy_spectra(
                    ds_t_m,
                    ds_p_m,
                    str(var),
                    None,
                    section_output / "_anchor.png",
                    dpi=dpi,
                    save_plot_data=True,
                    save_figure=False,
                    override_ensemble_token=token,
                )

    # Optional: spectrogram over lead_time (x) and wavenumber (y) with energy color
    do_spec = bool((plotting_cfg or {}).get("energy_spectra_spectrogram", False))
    # In multi-lead mode, always produce spectrograms regardless of plotting toggle
    if is_multi_lead:
        do_spec = True
    if (
        do_spec
        and ("lead_time" in pred.dims)
        and int(getattr(pred, "sizes", {}).get("lead_time", 0)) > 1
    ):
        # Create per-variable spectrograms for target and model
        def _reduce_time_like(da: xr.DataArray) -> xr.DataArray:
            # Average over init_time/time dimensions, keep lead_time and wavenumber
            red_dims = [d for d in ["time", "init_time"] if d in da.dims]
            return da.mean(dim=red_dims, skipna=True) if red_dims else da

        dpi = int((plotting_cfg or {}).get("dpi", 48))
        for var in variables_2d:
            # Compute spectra (already averaged over ensemble inside calculate_energy_spectra call)
            spec_t, spec_p = _compute_spectra_pair(
                tgt, pred, str(var), None, reduce_ensemble=(resolved == "mean"), weights=weights
            )
            # Reduce over init_time/time, retaining lead_time and wavenumber
            spec_t2 = _reduce_time_like(spec_t)
            spec_p2 = _reduce_time_like(spec_p)
            # Ensure both share identical coords
            spec_t2, spec_p2 = xr.align(spec_t2, spec_p2, join="inner")
            # If any extra dims remain (e.g., ensemble when not reducing),
            # collapse them by mean so we can plot 2D (lead_time × wavenumber)
            extra_dims_t = [d for d in spec_t2.dims if d not in ("lead_time", "wavenumber")]
            if extra_dims_t:
                spec_t2 = spec_t2.mean(dim=extra_dims_t, skipna=True)
            extra_dims_p = [d for d in spec_p2.dims if d not in ("lead_time", "wavenumber")]
            if extra_dims_p:
                spec_p2 = spec_p2.mean(dim=extra_dims_p, skipna=True)
            # Extract coordinates
            if "lead_time" not in spec_t2.dims:
                continue
            leads = spec_t2["lead_time"].values
            # Convert lead times to hours for x-axis
            try:
                x_hours = np.array(
                    [int(np.timedelta64(lt) / np.timedelta64(1, "h")) for lt in leads]
                )
            except Exception:
                x_hours = np.arange(len(leads))
            kvals = spec_t2["wavenumber"].values
            # Prepare arrays with shape (n_leads, n_k)
            Zt = np.asarray(spec_t2.transpose("lead_time", "wavenumber").values)
            Zp = np.asarray(spec_p2.transpose("lead_time", "wavenumber").values)

            def _plot_spec_img(
                Z: np.ndarray,
                qualifier: str,
                *,
                _x_hours=x_hours,
                _kvals=kvals,
                _var=str(var),
            ) -> None:
                fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi * 2)
                # log10 color scale with small epsilon to avoid -inf
                eps = 1e-10
                im = ax.pcolormesh(
                    _x_hours,
                    _kvals,
                    np.log10(Z.T + eps),
                    shading="auto",
                    cmap="viridis",
                )
                ax.set_xlabel("lead_time (h)")
                ax.set_ylabel("wavenumber (cycles/km)")
                _apply_lead_ticks(ax, _x_hours)
                cb = fig.colorbar(im, ax=ax, orientation="vertical")
                cb.set_label("log10 energy")
                out_png = section_output / build_output_filename(
                    metric="energy_spectra_per_lead",
                    variable=_var,
                    level=None,
                    qualifier=qualifier,
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token,
                    ext="png",
                )
                c.print(f"[energy_spectra] Saved {out_png}")
                plt.tight_layout()
                save_fig_helper(fig, out_png)
                plt.close(fig)

            _plot_spec_img(Zt, "target")
            _plot_spec_img(Zp, "model")

            # Difference spectrogram (model - target) in log10 energy space
            eps = 1e-10
            logZt = np.log10(Zt + eps)
            logZp = np.log10(Zp + eps)
            Zdiff = (logZp - logZt).T  # shape (k, leads) for plotting convenience
            fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi * 2)
            vmax = np.nanmax(np.abs(Zdiff))
            im = ax.pcolormesh(
                x_hours,
                kvals,
                Zdiff,
                shading="auto",
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
            )
            ax.set_xlabel("lead_time (h)")
            ax.set_ylabel("wavenumber (cycles/km)")
            _apply_lead_ticks(ax, x_hours)
            cb = fig.colorbar(im, ax=ax, orientation="vertical")
            cb.set_label("Δ log10 energy (model - target)")
            out_png = section_output / build_output_filename(
                metric="energy_spectra_per_lead",
                variable=str(var),
                level=None,
                qualifier="difference",
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token,
                ext="png",
            )
            plt.tight_layout()
            save_fig_helper(fig, out_png, module="energy_spectra")
            plt.close(fig)

            # Also compute LSD versus lead_time and emit one plot per variable
            lsd_da = _compute_lsd_da(spec_t2, spec_p2)  # sqrt(mean((log Et - log Em)^2) over k)
            # Reduce any non-lead dims if present
            red_dims = [d for d in lsd_da.dims if d != "lead_time"]
            if red_dims:
                lsd_da = lsd_da.mean(dim=red_dims, skipna=True)
            lsd_vals = np.asarray(lsd_da.values).ravel()
            for h, v in zip(x_hours.tolist(), lsd_vals.tolist(), strict=False):
                lsd_long_rows.append(
                    {
                        "lead_time_hours": float(h),
                        "variable": str(var),
                        "LSD": float(v),
                    }
                )
            fig, ax = plt.subplots(figsize=(6, 3), dpi=dpi * 2)
            ax.plot(x_hours, lsd_vals, marker="o")
            ax.set_xlabel("Lead Time [h]")
            ax.set_ylabel("LSD")
            display_var = str(var).split(".", 1)[1] if "." in str(var) else str(var)
            ax.set_title(
                f"{format_variable_name(display_var)} — LSD (Global) vs Lead Time", fontsize=10
            )
            out_png = section_output / build_output_filename(
                metric="energy_ratios_line_per_lead",
                variable=str(var),
                level=None,
                qualifier=None,
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token,
                ext="png",
            )
            plt.tight_layout()
            save_fig_helper(fig, out_png, module="energy_spectra")
            plt.close(fig)

            # Compute Banded LSD versus lead_time
            lsd_bands_da = _compute_banded_lsd_da(spec_t2, spec_p2)
            # Reduce any non-lead/band dims if present
            red_dims_band = [d for d in lsd_bands_da.dims if d not in ["lead_time", "band"]]
            if red_dims_band:
                lsd_bands_da = lsd_bands_da.mean(dim=red_dims_band, skipna=True)

            for bname in lsd_bands_da["band"].values:
                band_da = lsd_bands_da.sel(band=bname)
                lsd_vals_band = np.asarray(band_da.values).ravel()

                # Store in rows
                for h, v in zip(x_hours.tolist(), lsd_vals_band.tolist(), strict=False):
                    lsd_banded_long_rows.append(
                        {
                            "lead_time_hours": float(h),
                            "variable": str(var),
                            "band": str(bname),
                            "LSD": float(v),
                        }
                    )

                # Plot
                fig, ax = plt.subplots(figsize=(6, 3), dpi=dpi * 2)
                ax.plot(x_hours, lsd_vals_band, marker="o")
                ax.set_xlabel("Lead Time [h]")
                ax.set_ylabel("LSD")
                formatted_bname = str(bname).replace("_", " ").title()
                ax.set_title(
                    f"{format_variable_name(display_var)} — LSD ({formatted_bname}) vs Lead Time",
                    fontsize=10,
                )

                out_png_band = section_output / build_output_filename(
                    metric="energy_ratios_line_per_lead",
                    variable=str(var),
                    level=None,
                    qualifier=f"band_{bname}",
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token,
                    ext="png",
                )
                plt.tight_layout()
                save_fig_helper(fig, out_png_band, module="energy_spectra")
                plt.close(fig)

            # Save bundle NPZ for programmatic use
            out_npz = section_output / build_output_filename(
                metric="energy_spectra_per_lead",
                variable=str(var),
                level=None,
                qualifier="bundle",
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token,
                ext="npz",
            )
            save_data(
                out_npz,
                lead_hours=x_hours,
                wavenumber=kvals,
                energy_target=Zt,
                energy_prediction=Zp,
                log_energy_diff=(np.log10(Zp + 1e-10) - np.log10(Zt + 1e-10)),
                variable=str(var),
                module="energy_spectra",
            )

            # Tripanel shared-y spectrogram (target | model | diff) for quick comparison
            from matplotlib import gridspec as _gridspec

            fig = plt.figure(figsize=(18, 5), dpi=dpi * 2, constrained_layout=True)
            gs = _gridspec.GridSpec(
                1,
                4,
                figure=fig,
                width_ratios=[1.25, 1.25, 0.10, 1.25],
                wspace=0.08,
            )
            axs = [
                fig.add_subplot(gs[0, 0]),  # target
                fig.add_subplot(gs[0, 1]),  # model
                fig.add_subplot(gs[0, 2]),  # divider axis
                fig.add_subplot(gs[0, 3]),  # diff
            ]
            eps = 1e-10
            logZt = np.log10(Zt.T + eps)
            logZp = np.log10(Zp.T + eps)
            vmin_shared = float(np.nanmin([np.nanmin(logZt), np.nanmin(logZp)]))
            vmax_shared = float(np.nanmax([np.nanmax(logZt), np.nanmax(logZp)]))
            diff = logZp - logZt
            axs[0].pcolormesh(
                x_hours,
                kvals,
                logZt,
                shading="auto",
                cmap="viridis",
                vmin=vmin_shared,
                vmax=vmax_shared,
            )
            im1 = axs[1].pcolormesh(
                x_hours,
                kvals,
                logZp,
                shading="auto",
                cmap="viridis",
                vmin=vmin_shared,
                vmax=vmax_shared,
            )
            vmax_d = np.nanmax(np.abs(diff))
            im2 = axs[3].pcolormesh(
                x_hours,
                kvals,
                diff,
                shading="auto",
                cmap="coolwarm",
                vmin=-vmax_d,
                vmax=vmax_d,
            )
            titles = ["Target log10 energy", "Model log10 energy", "Δ log10 energy (M-T)"]
            for i, ax in enumerate([axs[0], axs[1], axs[3]]):
                ax.set_xlabel("lead_time (h)")
                ax.set_title(titles[i if i < 2 else 2], fontsize=11)
                _apply_lead_ticks(ax, x_hours)
            ymin = float(np.nanmin(kvals))
            ymax = float(np.nanmax(kvals))
            for ax in [axs[0], axs[1], axs[3]]:
                ax.set_ylim(ymin, ymax)
            axs[0].set_ylabel("wavenumber (cycles/km)")
            axs[0].tick_params(axis="y", which="both", labelleft=True)
            if not axs[0].get_yticklabels():
                locs = axs[0].get_yticks()
                axs[0].set_yticks(locs)
                axs[0].set_yticklabels([f"{v:g}" for v in locs])
            cbar1 = fig.colorbar(im1, ax=axs[1], orientation="vertical", fraction=0.046, pad=0.02)
            cbar1.set_label("log10 energy")
            cbar2 = fig.colorbar(im2, ax=axs[3], orientation="vertical", fraction=0.046, pad=0.05)
            cbar2.set_label("Δ log10 energy")
            axs[1].set_ylabel("")
            axs[1].tick_params(axis="y", labelleft=False)
            axs[1].set_yticklabels([])
            axs[3].set_ylabel("wavenumber (cycles/km)")
            axs[3].tick_params(axis="y", labelleft=True)
            div_ax = axs[2]
            div_ax.set_xticks([])
            div_ax.set_yticks([])
            for sp in div_ax.spines.values():
                sp.set_visible(False)
            div_ax.set_xlim(0, 1)
            div_ax.set_ylim(0, 1)
            div_ax.add_line(
                Line2D(
                    [0.5, 0.5],
                    [0.0, 1.0],
                    transform=div_ax.transAxes,
                    color="#666",
                    linewidth=1.6,
                    alpha=0.8,
                )
            )

            # Add suptitle
            t_str = f"Energy Spectrogram — {format_variable_name(var)}"
            if init_range:
                start_t, end_t = init_range
                if start_t == end_t:
                    t_str += f" ({start_t})"
                else:
                    t_str += f" ({start_t} — {end_t})"
            fig.suptitle(t_str, fontsize=13)

            out_tripanel = section_output / build_output_filename(
                metric="energy_spectra_per_lead",
                variable=str(var),
                level=None,
                qualifier="tripanel",
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token,
                ext="png",
            )
            c.print(f"[energy_spectra] Saved {out_tripanel}")
            save_fig_helper(fig, out_tripanel, module="energy_spectra")
            plt.close(fig)

        # Save aggregated LSD by-lead CSVs (long and wide) if any rows were collected
        if lsd_long_rows:
            import pandas as _pd

            ldf = _pd.DataFrame(lsd_long_rows)
            out_long = section_output / build_output_filename(
                metric="energy_ratios_per_lead",
                variable=None,
                level=None,
                qualifier="by_lead_long",
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token,
                ext="csv",
            )
            save_dataframe(ldf, out_long, index=False, module="energy_spectra")
            wdf = ldf.pivot_table(index="lead_time_hours", columns="variable", values="LSD")
            wdf = wdf.reset_index()
            out_wide = section_output / build_output_filename(
                metric="energy_ratios_per_lead",
                variable=None,
                level=None,
                qualifier="by_lead_wide",
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token,
                ext="csv",
            )
            save_dataframe(wdf, out_wide, index=False, module="energy_spectra")

        if lsd_banded_long_rows:
            bdf = _pd.DataFrame(lsd_banded_long_rows)
            out_banded = section_output / build_output_filename(
                metric="energy_ratios_bands_per_lead",
                variable=None,
                level=None,
                qualifier="per_lead_time",
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token,
                ext="csv",
            )
            save_dataframe(bdf, out_banded, index=False, module="energy_spectra")

            # Also save per-variable files
            for var_name, group in bdf.groupby("variable"):
                out_banded_var = section_output / build_output_filename(
                    metric="energy_ratios_bands_per_lead",
                    variable=str(var_name),
                    level=None,
                    qualifier="per_lead_time",
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token,
                    ext="csv",
                )
                save_dataframe(group, out_banded_var, index=False, module="energy_spectra")

    # Completion message
    if is_multi_lead:
        c.print("[energy_spectra] Completed per-lead spectrograms only.")
    else:
        c.print("[energy_spectra] Completed energy spectra metrics & plots.")


def add_wavelength_axis(ax, k_min: float, k_max: float) -> None:
    """Add a top axis with wavelength labels (km) corresponding to wavenumber (cycles/km)."""
    wavelength_candidates = [
        40000,
        20000,
        10000,
        5000,
        2000,
        1000,
        500,
        200,
        100,
        50,
        20,
        10,
        5,
        2,
        1,
        0.5,
        0.2,
        0.1,
    ]
    # Keep wavelengths within physical bounds (>= fundamental and <= resolvable small scale)
    wl_min_possible = 1.0 / k_max if k_max > 0 else 0
    wl_max_possible = 1.0 / k_min if k_min > 0 else float("inf")
    valid_wl = [
        wl for wl in wavelength_candidates if wl_min_possible <= wl <= wl_max_possible * 1.01
    ]
    # Convert to wavenumber (cycles/km) and sort ascending (log axis expects ascending positions)
    k_ticks = np.array([1.0 / wl for wl in valid_wl])
    k_ticks = k_ticks[(k_ticks >= k_min) & (k_ticks <= k_max)]
    if k_ticks.size == 0 and k_min > 0 and k_max > k_min:
        k_ticks = np.geomspace(k_min, k_max, num=6)

    if k_ticks.size > 0:
        ax_top = ax.twiny()
        ax_top.set_xscale("log")
        ax_top.set_xlim(k_min, k_max)
        ax_top.set_xticks(k_ticks)

        def _fmt_wl_from_k(k: float) -> str:
            wl = 1.0 / k
            if wl >= 1000:
                return f"{wl / 1000:.0f}k"  # show whole thousands
            if wl >= 100:
                return f"{wl:.0f}"
            if wl >= 10:
                return f"{wl:.0f}"
            if wl >= 1:
                return f"{wl:.1f}"
            return f"{wl:.2f}"

        ax_top.set_xticklabels([_fmt_wl_from_k(k) for k in k_ticks])
        ax_top.set_xlabel("Wavelength (km)")
        ax_top.tick_params(axis="x", which="both", labeltop=True, top=True)
