from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from ..helpers import (
    build_output_filename,
    ensemble_mode_to_token,
    get_variable_units,
    resolve_ensemble_mode,
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
    # Remove trivial level dimension
    if "level" in da_var.dims and da_var.sizes.get("level", 0) == 1:
        da_var = da_var.isel(level=0, drop=True)

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
    in_units = get_variable_units(da_var, da_var.name)
    if in_units:
        da_power.attrs["units"] = f"{in_units}^2"
    da_power.attrs["long_name"] = "Latitude-weighted zonal power spectrum"

    # Latitude weighting (cos φ) – retains any non-latitude dims (e.g. ensemble)
    if "latitude" in da_power.coords:
        lat_vals = da_power["latitude"]
        cosw = np.cos(np.deg2rad(lat_vals)).clip(1e-6)
        da_power = da_power.weighted(cosw).mean(dim="latitude")
    else:
        raise ValueError("latitude coordinate required for weighting")

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
        average_dims=["ensemble"] if "ensemble" in da_target.dims else None,
    )
    spec_p = calculate_energy_spectra(
        da_prediction,
        average_dims=["ensemble"] if "ensemble" in da_prediction.dims else None,
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
):
    """Create one spectrum comparison figure & optional NPZ."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi * 2)
    ax.loglog(wavenumber, arr_target, color="skyblue", label="Ground Truth")
    ax.loglog(wavenumber, arr_pred, color="salmon", label="Model Prediction")
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

    lev_part = f" Level {level}" if level is not None else " (sfc)"
    ax.set_title(f"{var}{lev_part} — init={init_label} lead={lead_label}", pad=24)
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    if save_figure and out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
    if save_plot_data and out_path is not None:
        np.savez(
            out_path.with_suffix(".npz"),
            wavenumber=wavenumber,
            spectrum_target=arr_target,
            spectrum_prediction=arr_pred,
            lsd=lsd_val,
            variable=var,
            level=-1 if level is None else level,
            init_time=init_label,
            lead_time=lead_label,
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
) -> xr.DataArray:
    """Generate ONE spectrum & LSD per (init_time, lead_time) combination (no temporal averaging).

    Returns
    -------
    xr.DataArray
        LSD values with remaining time-like dims (init_time, lead_time, ...).
    """
    spectrum_target, spectrum_pred = _compute_spectra_pair(ds_target, ds_prediction, var, level)

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
        lead_label = "none"
        base_dir = out_path.parent if out_path else Path(".")  # fallback
        fname = build_output_filename(
            metric="lsd",
            variable=var,
            level=f"{level}hPa" if level is not None else "surface",
            qualifier="single_spectrum",
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
        )
        return lsd_da

    # Create stacked iterator
    stacked_target = spectrum_target.stack(__time__=time_dims)
    # (stacked_pred not needed explicitly; we index spectrum_pred directly below)
    stacked_lsd = lsd_da.stack(__time__=time_dims)
    coords_df = stacked_target.__time__.to_index()  # MultiIndex with labels

    # Output directory – flattened layout consistent with other modules.
    section_output = out_path.parent if out_path else Path(".")
    section_output.mkdir(parents=True, exist_ok=True)

    for idx, key in enumerate(coords_df):
        sel_kwargs = {str(dim): key[i] for i, dim in enumerate(time_dims)}
        spec_t_1d = spectrum_target.isel(
            **{str(d): spectrum_target.get_index(d).get_loc(sel_kwargs[str(d)]) for d in time_dims}
        )
        spec_p_1d = spectrum_pred.isel(
            **{str(d): spectrum_pred.get_index(d).get_loc(sel_kwargs[str(d)]) for d in time_dims}
        )
        wn = spec_t_1d["wavenumber"].values
        arr_t = spec_t_1d.values
        arr_p = spec_p_1d.values
        lsd_val = float(stacked_lsd.isel(__time__=idx).values)

        # Robust init_time formatting (ensure numpy datetime64)
        if "init_time" in sel_kwargs:
            init_raw = sel_kwargs["init_time"]
            init_label = "noinit"
            try:
                init_np = np.datetime64(init_raw).astype("datetime64[h]")
                init_label = np.datetime_as_string(init_np, unit="h")
            except Exception:
                try:
                    # pandas Timestamp path
                    init_np = np.datetime64(init_raw.to_datetime64())
                    init_label = np.datetime_as_string(init_np.astype("datetime64[h]"), unit="h")
                except Exception:
                    init_label = str(init_raw)
            # sanitize for filename
            init_label = init_label.replace(":", "").replace("-", "")
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
            lead_label = f"{hours:03d}h"
        else:
            lead_label = "noLead"

        fname = build_output_filename(
            metric="lsd",
            variable=var,
            level=f"{level}hPa" if level is not None else None,
            qualifier="spectrum",
            init_time_range=None,
            lead_time_range=None,
            ensemble=ens_token,
            ext="png",
        )
        # Provide a path if either figure OR plot data requested
        target_path = section_output / fname if (save_figure or save_plot_data) else None
        _plot_single_spectrum(
            wn,
            arr_t,
            arr_p,
            lsd_val,
            var,
            level,
            init_label,
            lead_label,
            target_path,
            dpi,
            save_plot_data,
            save_figure,
        )

    return lsd_da  # shape: time dims only (no wavenumber)


def run(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    out_root: Path,
    plotting_cfg: dict[str, Any],
    select_cfg: dict[str, Any],
    ensemble_mode: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> None:
    """Compute Log Spectral Distance (LSD) metrics and optional plots."""

    section_output = out_root / "energy_spectra"
    section_output.mkdir(parents=True, exist_ok=True)

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
    if resolved_mode == "prob":
        raise ValueError("ensemble_mode=prob invalid for energy_spectra")
    members_indices: list[int] | None = None
    if resolved_mode == "members" and has_ens:
        members_indices = list(
            range(
                int(
                    ds_prediction_full.sizes.get("ensemble")
                    or ds_prediction_full.sizes.get("ensemble", 0)
                )
            )
        )
    if resolved_mode == "none" and has_ens:
        resolved_mode = "mean"  # historical behaviour (mean reduction)
    # Track members mode for metrics naming (per-member metrics aggregated without token)
    metrics_members_mode = resolved_mode == "members" and has_ens
    if resolved_mode == "mean" and has_ens:
        if "ensemble" in ds_target_full.dims:
            ds_target_full = ds_target_full.mean(dim="ensemble")
        if "ensemble" in ds_prediction_full.dims:
            ds_prediction_full = ds_prediction_full.mean(dim="ensemble")
        ens_token = ensemble_mode_to_token("mean")
    elif resolved_mode == "pooled" and has_ens:
        ens_token = ensemble_mode_to_token("pooled")
    else:
        ens_token = None

    # Unified suffix based on init_time/lead_time only

    # --- Determine variables & levels ---------------------------------------------
    if "level" in ds_target_full.dims and int(ds_target_full.level.size) > 1:
        variables_3d = [v for v in ds_target_full.data_vars if "level" in ds_target_full[v].dims]
        variables_2d = [v for v in ds_target_full.data_vars if v not in variables_3d]
        levels = select_cfg.get("levels") or list(ds_target_full.level.values)
    else:
        variables_3d = []
        variables_2d = list(ds_target_full.data_vars)
        levels = []

    mode = str(plotting_cfg.get("output_mode", "plot")).lower()
    dpi = int(plotting_cfg.get("dpi", 48))
    save_figures = mode in ("plot", "both")
    save_plot_data = mode in ("npz", "both")  # keep npz behaviour consistent

    # --- Helper: build member contexts (metrics + plotting) -----------------------
    def _member_contexts():
        if members_indices is None:
            yield {
                "member": None,
                "token": ens_token,
                "ds_target": ds_target_full,
                "ds_prediction": ds_prediction_full,
            }
        else:
            for mi in members_indices:
                yield {
                    "member": mi,
                    "token": ensemble_mode_to_token("members", mi),
                    "ds_target": (
                        ds_target_full.isel(ensemble=mi)
                        if "ensemble" in ds_target_full.dims
                        else ds_target_full
                    ),
                    "ds_prediction": ds_prediction_full.isel(ensemble=mi),
                }

    # --- Metrics (full dataset) ----------------------------------------------------
    detailed_rows_2d: list[pd.DataFrame] = []
    summary_rows_2d: list[dict] = []
    per_init_rows_2d: list[pd.DataFrame] = []
    # New banded metrics containers
    banded_detailed_rows_2d: list[pd.DataFrame] = []
    banded_summary_rows_2d: list[dict] = []
    banded_per_init_rows_2d: list[pd.DataFrame] = []
    last_lsd_dims: tuple[str, ...] | None = None

    for var in variables_2d:
        print(f"[energy_spectra] (metrics) 2D variable: {var}")
        member_means: list[float] = []
        for ctx in _member_contexts():
            token_ctx = ctx["token"]
            # Compute spectra once and reuse
            spec_t, spec_p = _compute_spectra_pair(
                ctx["ds_target"], ctx["ds_prediction"], str(var), None
            )
            lsd_da_ctx = _compute_lsd_da(spec_t, spec_p)
            df_lsd_ctx = lsd_da_ctx.to_dataframe(name="lsd").reset_index()
            df_lsd_ctx.insert(0, "variable", str(var))
            if ctx["member"] is not None:
                df_lsd_ctx.insert(1, "ensemble_member", ctx["member"])
                member_means.append(float(lsd_da_ctx.mean().values))
            detailed_rows_2d.append(df_lsd_ctx)
            last_lsd_dims = tuple(str(d) for d in lsd_da_ctx.dims)
            # Only build per-init rows for non-members case (preserve existing behaviour)
            if (ctx["member"] is None) and ("init_time" in lsd_da_ctx.dims):
                mean_over = [d for d in lsd_da_ctx.dims if d not in ("init_time",)]
                lsd_by_init = lsd_da_ctx.mean(dim=mean_over)
                df_init = lsd_by_init.to_dataframe(name="lsd_mean").reset_index()
                df_init.insert(0, "variable", var)
                per_init_rows_2d.append(df_init)

            # Banded LSD using the already computed spectra
            lsd_bands_da = _compute_banded_lsd_da(spec_t, spec_p)  # dims: band + time-like
            # Align banded LSD dims ordering and build DF
            df_bands = lsd_bands_da.to_dataframe(name="lsd").reset_index()
            df_bands.insert(0, "variable", str(var))
            if ctx["member"] is not None:
                df_bands.insert(1, "ensemble_member", ctx["member"])
            banded_detailed_rows_2d.append(df_bands)
            # Per-init (mean over other dims) for non-members
            if (ctx["member"] is None) and ("init_time" in lsd_bands_da.dims):
                mean_over_b = [d for d in lsd_bands_da.dims if d not in ("init_time", "band")]
                lsd_bands_by_init = lsd_bands_da.mean(dim=mean_over_b)
                df_bi = lsd_bands_by_init.to_dataframe(name="lsd_mean").reset_index()
                df_bi.insert(0, "variable", var)
                banded_per_init_rows_2d.append(df_bi)
        # Summary row: non-members -> single lsd_da; members -> mean of member means
        if members_indices is None:
            summary_rows_2d.append(
                {
                    "variable": str(var),
                    "lsd_mean": float(
                        detailed_rows_2d[-1]["lsd"].mean()
                    ),  # last corresponds to var
                }
            )
        elif member_means:
            summary_rows_2d.append(
                {
                    "variable": str(var),
                    "lsd_mean": float(sum(member_means) / len(member_means)),
                }
            )

    if detailed_rows_2d:
        # Per-lead_time if lead_time dim present else per-init_time (else drop 'per' file)
        per_dim_label = None
        if detailed_rows_2d and last_lsd_dims and "lead_time" in last_lsd_dims:
            per_dim_label = "lead_time"
        elif detailed_rows_2d and last_lsd_dims and "init_time" in last_lsd_dims:
            per_dim_label = "init_time"
        # If members mode: keep ensemble_member column in concatenated detailed output
        if per_dim_label:
            pd.concat(detailed_rows_2d, ignore_index=True).to_csv(
                section_output
                / build_output_filename(
                    metric="lsd_2d_metrics",
                    variable=None,
                    level=None,
                    qualifier=f"per_{per_dim_label}",
                    init_time_range=None,
                    lead_time_range=None,
                    ensemble=ens_token if not metrics_members_mode else None,
                    ext="csv",
                ),
                index=False,
            )
        df_summary2d = pd.DataFrame(summary_rows_2d).set_index("variable")
        df_summary2d.to_csv(
            section_output
            / build_output_filename(
                metric="lsd_2d_metrics",
                variable=None,
                level=None,
                qualifier="averaged",
                init_time_range=None,
                lead_time_range=None,
                ensemble=ens_token if not metrics_members_mode else None,
                ext="csv",
            )
        )
        if per_init_rows_2d:
            pd.concat(per_init_rows_2d, ignore_index=True).to_csv(
                section_output
                / build_output_filename(
                    metric="lsd_2d_metrics",
                    variable=None,
                    level=None,
                    qualifier="init_time",
                    init_time_range=None,
                    lead_time_range=None,
                    ensemble=ens_token,
                    ext="csv",
                ),
                index=False,
            )
        print("[energy_spectra] saved 2D LSD metrics (detailed, summary, per-init_time)")

    # Save banded 2D metrics (additive outputs)
    if banded_detailed_rows_2d:
        banded_last_dims = lsd_da_ctx.dims if "lsd_da_ctx" in locals() else last_lsd_dims
        per_dim_label_b = None
        if banded_last_dims and "lead_time" in banded_last_dims:
            per_dim_label_b = "lead_time"
        elif banded_last_dims and "init_time" in banded_last_dims:
            per_dim_label_b = "init_time"
        if per_dim_label_b:
            pd.concat(banded_detailed_rows_2d, ignore_index=True).to_csv(
                section_output
                / build_output_filename(
                    metric="lsd_bands_2d_metrics",
                    variable=None,
                    level=None,
                    qualifier=f"per_{per_dim_label_b}",
                    init_time_range=None,
                    lead_time_range=None,
                    ensemble=ens_token if not metrics_members_mode else None,
                    ext="csv",
                ),
                index=False,
            )
        # Summary over all dims except variable and band
        df_banded = pd.concat(banded_detailed_rows_2d, ignore_index=True)
        group_cols = [c for c in ["variable", "band"] if c in df_banded.columns]
        df_banded_summary = (
            df_banded.groupby(group_cols, as_index=False)["lsd"]
            .mean()
            .rename(columns={"lsd": "lsd_mean"})
        )
        df_banded_summary.to_csv(
            section_output
            / build_output_filename(
                metric="lsd_bands_2d_metrics",
                variable=None,
                level=None,
                qualifier="averaged",
                init_time_range=None,
                lead_time_range=None,
                ensemble=ens_token if not metrics_members_mode else None,
                ext="csv",
            ),
            index=False,
        )
        if banded_per_init_rows_2d:
            pd.concat(banded_per_init_rows_2d, ignore_index=True).to_csv(
                section_output
                / build_output_filename(
                    metric="lsd_bands_2d_metrics",
                    variable=None,
                    level=None,
                    qualifier="init_time",
                    init_time_range=None,
                    lead_time_range=None,
                    ensemble=ens_token,
                    ext="csv",
                ),
                index=False,
            )
        print("[energy_spectra] saved 2D LSD banded metrics (detailed, summary, per-init_time)")

    # 3D variables (refactored)
    detailed_rows_3d: list[pd.DataFrame] = []
    summary_levels: dict[str, list[float]] = {str(v): [] for v in variables_3d}
    per_init_rows_3d: list[pd.DataFrame] = []
    banded_detailed_rows_3d: list[pd.DataFrame] = []
    banded_summary_rows_3d: list[dict] = []
    banded_per_init_rows_3d: list[pd.DataFrame] = []
    for var in variables_3d:
        print(f"[energy_spectra] (metrics) 3D variable: {var}")
        for level in levels:
            member_means_lvl: list[float] = []
            for ctx in _member_contexts():
                token_ctx = ctx["token"]
                # Compute spectra once and reuse
                spec_t, spec_p = _compute_spectra_pair(
                    ctx["ds_target"], ctx["ds_prediction"], str(var), int(level)
                )
                lsd_da_ctx = _compute_lsd_da(spec_t, spec_p)
                df_lsd_ctx = lsd_da_ctx.to_dataframe(name="lsd").reset_index()
                df_lsd_ctx.insert(0, "variable", str(var))
                if "level" in df_lsd_ctx.columns:
                    df_lsd_ctx["level"] = int(level)
                else:
                    df_lsd_ctx.insert(1, "level", int(level))
                detailed_rows_3d.append(df_lsd_ctx)
                last_lsd_dims = tuple(str(d) for d in lsd_da_ctx.dims)
                if ctx["member"] is None:
                    # Only per-init collection for non-members
                    if "init_time" in lsd_da_ctx.dims:
                        mean_over = [d for d in lsd_da_ctx.dims if d not in ("init_time",)]
                        _ = lsd_da_ctx.mean(dim=mean_over)  # not used directly but kept for parity
                        df_init = lsd_da_ctx.to_dataframe(name="lsd").reset_index()
                        df_init.insert(0, "variable", str(var))
                        if "level" in df_init.columns:
                            df_init["level"] = int(level)
                        else:
                            df_init.insert(1, "level", int(level))
                        per_init_rows_3d.append(df_init)
                else:
                    member_means_lvl.append(float(lsd_da_ctx.mean().values))

                # Banded LSD for 3D using the same spectra
                lsd_bands_da = _compute_banded_lsd_da(spec_t, spec_p)
                df_bands = lsd_bands_da.to_dataframe(name="lsd").reset_index()
                df_bands.insert(0, "variable", str(var))
                if "level" in df_bands.columns:
                    df_bands["level"] = int(level)
                else:
                    df_bands.insert(1, "level", int(level))
                if ctx["member"] is not None:
                    df_bands.insert(2, "ensemble_member", ctx["member"])
                banded_detailed_rows_3d.append(df_bands)
                if (ctx["member"] is None) and ("init_time" in lsd_bands_da.dims):
                    mean_over_b = [d for d in lsd_bands_da.dims if d not in ("init_time", "band")]
                    lsd_bi = lsd_bands_da.mean(dim=mean_over_b)
                    df_bi = lsd_bi.to_dataframe(name="lsd_mean").reset_index()
                    df_bi.insert(0, "variable", str(var))
                    if "level" in df_bi.columns:
                        df_bi["level"] = int(level)
                    else:
                        df_bi.insert(1, "level", int(level))
                    banded_per_init_rows_3d.append(df_bi)
            if members_indices is None:
                summary_levels[str(var)].append(float(detailed_rows_3d[-1]["lsd"].mean()))
            elif member_means_lvl:
                summary_levels[str(var)].append(
                    float(sum(member_means_lvl) / len(member_means_lvl))
                )

    if variables_3d:
        if detailed_rows_3d:
            # Decide per-dimension label once using last computed lsd_da
            label = None
            if last_lsd_dims and "lead_time" in last_lsd_dims:
                label = "lead_time"
            elif last_lsd_dims and "init_time" in last_lsd_dims:
                label = "init_time"
            if label:
                pd.concat(detailed_rows_3d, ignore_index=True).to_csv(
                    section_output
                    / build_output_filename(
                        metric="lsd_3d_metrics",
                        variable=None,
                        level=None,
                        qualifier=f"per_{label}",
                        init_time_range=None,
                        lead_time_range=None,
                        ensemble=ens_token,
                        ext="csv",
                    ),
                    index=False,
                )
        df_summary = pd.DataFrame(summary_levels, index=levels)
        df_summary.index.name = "Height Level"

        # Always aggregate over levels for the "averaged" file to provide a single scalar metric
        # per variable (consistent with other modules).
        df_summary_agg = df_summary.mean(axis=0).to_frame().T
        df_summary_agg.index = pd.Index(["mean"], name="Height Level")
        df_summary_agg.to_csv(
            section_output
            / build_output_filename(
                metric="lsd_3d_metrics",
                variable=None,
                level=None,
                qualifier="averaged",
                init_time_range=None,
                lead_time_range=None,
                ensemble=ens_token,
                ext="csv",
            )
        )

        # Save per-level metrics if requested
        if report_per_level:
            df_long = df_summary.reset_index().melt(
                id_vars="Height Level", var_name="variable", value_name="LSD"
            )
            df_long = df_long.rename(columns={"Height Level": "level"})
            df_long.to_csv(
                section_output
                / build_output_filename(
                    metric="lsd_3d_metrics",
                    variable=None,
                    level=None,
                    qualifier="per_level",
                    init_time_range=None,
                    lead_time_range=None,
                    ensemble=ens_token,
                    ext="csv",
                ),
                index=False,
            )
        if per_init_rows_3d:
            pd.concat(per_init_rows_3d, ignore_index=True).to_csv(
                section_output
                / build_output_filename(
                    metric="lsd_3d_metrics",
                    variable=None,
                    level=None,
                    qualifier="init_time",
                    init_time_range=None,
                    lead_time_range=None,
                    ensemble=ens_token,
                    ext="csv",
                ),
                index=False,
            )
        print("[energy_spectra] saved 3D LSD metrics (detailed, summary, per-init_time)")

        # Save banded 3D metrics
        if banded_detailed_rows_3d:
            label_b = label
            if label_b:
                pd.concat(banded_detailed_rows_3d, ignore_index=True).to_csv(
                    section_output
                    / build_output_filename(
                        metric="lsd_bands_3d_metrics",
                        variable=None,
                        level=None,
                        qualifier=f"per_{label_b}",
                        init_time_range=None,
                        lead_time_range=None,
                        ensemble=ens_token,
                        ext="csv",
                    ),
                    index=False,
                )
        # Averaged across time dims, keep variable, level, band
        if banded_detailed_rows_3d:
            df_banded3 = pd.concat(banded_detailed_rows_3d, ignore_index=True)
            group_cols = [c for c in ["variable", "level", "band"] if c in df_banded3.columns]
            df_banded3_summary = (
                df_banded3.groupby(group_cols, as_index=False)["lsd"]
                .mean()
                .rename(columns={"lsd": "lsd_mean"})
            )

            # Always aggregate over levels for the "averaged" file
            group_cols_agg = [c for c in ["variable", "band"] if c in df_banded3_summary.columns]
            df_banded3_summary_agg = df_banded3_summary.groupby(group_cols_agg, as_index=False)[
                "lsd_mean"
            ].mean()
            df_banded3_summary_agg.to_csv(
                section_output
                / build_output_filename(
                    metric="lsd_bands_3d_metrics",
                    variable=None,
                    level=None,
                    qualifier="averaged",
                    init_time_range=None,
                    lead_time_range=None,
                    ensemble=ens_token,
                    ext="csv",
                ),
                index=False,
            )

            # Save per-level metrics if requested
            if report_per_level:
                df_banded3_summary.to_csv(
                    section_output
                    / build_output_filename(
                        metric="lsd_bands_3d_metrics",
                        variable=None,
                        level=None,
                        qualifier="per_level",
                        init_time_range=None,
                        lead_time_range=None,
                        ensemble=ens_token,
                        ext="csv",
                    ),
                    index=False,
                )
        if banded_per_init_rows_3d:
            pd.concat(banded_per_init_rows_3d, ignore_index=True).to_csv(
                section_output
                / build_output_filename(
                    metric="lsd_bands_3d_metrics",
                    variable=None,
                    level=None,
                    qualifier="init_time",
                    init_time_range=None,
                    lead_time_range=None,
                    ensemble=ens_token,
                    ext="csv",
                ),
                index=False,
            )
        print("[energy_spectra] saved 3D LSD banded metrics (detailed, summary, per-init_time)")

    # --- Plotting subset (figures only) -------------------------------------------
    ds_target_plot = ds_target_full
    ds_prediction_plot = ds_prediction_full
    plot_dt_str = (plotting_cfg or {}).get("plot_datetime")
    if plot_dt_str and "init_time" in ds_prediction_full.dims:
        try:
            plot_dt = np.datetime64(plot_dt_str).astype("datetime64[ns]")
            if plot_dt in ds_prediction_full["init_time"].values:
                ds_prediction_plot = ds_prediction_full.sel(init_time=[plot_dt])
                if (
                    "init_time" in ds_target_full.dims
                    and plot_dt in ds_target_full["init_time"].values
                ):
                    ds_target_plot = ds_target_full.sel(init_time=[plot_dt])
                print(f"[energy_spectra] Plot subset init_time={plot_dt_str}")
        except Exception as e:  # pragma: no cover
            print(
                "[energy_spectra] Warning: plot_datetime failed ("
                f"{e}); using full dataset for plots."
            )
    elif (not plot_dt_str) and ("init_time" in ds_prediction_full.dims):
        # Default: plot only first init_time to avoid generating very large number of figures
        try:
            first_dt = ds_prediction_full["init_time"].values[0]
            ds_prediction_plot = ds_prediction_full.sel(init_time=[first_dt])
            if (
                "init_time" in ds_target_full.dims
                and first_dt in ds_target_full["init_time"].values
            ):
                ds_target_plot = ds_target_full.sel(init_time=[first_dt])
            dt_str = np.datetime_as_string(first_dt, unit="h").replace("-", "").replace(":", "")
            print(
                "[energy_spectra] Plotting only first init_time: "
                f"{dt_str} (metrics cover full range)"
            )
        except Exception:
            pass

    if save_figures or save_plot_data:
        for ctx in _member_contexts():
            token_ctx = ctx["token"]
            # Select plotting subset dataset (non-member context). For members we already sliced.
            if ctx["member"] is None:
                ds_tgt_plot_ctx = ds_target_plot
                ds_pred_plot_ctx = ds_prediction_plot
            else:
                ds_tgt_plot_ctx = ctx["ds_target"]
                ds_pred_plot_ctx = ctx["ds_prediction"]
            for var in variables_2d:
                _plot_energy_spectra(
                    ds_tgt_plot_ctx,
                    ds_pred_plot_ctx,
                    str(var),
                    None,
                    section_output
                    / build_output_filename(
                        metric="lsd",
                        variable=str(var),
                        level="surface",
                        qualifier="spectrum",
                        init_time_range=None,
                        lead_time_range=None,
                        ensemble=token_ctx,
                        ext="png",
                    ),
                    dpi,
                    save_plot_data=save_plot_data,
                    save_figure=save_figures,
                    override_ensemble_token=token_ctx,
                )
            for var in variables_3d:
                for level in levels:
                    _plot_energy_spectra(
                        ds_tgt_plot_ctx,
                        ds_pred_plot_ctx,
                        str(var),
                        int(level),
                        section_output
                        / build_output_filename(
                            metric="lsd",
                            variable=str(var),
                            level=f"{level}hPa",
                            qualifier="spectrum",
                            init_time_range=None,
                            lead_time_range=None,
                            ensemble=token_ctx,
                            ext="png",
                        ),
                        dpi,
                        save_plot_data=save_plot_data,
                        save_figure=save_figures,
                        override_ensemble_token=token_ctx,
                    )
        print("[energy_spectra] Figures/NPZ saved (subset dataset)")

    print("[energy_spectra] Completed energy spectra metrics & plots.")


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
