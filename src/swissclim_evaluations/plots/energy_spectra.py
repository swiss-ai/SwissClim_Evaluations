from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from ..helpers import build_output_filename, ensemble_mode_to_token, resolve_ensemble_mode

EARTH_RADIUS_KM = 6371.0
EARTH_CIRCUMFERENCE_KM = 2 * np.pi * EARTH_RADIUS_KM

# Standard wavebands in km (wavelength ranges)
WAVE_BANDS: list[dict[str, float | str]] = [
    {"name": "planetary", "min_km": 5000.0, "max_km": 20000.0},
    {"name": "synoptic", "min_km": 1000.0, "max_km": 5000.0},
    {"name": "mesoscale", "min_km": 10.0, "max_km": 1000.0},
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
    in_units = da_var.attrs.get("units")
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
    wl_min_possible = 1.0 / k_max
    wl_max_possible = 1.0 / k_min
    valid_wl = [
        wl for wl in wavelength_candidates if wl_min_possible <= wl <= wl_max_possible * 1.01
    ]
    # Convert to wavenumber (cycles/km) and sort ascending (log axis expects ascending positions)
    k_ticks = np.array([1.0 / wl for wl in valid_wl])
    k_ticks = k_ticks[(k_ticks >= k_min) & (k_ticks <= k_max)]
    if k_ticks.size == 0:  # fallback to previous geometric spacing
        k_ticks = np.geomspace(k_min, k_max, num=6)

    ax_top = ax.twiny()
    ax_top.set_xscale("log")
    ax_top.set_xlim(ax.get_xlim())
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
    from ..helpers import build_output_filename

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
            level=level if level is not None else None,
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
            level=level if level is not None else None,
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
) -> None:
    """Compute basic 2D energy spectra metrics and write an averaged LSD CSV.

    This minimal implementation restores a valid, lint-clean `run()` after a
    bad merge. It computes per-variable spectra for 2D variables (no 'level'),
    evaluates the Log Spectral Distance (LSD) across remaining time-like dims,
    and writes a single summary CSV using standardized filenames.
    """
    section_output = out_root / "energy_spectra"
    section_output.mkdir(parents=True, exist_ok=True)

    # Resolve ensemble handling (default to mean reduction when present)
    resolved = resolve_ensemble_mode("energy_spectra", ensemble_mode, ds_target, ds_prediction)
    has_ens = "ensemble" in ds_prediction.dims or "ensemble" in ds_target.dims
    ens_token: str | None = None
    tgt = ds_target
    pred = ds_prediction
    if resolved == "prob":
        # Probabilistic semantics are not applicable here; fall back to mean
        resolved = "mean"
    if resolved == "mean" and has_ens:
        if "ensemble" in tgt.dims:
            tgt = tgt.mean(dim="ensemble", keep_attrs=True)
        if "ensemble" in pred.dims:
            pred = pred.mean(dim="ensemble", keep_attrs=True)
        ens_token = ensemble_mode_to_token("mean")
    elif resolved == "pooled" and has_ens:
        ens_token = ensemble_mode_to_token("pooled")
    elif resolved == "members" and has_ens:
        # Members mode not supported in this minimal path; treat as mean for summary
        if "ensemble" in tgt.dims:
            tgt = tgt.mean(dim="ensemble", keep_attrs=True)
        if "ensemble" in pred.dims:
            pred = pred.mean(dim="ensemble", keep_attrs=True)
        ens_token = ensemble_mode_to_token("mean")
    else:
        ens_token = None

    # Helpers to extract time range tokens
    def _extract_init_range(ds: xr.Dataset) -> tuple[str, str] | None:
        if "init_time" not in ds.dims:
            return None
        try:
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
        except Exception:
            return None

    def _extract_lead_range(ds: xr.Dataset) -> tuple[str, str] | None:
        if "lead_time" not in ds.dims:
            return None
        try:
            vals = ds["lead_time"].values
            if getattr(vals, "size", 0) == 0:
                return None
            hours = (vals / np.timedelta64(1, "h")).astype(int)
            sh = int(hours.min())
            eh = int(hours.max())

            def _fmt(h: int) -> str:
                return f"{h:03d}h"

            return (_fmt(sh), _fmt(eh))
        except Exception:
            return None

    init_range = _extract_init_range(pred)
    lead_range = _extract_lead_range(pred)

    # Select 2D variables (exclude true 3D with 'level' dim)
    if "level" in tgt.dims and int(getattr(tgt.level, "size", 0)) > 1:
        variables_2d = [v for v in tgt.data_vars if "level" not in tgt[v].dims]
    else:
        variables_2d = list(tgt.data_vars)

    # Basic schema validation: require longitude for spectra
    has_lon_any = ("longitude" in tgt.dims) or any(
        ("longitude" in tgt[v].dims) for v in variables_2d
    )
    if not has_lon_any:
        raise ValueError("longitude dimension required for energy spectra")

    summary_rows: list[dict[str, Any]] = []
    for var in variables_2d:
        spec_t, spec_p = _compute_spectra_pair(tgt, pred, str(var), None)
        lsd_da = _compute_lsd_da(spec_t, spec_p)
        lsd_mean = float(lsd_da.mean().values)
        summary_rows.append({"variable": str(var), "lsd_mean": lsd_mean})

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows).set_index("variable")
        out_csv = section_output / build_output_filename(
            metric="lsd_2d_metrics",
            variable=None,
            level=None,
            qualifier="averaged",
            init_time_range=init_range,
            lead_time_range=lead_range,
            ensemble=ens_token,
            ext="csv",
        )
        df_summary.to_csv(out_csv)
        print(f"[energy_spectra] saved {out_csv}")

    # Optional: per-member NPZ spectrum exports when requested
    mode = str((plotting_cfg or {}).get("output_mode", "plot")).lower()
    save_npz = mode in ("npz", "both")
    if save_npz and (resolved == "members") and ("ensemble" in ds_prediction.dims):
        dpi = int((plotting_cfg or {}).get("dpi", 48))
        for mi in range(int(ds_prediction.sizes["ensemble"])):
            token = ensemble_mode_to_token("members", mi)
            ds_t_m = tgt.isel(ensemble=mi) if "ensemble" in tgt.dims else tgt
            ds_p_m = pred.isel(ensemble=mi) if "ensemble" in pred.dims else pred
            for var in variables_2d:
                # Use out_path only to convey the parent directory; filenames are built internally
                _ = _plot_energy_spectra(
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
    print("[energy_spectra] Completed energy spectra metrics & plots.")

    # Optional: spectrogram over lead_time (x) and wavenumber (y) with energy color
    try:
        do_spec = bool((plotting_cfg or {}).get("energy_spectra_spectrogram", False))
    except Exception:
        do_spec = False
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
            spec_t, spec_p = _compute_spectra_pair(tgt, pred, str(var), None)
            # Reduce over init_time/time, retaining lead_time and wavenumber
            spec_t2 = _reduce_time_like(spec_t)
            spec_p2 = _reduce_time_like(spec_p)
            # Ensure both share identical coords
            spec_t2, spec_p2 = xr.align(spec_t2, spec_p2, join="inner")
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
                cb = fig.colorbar(im, ax=ax, orientation="vertical")
                cb.set_label("log10 energy")
                out_png = section_output / build_output_filename(
                    metric="energy_spectrogram",
                    variable=_var,
                    level=None,
                    qualifier=qualifier,
                    init_time_range=init_range,
                    lead_time_range=lead_range,
                    ensemble=ens_token,
                    ext="png",
                )
                plt.tight_layout()
                plt.savefig(out_png, bbox_inches="tight", dpi=200)
                print(f"[energy_spectra] saved {out_png}")
                plt.close(fig)

            _plot_spec_img(Zt, "target")
            _plot_spec_img(Zp, "model")

            # Save bundle NPZ for programmatic use
            out_npz = section_output / build_output_filename(
                metric="energy_spectrogram",
                variable=str(var),
                level=None,
                qualifier="bundle",
                init_time_range=init_range,
                lead_time_range=lead_range,
                ensemble=ens_token,
                ext="npz",
            )
            np.savez(
                out_npz,
                lead_hours=x_hours,
                wavenumber=kvals,
                energy_target=Zt,
                energy_model=Zp,
                variable=str(var),
            )
            print(f"[energy_spectra] saved {out_npz}")
