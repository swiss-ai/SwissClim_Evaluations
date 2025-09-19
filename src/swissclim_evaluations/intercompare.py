from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr


def _as_paths(items: Iterable[str]) -> list[Path]:
    return [Path(x).resolve() for x in items]


def _model_label(p: Path, explicit: str | None = None) -> str:
    return explicit if explicit else p.name


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _common_files(models: list[Path], rel_glob: str) -> list[str]:
    """Find filenames (basenames) that exist in ALL model folders for a given relative glob.

    Returns a sorted list of basenames present in each folder's pattern.
    """
    sets: list[set[str]] = []
    for m in models:
        files = (
            list(
                (m / rel_glob.split("/")[0]).glob(
                    "/".join(rel_glob.split("/")[1:])
                )
            )
            if "/" in rel_glob
            else list(m.glob(rel_glob))
        )
        sets.append({f.name for f in files if f.is_file()})
    if not sets:
        return []
    common = set.intersection(*sets) if len(sets) > 1 else sets[0]
    return sorted(common)


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def _find_vertical_profile_files(models: list[Path]) -> tuple[list[str], str]:
    """Return common vertical profile NPZ basenames and a mode label.

    Supports new pattern (*_pl_nmae_combined.npz) and legacy (*_pl_rel_error_combined.npz).
    Preference: use new pattern if any found; otherwise try legacy. Only returns files that
    exist in ALL model folders.
    """
    vp_dir = Path("vertical_profiles")
    # New pattern first
    new_pattern = "*_pl_nmae_combined.npz"
    legacy_pattern = "*_pl_rel_error_combined.npz"
    new_sets: list[set[str]] = []
    for m in models:
        files = list((m / vp_dir).glob(new_pattern))
        new_sets.append({f.name for f in files if f.is_file()})
    if new_sets and all(len(s) > 0 for s in new_sets):
        common_new = (
            set.intersection(*new_sets) if len(new_sets) > 1 else new_sets[0]
        )
        if common_new:
            return sorted(common_new), "nmae"
    # Legacy fallback
    legacy_sets: list[set[str]] = []
    for m in models:
        files = list((m / vp_dir).glob(legacy_pattern))
        legacy_sets.append({f.name for f in files if f.is_file()})
    if legacy_sets and all(len(s) > 0 for s in legacy_sets):
        common_legacy = (
            set.intersection(*legacy_sets)
            if len(legacy_sets) > 1
            else legacy_sets[0]
        )
        if common_legacy:
            return sorted(common_legacy), "legacy"
    return [], "none"


def intercompare_vertical_profiles(
    models: list[Path], labels: list[str], out_root: Path
) -> None:
    """Overlay vertical profile NMAE (or legacy relative error) curves across models.

    For each variable present in all model folders we create per-lat-band figure
    (mirrors original 9 south + 9 north band layout => 9 rows x 2 cols) with DS (ground truth) not expressly stored.
    The NPZ files only contain metric curves already reduced vs. level; DS baseline is implicit (NMAE uses target stats).

    We therefore only plot model curves. If legacy rel_error files are used, label plots accordingly.
    """
    basenames, mode = _find_vertical_profile_files(models)
    if not basenames:
        return
    dst = _ensure_dir(out_root / "vertical_profiles")
    color_palette = sns.color_palette("tab10", n_colors=len(models))
    for base in basenames:
        payloads = []
        for m in models:
            try:
                payloads.append(_load_npz(m / "vertical_profiles" / base))
            except Exception:
                payloads.append({})
        if not payloads or any(len(p) == 0 for p in payloads):
            continue
        key_neg = (
            "nmae_neg"
            if mode == "nmae"
            else (
                "rel_error_neg"
                if "rel_error_neg" in payloads[0]
                else "nmae_neg"
            )
        )
        key_pos = (
            "nmae_pos"
            if mode == "nmae"
            else (
                "rel_error_pos"
                if "rel_error_pos" in payloads[0]
                else "nmae_pos"
            )
        )
        if key_neg not in payloads[0] or key_pos not in payloads[0]:
            continue
        neg_arr0 = np.asarray(payloads[0][key_neg])
        bands = neg_arr0.shape[0]
        neg_lat_min = payloads[0].get("neg_lat_min")
        neg_lat_max = payloads[0].get("neg_lat_max")
        pos_lat_min = payloads[0].get("pos_lat_min")
        pos_lat_max = payloads[0].get("pos_lat_max")
        level_values = payloads[0].get("level")
        if level_values is None:
            continue
        fig, axs = plt.subplots(
            bands, 2, figsize=(14, 2.2 * bands), dpi=160, sharey=True
        )
        for j in range(bands):
            axn = axs[j, 0]
            for c, (lab, pay) in enumerate(zip(labels, payloads)):
                arr = np.asarray(pay.get(key_pos))
                if arr is None or arr.shape[0] <= j:
                    continue
                axn.plot(
                    arr[j], level_values, label=lab, color=color_palette[c]
                )
            if pos_lat_min is not None and pos_lat_max is not None:
                axn.set_title(
                    f"Lat {float(pos_lat_min[j])}° to {float(pos_lat_max[j])}° (North)"
                )
            axn.invert_yaxis()
            axn.set_xlabel("NMAE (%)" if mode == "nmae" else "Rel Error")
            axsou = axs[j, 1]
            for c, (lab, pay) in enumerate(zip(labels, payloads)):
                arr = np.asarray(pay.get(key_neg))
                if arr is None or arr.shape[0] <= j:
                    continue
                axsou.plot(
                    arr[j], level_values, label=lab, color=color_palette[c]
                )
            if neg_lat_min is not None and neg_lat_max is not None:
                axsou.set_title(
                    f"Lat {float(neg_lat_min[j])}° to {float(neg_lat_max[j])}° (South)"
                )
            axsou.invert_yaxis()
            axsou.set_xlabel("NMAE (%)" if mode == "nmae" else "Rel Error")
        handles, labels_leg = axs[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels_leg,
                loc="lower center",
                ncol=min(6, len(models)),
            )
        var = base.replace("_pl_nmae_combined.npz", "").replace(
            "_pl_rel_error_combined.npz", ""
        )
        fig.suptitle(
            f"Vertical Profiles — {var} ({'NMAE %' if mode == 'nmae' else 'Relative Error'})",
            y=1.02,
        )
        plt.tight_layout(rect=(0, 0.04, 1, 1))
        out_png = dst / base.replace(".npz", "_compare.png")
        plt.savefig(out_png, bbox_inches="tight", dpi=200)
        plt.close(fig)
        rows = []
        for lab, pay in zip(labels, payloads):
            neg_arr = np.asarray(pay.get(key_neg))
            pos_arr = np.asarray(pay.get(key_pos))
            if neg_arr is None or pos_arr is None:
                continue
            for j in range(bands):
                rows.append({
                    "variable": var,
                    "band_index": j,
                    "hemisphere": "north",
                    "model": lab,
                    "value": float(np.nanmean(pos_arr[j])),
                    "metric": "NMAE" if mode == "nmae" else "RelError",
                })
                rows.append({
                    "variable": var,
                    "band_index": j,
                    "hemisphere": "south",
                    "model": lab,
                    "value": float(np.nanmean(neg_arr[j])),
                    "metric": "NMAE" if mode == "nmae" else "RelError",
                })
        if rows:
            df = pd.DataFrame(rows)
            out_csv = dst / base.replace(".npz", "_summary.csv")
            df.to_csv(out_csv, index=False)
            print(f"[intercompare] saved {out_csv}")


def intercompare_energy_spectra(
    models: list[Path], labels: list[str], out_root: Path
) -> None:
    src_rel = Path("energy_spectra")
    dst = _ensure_dir(out_root / "energy_spectra")

    # 2D variables pattern: {var}_sfc_energy.npz
    common_2d = _common_files(models, str(src_rel / "*_sfc_energy.npz"))
    for base in common_2d:
        datas: list[dict] = []
        for m in models:
            datas.append(_load_npz(m / src_rel / base))
        # Use DS from first model as baseline (identical across models by construction)
        wn = datas[0].get("wavenumber_ds")
        spec_ds = datas[0].get("spectrum_ds")
        fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
        if wn is not None and spec_ds is not None and len(spec_ds) > 0:
            ax.loglog(
                wn[2:-2], spec_ds[2:-2], color="k", lw=2.0, label="Ground Truth"
            )
        colors = sns.color_palette("tab10", n_colors=len(models))
        for i, (lab, dat) in enumerate(zip(labels, datas)):
            wnm = dat.get("wavenumber_ml")
            specm = dat.get("spectrum_ml")
            if wnm is None or specm is None or len(specm) == 0:
                continue
            ax.loglog(wnm[2:-2], specm[2:-2], label=lab, color=colors[i])
        ax.set_xlabel("Longitudinal Wavenumber (cycles/Earth)")
        ax.set_ylabel("Energy Density")
        var = datas[0].get("variable")
        title = f"Energy Spectra — {str(var)} (sfc)"
        ax.set_title(title)
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.legend(frameon=False)
        out_png = dst / base.replace(".npz", "_compare.png")
        plt.tight_layout()
        plt.savefig(out_png, bbox_inches="tight", dpi=200)
        plt.close(fig)

    # 3D variables pattern: {var}_{level}hPa_energy.npz
    common_3d = _common_files(models, str(src_rel / "*_*hPa_energy.npz"))
    for base in common_3d:
        datas = [_load_npz(m / src_rel / base) for m in models]
        wn = datas[0].get("wavenumber_ds")
        spec_ds = datas[0].get("spectrum_ds")
        fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
        if wn is not None and spec_ds is not None and len(spec_ds) > 0:
            ax.loglog(
                wn[2:-2], spec_ds[2:-2], color="k", lw=2.0, label="Ground Truth"
            )
        colors = sns.color_palette("tab10", n_colors=len(models))
        for i, (lab, dat) in enumerate(zip(labels, datas)):
            wnm = dat.get("wavenumber_ml")
            specm = dat.get("spectrum_ml")
            if wnm is None or specm is None or len(specm) == 0:
                continue
            ax.loglog(wnm[2:-2], specm[2:-2], label=lab, color=colors[i])
        var = datas[0].get("variable")
        level = datas[0].get("level")
        ax.set_xlabel("Longitudinal Wavenumber (cycles/Earth)")
        ax.set_ylabel("Energy Density")
        ax.set_title(f"Energy Spectra — {str(var)} {int(level)} hPa")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.legend(frameon=False)
        out_png = dst / base.replace(".npz", "_compare.png")
        plt.tight_layout()
        plt.savefig(out_png, bbox_inches="tight", dpi=200)
        plt.close(fig)

    # Combine LSD CSVs if present
    lsd_2d_rows: list[pd.DataFrame] = []
    for lab, m in zip(labels, models):
        f = m / src_rel / "lsd_2d_metrics.csv"
        if f.is_file():
            df = pd.read_csv(f)
            if "variable" in df.columns:
                df = df.set_index("variable")
            df = df.rename(columns={"lsd": "LSD"})
            df["model"] = lab
            lsd_2d_rows.append(df.reset_index())
    if lsd_2d_rows:
        out_csv = dst / "lsd_2d_metrics_combined.csv"
        pd.concat(lsd_2d_rows, ignore_index=True).to_csv(out_csv, index=False)


def _plot_hist_counts(
    ax, edges: np.ndarray, counts: np.ndarray, label: str, color: str
):
    # Draw as stairs to avoid bar clutter across models and ensure proper alignment
    counts = np.asarray(counts, dtype=float)
    edges = np.asarray(edges, dtype=float)
    ax.stairs(counts, edges, label=label, color=color, alpha=0.9)


def intercompare_histograms(
    models: list[Path],
    labels: list[str],
    out_root: Path,
    max_models_in_legend: int = 12,
) -> None:
    src_rel = Path("histograms")
    dst = _ensure_dir(out_root / "histograms")
    common = _common_files(models, str(src_rel / "*_sfc_latbands_combined.npz"))
    colors = sns.color_palette("tab20", n_colors=max(12, len(models)))
    for base in common:
        payloads = [_load_npz(m / src_rel / base) for m in models]
        # Layout: 9 rows x 2 columns (same as original)
        lat_neg_min = payloads[0].get("neg_lat_min")
        lat_neg_max = payloads[0].get("neg_lat_max")
        lat_pos_min = payloads[0].get("pos_lat_min")
        lat_pos_max = payloads[0].get("pos_lat_max")
        n_rows = len(lat_neg_min)
        fig, axs = plt.subplots(n_rows, 2, figsize=(16, 3 * n_rows), dpi=160)

        # Right column: southern hemisphere bands
        for j in range(n_rows):
            ax = axs[j, 1]
            # Baseline DS from first payload
            ds_ml_pairs = payloads[0]["neg_counts"][j]
            # Each element is (counts_ds, counts_ml)
            counts_ds = ds_ml_pairs[0]
            bins_ds = payloads[0]["neg_bins"][j]
            _plot_hist_counts(
                ax, bins_ds, counts_ds, label="Ground Truth", color="k"
            )
            # Plot each model ML
            for i, (lab, pay) in enumerate(zip(labels, payloads)):
                counts_ml = pay["neg_counts"][j][1]
                bins_ml = pay["neg_bins"][j]
                _plot_hist_counts(
                    ax, bins_ml, counts_ml, label=lab, color=colors[i]
                )
            lat_min = float(lat_neg_min[j])
            lat_max = float(lat_neg_max[j])
            ax.set_title(f"Lat {lat_min}° to {lat_max}° (South)")

        # Left column: northern hemisphere bands
        for j in range(n_rows):
            ax = axs[j, 0]
            ds_ml_pairs = payloads[0]["pos_counts"][j]
            counts_ds = ds_ml_pairs[0]
            bins_ds = payloads[0]["pos_bins"][j]
            _plot_hist_counts(
                ax, bins_ds, counts_ds, label="Ground Truth", color="k"
            )
            for i, (lab, pay) in enumerate(zip(labels, payloads)):
                counts_ml = pay["pos_counts"][j][1]
                bins_ml = pay["pos_bins"][j]
                _plot_hist_counts(
                    ax, bins_ml, counts_ml, label=lab, color=colors[i]
                )
            lat_min = float(lat_pos_min[j])
            lat_max = float(lat_pos_max[j])
            ax.set_title(f"Lat {lat_min}° to {lat_max}° (North)")

        # Legends: add a single shared legend
        handles, labels_leg = axs[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles[: 1 + len(models)],
                labels_leg[: 1 + len(models)],
                loc="lower center",
                ncol=min(6, 1 + len(models)),
            )
        plt.tight_layout(rect=(0, 0.05, 1, 1))
        var = base.replace("_sfc_latbands_combined.npz", "")
        fig.suptitle(f"Distributions by Latitude Bands — {var}", y=1.02)
        out_png = dst / base.replace(".npz", "_compare.png")
        plt.savefig(out_png, bbox_inches="tight", dpi=200)
        plt.close(fig)


def intercompare_wd_kde(
    models: list[Path], labels: list[str], out_root: Path
) -> None:
    src_rel = Path("wd_kde")
    dst = _ensure_dir(out_root / "wd_kde")
    common = _common_files(
        models, str(src_rel / "*_sfc_latbands_kde_combined.npz")
    )
    colors = sns.color_palette("tab10", n_colors=len(models))
    for base in common:
        payloads = [_load_npz(m / src_rel / base) for m in models]
        # Assume each payload carries arrays of object dtype per band
        pos_x0 = payloads[0]["pos_x"]
        n_rows = len(pos_x0)
        fig, axs = plt.subplots(n_rows, 2, figsize=(16, 3 * n_rows), dpi=160)
        # South (right)
        for j in range(n_rows):
            ax = axs[j, 1]
            x_ds = payloads[0]["neg_x"][j]
            kde_ds = payloads[0]["neg_kde_ds"][j]
            ax.plot(x_ds, kde_ds, color="k", lw=2.0, label="Ground Truth")
            for i, (lab, pay) in enumerate(zip(labels, payloads)):
                ax.plot(
                    pay["neg_x"][j],
                    pay["neg_kde_ml"][j],
                    color=colors[i],
                    label=lab,
                )
            lat_min = float(payloads[0]["neg_lat_min"][j])
            lat_max = float(payloads[0]["neg_lat_max"][j])
            ax.set_title(f"Lat {lat_min}° to {lat_max}° (South)")

        # North (left)
        for j in range(n_rows):
            ax = axs[j, 0]
            x_ds = payloads[0]["pos_x"][j]
            kde_ds = payloads[0]["pos_kde_ds"][j]
            ax.plot(x_ds, kde_ds, color="k", lw=2.0, label="Ground Truth")
            for i, (lab, pay) in enumerate(zip(labels, payloads)):
                ax.plot(
                    pay["pos_x"][j],
                    pay["pos_kde_ml"][j],
                    color=colors[i],
                    label=lab,
                )
            lat_min = float(payloads[0]["pos_lat_min"][j])
            lat_max = float(payloads[0]["pos_lat_max"][j])
            ax.set_title(f"Lat {lat_min}° to {lat_max}° (North)")

        handles, labels_leg = axs[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles[: 1 + len(models)],
                labels_leg[: 1 + len(models)],
                loc="lower center",
                ncol=min(6, 1 + len(models)),
            )
        plt.tight_layout(rect=(0, 0.05, 1, 1))
        var = base.replace("_sfc_latbands_kde_combined.npz", "")
        fig.suptitle(f"Normalized KDE by Latitude Bands — {var}", y=1.02)
        out_png = dst / base.replace(".npz", "_compare.png")
        plt.savefig(out_png, bbox_inches="tight", dpi=200)
        plt.close(fig)


def _parse_map_filename(name: str) -> Tuple[str, str]:
    """Return (kind, key) where kind in {"sfc","pl"} and key without extension for grouping.

    Examples:
      202301021200_10m_u_component_of_wind_sfc_ens0.npz -> ("sfc", same name)
      202301021200_temperature_pl_ens0.npz -> ("pl", same name)
    """
    base = name.replace(".npz", "")
    if "_pl" in base:
        return "pl", base
    return "sfc", base


def intercompare_maps(
    models: list[Path], labels: list[str], out_root: Path, max_panels: int = 4
) -> None:
    src_rel = Path("maps")
    dst = _ensure_dir(out_root / "maps")
    common = _common_files(models, str(src_rel / "*.npz"))
    # Limit to first N common map artifacts to avoid huge outputs
    for base in common[:max_panels]:
        kind, key = _parse_map_filename(base)
        payloads = [_load_npz(m / src_rel / (key + ".npz")) for m in models]
        # Extract DS from first payload
        nwp = payloads[0].get("nwp")
        mls = [p.get("ml") for p in payloads]
        if any(x is None for x in mls) or nwp is None:
            continue
        lats = payloads[0].get("latitude")
        lons = payloads[0].get("longitude")
        if lats is None or lons is None:
            continue
        # Determine if data are (lat, lon) or (level, lat, lon)
        def _is_3d(arr: np.ndarray) -> bool:
            return isinstance(arr, np.ndarray) and arr.ndim == 3
        n_levels = nwp.shape[0] if _is_3d(nwp) else 1
        # Sanity: all ML arrays must match dimensionality
        if any((_is_3d(nwp) and (not isinstance(m, np.ndarray) or m.ndim != 3)) or ((not _is_3d(nwp)) and (not isinstance(m, np.ndarray) or m.ndim != 2)) for m in mls):
            print(f"[intercompare][maps] shape mismatch for {key}; skipping")
            continue
        # Loop levels (single iteration for 2D)
        for lvl in range(n_levels):
            nwp_slice = nwp[lvl] if n_levels > 1 else nwp
            ml_slices = [m[lvl] if n_levels > 1 else m for m in mls]
            # Compute vmin/vmax per level for balanced contrast across models
            try:
                vmin = float(np.nanmin([np.nanmin(nwp_slice)] + [np.nanmin(x) for x in ml_slices]))
                vmax = float(np.nanmax([np.nanmax(nwp_slice)] + [np.nanmax(x) for x in ml_slices]))
            except ValueError:
                # In case all NaNs
                print(f"[intercompare][maps] all-NaN data for {key} level {lvl}; skipping")
                continue
            ncols = 1 + len(models)
            fig, axes = plt.subplots(
                1,
                ncols,
                figsize=(6 * ncols, 4),
                dpi=160,
                subplot_kw={"projection": ccrs.PlateCarree()},
            )
            if ncols == 1:
                axes = [axes]
            im0 = axes[0].pcolormesh(
                lons,
                lats,
                nwp_slice,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree(),
            )
            axes[0].add_feature(cfeature.BORDERS, linewidth=0.5)
            axes[0].coastlines(linewidth=0.5)
            title_base = "Ground Truth"
            if n_levels > 1:
                # Attempt to fetch level coordinate if present
                level_vals = payloads[0].get("level")
                if isinstance(level_vals, np.ndarray) and len(level_vals) == n_levels:
                    title_base += f" (level {int(level_vals[lvl])})"
                else:
                    title_base += f" (level {lvl})"
            axes[0].set_title(title_base)
            for ax, lab, ml_slice in zip(axes[1:], labels, ml_slices):
                ax.pcolormesh(
                    lons,
                    lats,
                    ml_slice,
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                    transform=ccrs.PlateCarree(),
                )
                ax.add_feature(cfeature.BORDERS, linewidth=0.5)
                ax.coastlines(linewidth=0.5)
                ax.set_title(lab if n_levels == 1 else f"{lab}")
            cbar_ax = plt.gcf().add_axes([0.15, 0.08, 0.7, 0.03])
            cbar_label = "Value"
            plt.colorbar(im0, cax=cbar_ax, orientation="horizontal", label=cbar_label)
            plt.tight_layout(rect=(0, 0.1, 1, 1))
            suffix = f"_level{lvl}" if n_levels > 1 else ""
            out_png = dst / (key + suffix + "_compare.png")
            plt.savefig(out_png, bbox_inches="tight", dpi=200)
            plt.close(fig)


def intercompare_metrics_csv(
    models: list[Path], labels: list[str], out_root: Path
) -> None:
    # Deterministic metrics
    dst_det = _ensure_dir(out_root / "deterministic")
    frames: list[pd.DataFrame] = []
    frames_std: list[pd.DataFrame] = []
    for lab, m in zip(labels, models):
        f = m / "deterministic" / "metrics.csv"
        if f.is_file():
            df = pd.read_csv(f)
            # Normalize variable column
            if "variable" not in df.columns:
                if "Unnamed: 0" in df.columns:
                    df = df.rename(columns={"Unnamed: 0": "variable"})
                else:
                    # assume first column is variable name
                    first = df.columns[0]
                    df = df.rename(columns={first: "variable"})
            df.insert(0, "model", lab)
            frames.append(df)
        fstd = m / "deterministic" / "metrics_standardized.csv"
        if fstd.is_file():
            df = pd.read_csv(fstd)
            if "variable" not in df.columns:
                if "Unnamed: 0" in df.columns:
                    df = df.rename(columns={"Unnamed: 0": "variable"})
                else:
                    first = df.columns[0]
                    df = df.rename(columns={first: "variable"})
            df.insert(0, "model", lab)
            frames_std.append(df)
    if frames:
        comb = pd.concat(frames, ignore_index=True)
        comb.to_csv(dst_det / "metrics_combined.csv", index=False)
        # Optional: simple bar plots; coerce to numeric and handle all-NaN gracefully
        for metric in ("RMSE", "MAE", "FSS"):
            if metric in comb.columns:
                tmp = comb.copy()
                tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
                pivot = tmp.pivot(
                    index="variable", columns="model", values=metric
                )
                # Drop rows/columns that are entirely NaN
                pivot = pivot.dropna(axis=0, how="all").dropna(
                    axis=1, how="all"
                )
                out_png = dst_det / f"{metric}_compare.png"
                if pivot.empty or pivot.notna().sum().sum() == 0:
                    # Save a one-panel message so users see why it's empty
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.axis("off")
                    ax.text(
                        0.5,
                        0.5,
                        f"No finite values available for {metric} across models/variables.",
                        ha="center",
                        va="center",
                        fontsize=11,
                    )
                    plt.tight_layout()
                    plt.savefig(out_png, bbox_inches="tight", dpi=200)
                    plt.close(fig)
                    print(f"[intercompare] saved placeholder {out_png}")
                    continue
                ax = pivot.plot(kind="bar", figsize=(12, 6))
                ax.set_title(f"{metric} by variable and model")
                ax.set_ylabel(metric)
                plt.tight_layout()
                plt.savefig(out_png, bbox_inches="tight", dpi=200)
                plt.close()
    if frames_std:
        combs = pd.concat(frames_std, ignore_index=True)
        combs.to_csv(dst_det / "metrics_standardized_combined.csv", index=False)

    # ETS
    dst_ets = _ensure_dir(out_root / "ets")
    frames_ets: list[pd.DataFrame] = []
    for lab, m in zip(labels, models):
        f = m / "ets" / "ets_metrics.csv"
        if f.is_file():
            df = pd.read_csv(f)
            df.insert(0, "model", lab)
            frames_ets.append(df)
    if frames_ets:
        comb = pd.concat(frames_ets, ignore_index=True)
        comb.to_csv(dst_ets / "ets_metrics_combined.csv", index=False)


def _plot_step_from_hist(
    ax, edges: np.ndarray, counts: np.ndarray, label: str, color: str
):
    counts = np.asarray(counts, dtype=float)
    edges = np.asarray(edges, dtype=float)
    ax.stairs(counts, edges, label=label, color=color)


def intercompare_probabilistic(
    models: list[Path],
    labels: list[str],
    out_root: Path,
    max_crps_map_panels: int = 4,
) -> None:
    src_rel = Path("probabilistic")
    dst = _ensure_dir(out_root / "probabilistic")

    # 1) Combine CRPS summary (non-WBX) across models
    frames_crps: list[pd.DataFrame] = []
    for lab, m in zip(labels, models):
        f = m / src_rel / "crps_summary.csv"
        if f.is_file():
            df = pd.read_csv(f)
            df.insert(0, "model", lab)
            frames_crps.append(df)
    if frames_crps:
        comb = pd.concat(frames_crps, ignore_index=True)
        comb.to_csv(dst / "crps_summary_combined.csv", index=False)

    # 2) Combine WBX CSV summaries if present
    for basename, outname in (
        ("spread_skill_ratio.csv", "spread_skill_ratio_combined.csv"),
        ("crps_ensemble.csv", "crps_ensemble_combined.csv"),
    ):
        frames: list[pd.DataFrame] = []
        for lab, m in zip(labels, models):
            f = m / src_rel / basename
            if f.is_file():
                df = pd.read_csv(f)
                df.insert(0, "model", lab)
                frames.append(df)
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(
                dst / outname, index=False
            )

    # 3) Overlay PIT histograms by variable
    common_pit = _common_files(models, str(src_rel / "*_pit_hist.npz"))
    colors = sns.color_palette("tab10", n_colors=len(models))
    for base in common_pit:
        payloads = [_load_npz(m / src_rel / base) for m in models]
        fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
        # Uniform reference line at y=1
        ax.axhline(
            1.0, color="brown", linestyle="--", linewidth=1, label="Uniform"
        )
        for i, (lab, pay) in enumerate(zip(labels, payloads)):
            counts = pay.get("counts")
            edges = pay.get("edges")
            if counts is None or edges is None:
                continue
            _plot_step_from_hist(ax, edges, counts, label=lab, color=colors[i])
        var = base.replace("_pit_hist.npz", "")
        ax.set_title(f"PIT histogram — {var}")
        ax.set_xlabel("PIT value")
        ax.set_ylabel("Density")
        ax.legend()
        plt.tight_layout()
        out_png = dst / base.replace(".npz", "_compare.png")
        plt.savefig(out_png, bbox_inches="tight", dpi=200)
        plt.close(fig)

    # 4) Panel CRPS maps from saved NPZ (if available)
    common_crps_map_npz = _common_files(models, str(src_rel / "crps_map_*.npz"))
    for base in common_crps_map_npz[:max_crps_map_panels]:
        payloads = [_load_npz(m / src_rel / base) for m in models]
        # Compute global vmin/vmax across models for consistent color scale
        arrays = [p.get("crps") for p in payloads]
        if any(a is None for a in arrays):
            continue
        vmin = float(np.nanmin([np.nanmin(a) for a in arrays]))
        vmax = float(np.nanmax([np.nanmax(a) for a in arrays]))
        lats = payloads[0].get("latitude")
        lons = payloads[0].get("longitude")
        ncols = len(models)
        fig, axes = plt.subplots(
            1,
            ncols,
            figsize=(6 * ncols, 4),
            dpi=160,
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        if ncols == 1:
            axes = [axes]
        for ax, lab, arr in zip(axes, labels, arrays):
            mesh = ax.pcolormesh(
                lons,
                lats,
                arr,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree(),
            )
            ax.coastlines(linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.set_title(lab)
        cbar_ax = plt.gcf().add_axes([0.15, 0.08, 0.7, 0.03])
        plt.colorbar(mesh, cax=cbar_ax, orientation="horizontal", label="CRPS")
        plt.tight_layout(rect=(0, 0.1, 1, 1))
        out_png = dst / base.replace(".npz", "_compare.png")
        plt.savefig(out_png, bbox_inches="tight", dpi=200)
        plt.close(fig)

    # 5) Combine spatial/temporal WBX NetCDF aggregates into tidy CSVs and simple plots
    # Spatial aggregates
    spatial_rows: list[pd.DataFrame] = []
    for lab, m in zip(labels, models):
        f = m / src_rel / "probabilistic_metrics_spatial.nc"
        if f.is_file():
            try:
                ds = xr.open_dataset(f)
                df = ds.to_dataframe().reset_index()
                # Keep only metric variables we know (columns like 'CRPS.<var>' or 'SSR.<var>')
                value_cols = [
                    c
                    for c in df.columns
                    if isinstance(c, str)
                    and (c.startswith("CRPS.") or c.startswith("SSR."))
                ]
                if not value_cols:
                    continue
                dims_cols = [c for c in df.columns if c not in value_cols]
                # Melt into tidy form
                long = df.melt(
                    id_vars=dims_cols,
                    value_vars=value_cols,
                    var_name="metric_var",
                    value_name="value",
                )
                # Split metric and variable
                parts = long["metric_var"].str.split(".", n=1, expand=True)
                long["metric"] = parts[0]
                long["variable"] = parts[1]
                long = long.drop(columns=["metric_var"])  # cleanup
                long["model"] = lab
                spatial_rows.append(long)
            except Exception:
                pass
    if spatial_rows:
        spatial_df = pd.concat(spatial_rows, ignore_index=True)
        out_csv = dst / "spatial_metrics_combined.csv"
        spatial_df.to_csv(out_csv, index=False)
        print(f"[intercompare] saved {out_csv}")
        # Simple plot: if a region-like column exists, average across variables and plot by region
        region_col = None
        # Prefer canonical 'region' column; else pick the first object-type column among dims
        cand_cols = [c for c in spatial_df.columns if c.lower() == "region"]
        if cand_cols:
            region_col = cand_cols[0]
        else:
            obj_cols = [
                c
                for c in spatial_df.columns
                if spatial_df[c].dtype == object
                and c not in ("metric", "variable", "model")
            ]
            region_col = obj_cols[0] if obj_cols else None
        if region_col:
            for metric in sorted(spatial_df["metric"].unique()):
                tmp = spatial_df[spatial_df["metric"] == metric].copy()
                tmp = tmp.groupby([region_col, "model"], as_index=False)[
                    "value"
                ].mean()
                pivot = tmp.pivot(
                    index=region_col, columns="model", values="value"
                ).sort_index()
                if not pivot.empty and pivot.notna().sum().sum() > 0:
                    ax = pivot.plot(kind="bar", figsize=(12, 6))
                    ax.set_title(f"{metric} (spatial aggregates)")
                    ax.set_ylabel(metric)
                    plt.tight_layout()
                    out_png = dst / f"spatial_{metric}_compare.png"
                    plt.savefig(out_png, bbox_inches="tight", dpi=200)
                    print(f"[intercompare] saved {out_png}")
                    plt.close()

    # Temporal aggregates
    temporal_rows: list[pd.DataFrame] = []
    for lab, m in zip(labels, models):
        f = m / src_rel / "probabilistic_metrics_temporal.nc"
        if f.is_file():
            try:
                ds = xr.open_dataset(f)
                df = ds.to_dataframe().reset_index()
                value_cols = [
                    c
                    for c in df.columns
                    if isinstance(c, str)
                    and (c.startswith("CRPS.") or c.startswith("SSR."))
                ]
                if not value_cols:
                    continue
                dims_cols = [c for c in df.columns if c not in value_cols]
                long = df.melt(
                    id_vars=dims_cols,
                    value_vars=value_cols,
                    var_name="metric_var",
                    value_name="value",
                )
                parts = long["metric_var"].str.split(".", n=1, expand=True)
                long["metric"] = parts[0]
                long["variable"] = parts[1]
                long = long.drop(columns=["metric_var"])  # cleanup
                long["model"] = lab
                temporal_rows.append(long)
            except Exception:
                pass
    if temporal_rows:
        temporal_df = pd.concat(temporal_rows, ignore_index=True)
        out_csv = dst / "temporal_metrics_combined.csv"
        temporal_df.to_csv(out_csv, index=False)
        print(f"[intercompare] saved {out_csv}")
        # Pick a time-bin column to plot if present (e.g., 'season'); else skip plotting
        timebin_col = None
        pref_cols = ["season", "month", "time_bin"]
        for c in pref_cols:
            if c in temporal_df.columns:
                timebin_col = c
                break
        if timebin_col is None:
            # Try any object-like dim besides variable/metric/model
            obj_cols = [
                c
                for c in temporal_df.columns
                if temporal_df[c].dtype == object
                and c not in ("metric", "variable", "model")
            ]
            timebin_col = obj_cols[0] if obj_cols else None
        if timebin_col:
            for metric in sorted(temporal_df["metric"].unique()):
                tmp = temporal_df[temporal_df["metric"] == metric].copy()
                tmp = tmp.groupby([timebin_col, "model"], as_index=False)[
                    "value"
                ].mean()
                # Ensure categorical ordering if seasons
                if timebin_col == "season":
                    order = ["DJF", "MAM", "JJA", "SON"]
                    tmp[timebin_col] = pd.Categorical(
                        tmp[timebin_col], categories=order, ordered=True
                    )
                piv = tmp.pivot(
                    index=timebin_col, columns="model", values="value"
                ).sort_index()
                if not piv.empty and piv.notna().sum().sum() > 0:
                    ax = piv.plot(kind="line", marker="o", figsize=(10, 4))
                    ax.set_title(f"{metric} (temporal aggregates)")
                    ax.set_ylabel(metric)
                    ax.set_xlabel(timebin_col.capitalize())
                    plt.tight_layout()
                    out_png = dst / f"temporal_{metric}_compare.png"
                    plt.savefig(out_png, bbox_inches="tight", dpi=200)
                    print(f"[intercompare] saved {out_png}")
                    plt.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SwissClim Evaluations — Intercomparison of saved artifacts"
    )
    p.add_argument(
        "models",
        nargs="+",
        help="Paths to per-model output folders (e.g., output/modelA output/modelB)",
    )
    p.add_argument(
        "--labels",
        nargs="*",
        help="Optional labels for models (same order as models)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="output/intercomparison",
        help="Output directory for combined plots/CSVs",
    )
    p.add_argument(
        "--modules",
        nargs="*",
        default=["spectra", "hist", "kde", "maps", "metrics", "prob", "vprof"],
        help="Subset of modules to run: spectra, hist, kde, maps, metrics, prob, vprof",
    )
    p.add_argument(
        "--max-map-panels",
        type=int,
        default=4,
        help="Max number of map panels to generate (to limit output size)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    models = _as_paths(args.models)
    labels = (
        args.labels
        if args.labels and len(args.labels) == len(models)
        else [_model_label(p) for p in models]
    )
    out_root = _ensure_dir(Path(args.out))

    # Validate expected sub-structure lightly
    for m in models:
        if not m.exists():
            print(f"[intercompare] WARNING: model folder does not exist: {m}")

    mods = set(args.modules)
    if "spectra" in mods:
        intercompare_energy_spectra(models, labels, out_root)
    if "hist" in mods:
        intercompare_histograms(models, labels, out_root)
    if "kde" in mods:
        intercompare_wd_kde(models, labels, out_root)
    if "maps" in mods:
        intercompare_maps(
            models, labels, out_root, max_panels=int(args.max_map_panels)
        )
    if "metrics" in mods:
        intercompare_metrics_csv(models, labels, out_root)
    if "prob" in mods:
        intercompare_probabilistic(models, labels, out_root)
    if "vprof" in mods:
        intercompare_vertical_profiles(models, labels, out_root)


if __name__ == "__main__":
    main()
