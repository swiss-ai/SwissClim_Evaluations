from __future__ import annotations

import contextlib
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from swissclim_evaluations.helpers import (
    COLOR_GROUND_TRUTH,
    format_level_label,
    format_variable_name,
)
from swissclim_evaluations.intercomparison.core import (
    c,
    common_files,
    ensure_dir,
    load_npz,
    print_file_list,
    report_checklist,
    report_missing,
    scan_model_sets,
)
from swissclim_evaluations.plots.energy_spectra import add_wavelength_axis

# Canonical spectral band ordering and human-readable labels used by comparison plots.
_BAND_ORDER = ["planetary", "synoptic", "upper_mesoscale", "lower_mesoscale"]
_BAND_LABELS: dict[str, str] = {
    "planetary": "Planetary\n(5–20 Mm)",
    "synoptic": "Synoptic\n(1–5 Mm)",
    "upper_mesoscale": "Upper Meso\n(250–1000 km)",
    "lower_mesoscale": "Lower Meso\n(10–250 km)",
}


def _plot_banded_lsd_bar(
    combined: pd.DataFrame,
    labels: list[str],
    dst: Path,
    out_root: Path,
) -> None:
    """Grouped bar chart of mean LSD per spectral band, one subplot per variable.

    x-axis: spectral band (planetary → lower_mesoscale)
    y-axis: mean LSD
    hue/colour: model
    """
    variables = sorted(combined["variable"].dropna().unique().tolist())
    if not variables:
        return

    ncols = min(3, len(variables))
    nrows = (len(variables) + ncols - 1) // ncols
    colors = sns.color_palette("tab10", n_colors=max(len(labels), 1))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(5, 4 * ncols), max(3.5, 3 * nrows)),
        dpi=160,
        squeeze=False,
    )
    flat_axes = axes.ravel()

    # Build model→colour mapping so colours are consistent across subplots.
    model_colors = {lab: colors[i % len(colors)] for i, lab in enumerate(labels)}

    # Ordered bands (keep only those present in the data).
    bands_present = [b for b in _BAND_ORDER if b in combined["band"].values]
    x = np.arange(len(bands_present))
    n_models = len(labels)
    bar_w = 0.8 / max(n_models, 1)

    for ax_idx, var in enumerate(variables):
        ax = flat_axes[ax_idx]
        sub = combined[combined["variable"] == var]

        for m_idx, lab in enumerate(labels):
            sub_m = sub[sub["model"] == lab]
            heights = []
            for band in bands_present:
                row = sub_m[sub_m["band"] == band]
                heights.append(float(row["lsd_mean"].mean()) if not row.empty else float("nan"))

            offset = (m_idx - (n_models - 1) / 2) * bar_w
            ax.bar(
                x + offset,
                heights,
                width=bar_w * 0.9,
                label=lab,
                color=model_colors.get(lab, colors[m_idx % len(colors)]),
                alpha=0.85,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([_BAND_LABELS.get(b, b) for b in bands_present], fontsize=8)
        ax.set_ylabel("LSD", fontsize=9)
        ax.set_title(format_variable_name(var), fontsize=9)
        ax.grid(axis="y", ls="--", alpha=0.4)
        ax.set_ylim(bottom=0)
        if ax_idx == 0:
            ax.legend(frameon=False, fontsize=8)

    # Hide unused axes.
    for ax in flat_axes[len(variables) :]:
        ax.set_visible(False)

    fig.suptitle("LSD per Spectral Band — Model Comparison", fontsize=11, y=1.01)
    out_png = dst / "lsd_banded_bar_compare.png"
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    c.success(f"Saved {out_png.relative_to(out_root)}")


def _plot_banded_lsd_by_lead(
    combined: pd.DataFrame,
    labels: list[str],
    dst: Path,
    out_root: Path,
) -> None:
    """LSD vs lead time per spectral band, one figure per variable.

    Each figure has one subplot per band.  Each subplot has one line per model.
    """
    variables = sorted(combined["variable"].dropna().unique().tolist())
    if not variables:
        return

    colors = sns.color_palette("tab10", n_colors=max(len(labels), 1))
    model_colors = {lab: colors[i % len(colors)] for i, lab in enumerate(labels)}

    bands_present = [b for b in _BAND_ORDER if b in combined["band"].values]
    if not bands_present:
        return

    # LSD column may be 'LSD' or 'lsd_mean' depending on source.
    lsd_col = "LSD" if "LSD" in combined.columns else "lsd_mean"

    for var in variables:
        sub_var = combined[combined["variable"] == var]
        ncols = len(bands_present)
        fig, axes = plt.subplots(
            1,
            ncols,
            figsize=(max(5, 4 * ncols), 3.5),
            dpi=160,
            squeeze=False,
            sharey=False,
        )

        for b_idx, band in enumerate(bands_present):
            ax = axes[0, b_idx]
            sub_band = sub_var[sub_var["band"] == band]

            for lab in labels:
                sub_m = sub_band[sub_band["model"] == lab].copy()
                if sub_m.empty:
                    continue
                sub_m = sub_m.sort_values("lead_time_hours")
                grouped = sub_m.groupby("lead_time_hours", sort=True)[lsd_col].mean()
                ax.plot(
                    grouped.index,
                    grouped.values,
                    marker="o",
                    markersize=3,
                    label=lab,
                    color=model_colors.get(lab, "tab:blue"),
                )

            ax.set_title(_BAND_LABELS.get(band, band), fontsize=9)
            ax.set_xlabel("Lead Time [h]", fontsize=8)
            if b_idx == 0:
                ax.set_ylabel("LSD", fontsize=9)
            ax.grid(ls="--", alpha=0.4)
            ax.set_ylim(bottom=0)
            if b_idx == 0:
                ax.legend(frameon=False, fontsize=7)

        fig.suptitle(
            f"LSD per Band vs Lead Time — {format_variable_name(var)}",
            fontsize=10,
        )
        safe_var = str(var).replace("/", "_").replace(" ", "_")
        out_png = dst / f"lsd_banded_lead_time_{safe_var}_compare.png"
        plt.tight_layout()
        plt.savefig(out_png, bbox_inches="tight", dpi=200)
        plt.close(fig)
        c.success(f"Saved {out_png.relative_to(out_root)}")


def intercompare_energy_spectra(
    models: list[Path],
    labels: list[str],
    out_root: Path,
    *,
    individual_plots: bool = False,
) -> None:
    """Overlay energy spectra (and ratios) from multiple models.

    Args:
        models: List of model output root directories.
        labels: Display names for each model.
        out_root: Root directory for intercomparison outputs.
        individual_plots: When *True*, generate per-spectrum overlay plots
            (``*_compare.png``, ``*_compare_ratio.png``) and per-lead individual
            spectrum plots (``energy_spectrum_*_lead{h}h_compare.png``) in
            addition to the delta spectrograms.  When *False* (default) and a
            multi-lead dataset is detected, only the compact delta spectrograms
            (``*_spectrogram_delta_compare.png``) are produced.  For single-lead
            datasets (no bundle NPZ files found) individual plots are always
            generated regardless of this flag.
    """
    dst = ensure_dir(out_root / "energy_spectra")

    # Availability report
    total_per_model: list[set[str]] = [set() for _ in models]
    total_union: set[str] = set()

    # We check for spectra NPZ, bundles, and LSD CSVs
    patterns = [
        "energy_spectra/*_spectrum*.npz",
        "energy_spectra/energy_spectra_per_lead_*_bundle*.npz",
        "energy_spectra/lsd_*.csv",
        "energy_spectra/energy_ratios_*.csv",
    ]
    for pat in patterns:
        pm, _, uni = scan_model_sets(models, pat)
        for i, s in enumerate(pm):
            total_per_model[i].update(s)
        total_union.update(uni)

    report_missing("energy_spectra", models, labels, total_per_model, total_union)

    results = {}
    # Spectra: 1-to-1 mapping (each file -> one plot)
    spectra_files = common_files(models, "energy_spectra/*_spectrum*.npz")
    bundles = common_files(models, "energy_spectra/energy_spectra_per_lead_*_bundle*.npz")

    # We generate 2 plots per file (standard + ratio) for spectra, and 1 per lead for bundles
    # But for the checklist, we just count the files found
    results["Energy Spectra Plots"] = len(spectra_files) * 2 + len(bundles)

    # LSD: Many-to-1 mapping (Combined CSVs)
    # We check for presence of inputs to determine if the Combined output
    # will be generated (1) or not (0).
    all_lsd = common_files(models, "energy_spectra/lsd_*.csv")
    all_ratios = common_files(models, "energy_spectra/energy_ratios_*.csv")
    all_metrics = sorted(set(all_lsd + all_ratios))

    avg_no_bands = [
        f for f in all_metrics if "averaged" in f and "bands" not in f and "3d" not in f
    ]
    lvl_no_bands = [f for f in all_metrics if "per_level" in f and "bands" not in f]
    avg_bands = [f for f in all_metrics if "averaged" in f and "bands" in f and "3d" not in f]
    lvl_bands = [f for f in all_metrics if "per_level" in f and "bands" in f]
    # New 3D summaries
    avg_3d = [f for f in all_metrics if "averaged" in f and "3d" in f and "bands" not in f]

    # New temporal summaries (replacing init_time)
    lead_time = [f for f in all_metrics if "lead_time" in f and "3d" not in f]
    plot_datetime = [f for f in all_metrics if "plot_datetime" in f and "3d" not in f]
    lead_time_3d = [f for f in all_metrics if "lead_time" in f and "3d" in f]
    plot_datetime_3d = [f for f in all_metrics if "plot_datetime" in f and "3d" in f]

    def _count_2d_3d(files: list[str]) -> int:
        # Count distinct categories (2D vs 3D) present in the file list
        has_3d = any("3d" in f for f in files)
        has_2d = any("3d" not in f for f in files)
        return (1 if has_3d else 0) + (1 if has_2d else 0)

    # Separate banded-per-lead from global lead_time files.
    band_lead_time = [f for f in all_metrics if "bands" in f and "per_lead_time" in f]
    lead_time_global = [f for f in lead_time if "bands" not in f]

    results["LSD Averaged Metrics"] = _count_2d_3d(avg_no_bands)
    results["LSD Banded Averaged Metrics"] = _count_2d_3d(avg_bands)
    results["LSD Per-Level Metrics"] = _count_2d_3d(lvl_no_bands)
    results["LSD Banded Per-Level Metrics"] = _count_2d_3d(lvl_bands)
    results["LSD 3D Averaged Metrics"] = _count_2d_3d(avg_3d)

    results["LSD Lead Time Metrics"] = _count_2d_3d(lead_time_global)
    results["LSD Banded Lead Time Metrics"] = len(band_lead_time)
    results["LSD Plot Datetime Metrics"] = _count_2d_3d(plot_datetime)
    results["LSD 3D Lead Time Metrics"] = _count_2d_3d(lead_time_3d)
    results["LSD 3D Plot Datetime Metrics"] = _count_2d_3d(plot_datetime_3d)

    report_checklist("energy_spectra", results)

    if all_metrics:
        print_file_list(f"Found {len(all_metrics)} common LSD/Ratio metric files", all_metrics)

    # Process LSD Metrics
    # 1. Averaged (Global)
    frames_avg: list[pd.DataFrame] = []
    for lab, m in zip(labels, models, strict=False):
        cands = [
            f for f in (m / "energy_spectra").glob("lsd_*averaged*.csv") if "bands" not in f.name
        ]
        cands += [
            f
            for f in (m / "energy_spectra").glob("energy_ratios_*averaged*.csv")
            if "bands" not in f.name and "3d" not in f.name
        ]
        for f in cands:
            try:
                df = pd.read_csv(f)
                df.insert(0, "model", lab)
                df["source_file"] = f.name
                frames_avg.append(df)
            except Exception:
                pass

    if frames_avg:
        combined = pd.concat(frames_avg, ignore_index=True)
        if combined["model"].nunique() >= 2:
            out_csv = dst / "lsd_metrics_averaged_combined.csv"
            combined.to_csv(out_csv, index=False)
            c.success(f"Saved {out_csv.relative_to(out_root)}")

    # 2. Banded Averaged
    frames_band_avg: list[pd.DataFrame] = []
    for lab, m in zip(labels, models, strict=False):
        cands = list((m / "energy_spectra").glob("lsd_bands_*averaged*.csv"))
        cands += [
            f
            for f in (m / "energy_spectra").glob("energy_ratios_bands_*averaged*.csv")
            if "3d" not in f.name
        ]
        for f in cands:
            try:
                df = pd.read_csv(f)
                df.insert(0, "model", lab)
                df["source_file"] = f.name
                frames_band_avg.append(df)
            except Exception:
                pass

    if frames_band_avg:
        combined = pd.concat(frames_band_avg, ignore_index=True)
        if combined["model"].nunique() >= 2:
            out_csv = dst / "lsd_metrics_banded_averaged_combined.csv"
            combined.to_csv(out_csv, index=False)
            c.success(f"Saved {out_csv.relative_to(out_root)}")
            # Grouped bar chart: LSD per band, one subplot per variable.
            if "band" in combined.columns and "variable" in combined.columns:
                _plot_banded_lsd_bar(combined, labels, dst, out_root)

    # 3. Per-Level (Global)
    frames_lvl: list[pd.DataFrame] = []
    for lab, m in zip(labels, models, strict=False):
        cands = [
            f for f in (m / "energy_spectra").glob("lsd_*per_level*.csv") if "bands" not in f.name
        ]
        cands += [
            f
            for f in (m / "energy_spectra").glob("energy_ratios_3d_*per_level*.csv")
            if "bands" not in f.name
        ]
        for f in cands:
            try:
                df = pd.read_csv(f)
                df.insert(0, "model", lab)
                df["source_file"] = f.name
                frames_lvl.append(df)
            except Exception:
                pass

    if frames_lvl:
        combined = pd.concat(frames_lvl, ignore_index=True)
        if combined["model"].nunique() >= 2:
            out_csv = dst / "lsd_metrics_per_level_combined.csv"
            combined.to_csv(out_csv, index=False)
            c.success(f"Saved {out_csv.relative_to(out_root)}")

    # 4. Banded Per-Level
    frames_band_lvl: list[pd.DataFrame] = []
    for lab, m in zip(labels, models, strict=False):
        cands = list((m / "energy_spectra").glob("lsd_bands_*per_level*.csv"))
        cands += list((m / "energy_spectra").glob("energy_ratios_bands_3d_*per_level*.csv"))
        for f in cands:
            try:
                df = pd.read_csv(f)
                df.insert(0, "model", lab)
                df["source_file"] = f.name
                frames_band_lvl.append(df)
            except Exception:
                pass

    if frames_band_lvl:
        combined = pd.concat(frames_band_lvl, ignore_index=True)
        if combined["model"].nunique() >= 2:
            out_csv = dst / "lsd_metrics_banded_per_level_combined.csv"
            combined.to_csv(out_csv, index=False)
            c.success(f"Saved {out_csv.relative_to(out_root)}")

    # 5. 3D Averaged (New)
    frames_3d_avg: list[pd.DataFrame] = []
    for lab, m in zip(labels, models, strict=False):
        cands = list((m / "energy_spectra").glob("energy_ratios_3d_*averaged*.csv"))
        for f in cands:
            try:
                df = pd.read_csv(f)
                df.insert(0, "model", lab)
                df["source_file"] = f.name
                frames_3d_avg.append(df)
            except Exception:
                pass

    if frames_3d_avg:
        combined = pd.concat(frames_3d_avg, ignore_index=True)
        if combined["model"].nunique() >= 2:
            out_csv = dst / "lsd_metrics_3d_averaged_combined.csv"
            combined.to_csv(out_csv, index=False)
            c.success(f"Saved {out_csv.relative_to(out_root)}")

    # 6. 3D Lead Time (New)
    frames_3d_lead: list[pd.DataFrame] = []
    for lab, m in zip(labels, models, strict=False):
        cands = list((m / "energy_spectra").glob("energy_ratios_3d_*lead_time*.csv"))
        for f in cands:
            try:
                df = pd.read_csv(f)
                df.insert(0, "model", lab)
                df["source_file"] = f.name
                frames_3d_lead.append(df)
            except Exception:
                pass

    if frames_3d_lead:
        combined = pd.concat(frames_3d_lead, ignore_index=True)
        if combined["model"].nunique() >= 2:
            out_csv = dst / "lsd_metrics_3d_lead_time_combined.csv"
            combined.to_csv(out_csv, index=False)
            c.success(f"Saved {out_csv.relative_to(out_root)}")

    # 8. 2D Lead Time (New) — global LSD only (excluding banded-per-lead handled below)
    frames_lead: list[pd.DataFrame] = []
    for lab, m in zip(labels, models, strict=False):
        cands = [
            f for f in (m / "energy_spectra").glob("lsd_*per_lead_time*.csv") if "3d" not in f.name
        ]
        cands += [
            f
            for f in (m / "energy_spectra").glob("energy_ratios_*lead_time*.csv")
            if "3d" not in f.name and "bands" not in f.name
        ]
        for f in cands:
            try:
                df = pd.read_csv(f)
                df.insert(0, "model", lab)
                df["source_file"] = f.name
                frames_lead.append(df)
            except Exception:
                pass

    if frames_lead:
        combined = pd.concat(frames_lead, ignore_index=True)
        if combined["model"].nunique() >= 2:
            out_csv = dst / "lsd_metrics_lead_time_combined.csv"
            combined.to_csv(out_csv, index=False)
            c.success(f"Saved {out_csv.relative_to(out_root)}")

    # 9. Banded Per-Lead (from multi-lead bundle pipeline)
    frames_band_lead: list[pd.DataFrame] = []
    for lab, m in zip(labels, models, strict=False):
        cands = list((m / "energy_spectra").glob("energy_ratios_bands_per_lead_per_lead_time*.csv"))
        for f in cands:
            try:
                df = pd.read_csv(f)
                df.insert(0, "model", lab)
                df["source_file"] = f.name
                frames_band_lead.append(df)
            except Exception:
                pass

    if frames_band_lead:
        combined_bl = pd.concat(frames_band_lead, ignore_index=True)
        if combined_bl["model"].nunique() >= 2:
            out_csv = dst / "lsd_metrics_banded_lead_time_combined.csv"
            combined_bl.to_csv(out_csv, index=False)
            c.success(f"Saved {out_csv.relative_to(out_root)}")
            # Line chart per variable: LSD vs lead time, one panel per band.
            if "band" in combined_bl.columns and "lead_time_hours" in combined_bl.columns:
                _plot_banded_lsd_by_lead(combined_bl, labels, dst, out_root)

    # Helper to plot a group of NPZ with baseline
    def _plot_group(basenames: list[str]) -> None:
        print_file_list(f"Found {len(basenames)} common energy spectra files", basenames)
        for base in basenames:
            datas = [load_npz(m / "energy_spectra" / base) for m in models]

            # Use explicit fallback logic to avoid ambiguous truth-value evaluation on numpy arrays
            wn = datas[0].get("wavenumber")
            if wn is None:
                wn = datas[0].get("wavenumber_ds")
            spec_ds = datas[0].get("spectrum_target")
            if spec_ds is None:
                spec_ds = datas[0].get("spectrum_ds")

            # Determine surface vs level from metadata
            var = datas[0].get("variable") or "var"
            # Ensure var is a string (handle 0-d numpy arrays from NPZ)
            var = str(var.item()) if hasattr(var, "item") else str(var)

            level_raw = datas[0].get("level")

            fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
            if wn is not None and spec_ds is not None and len(spec_ds) > 0:
                try:  # noqa: SIM105 (allow explicit clarity)
                    ax.loglog(
                        wn[2:-2],
                        np.asarray(spec_ds)[2:-2],
                        color=COLOR_GROUND_TRUTH,
                        lw=2.0,
                        label="Target",
                    )
                except Exception:  # pragma: no cover
                    pass
            colors = sns.color_palette("tab10", n_colors=len(models))
            for i, (lab, dat) in enumerate(zip(labels, datas, strict=False)):
                specm = dat.get("spectrum_prediction")
                wnm = dat.get("wavenumber")
                if wnm is None or specm is None or len(np.asarray(specm)) == 0:
                    continue
                try:
                    ax.loglog(
                        wnm[2:-2],
                        np.asarray(specm)[2:-2],  # ensure numpy array slicing
                        label=lab,
                        color=colors[i],
                    )
                except Exception:
                    continue
            ax.set_xlabel("Zonal Wavenumber (cycles/km)")
            ax.set_ylabel("Energy Density (weighted)")
            title = f"Energy Spectra — {format_variable_name(var)}{format_level_label(level_raw)}"

            # Extract date info if available (single init time)
            init_t = datas[0].get("init_time")
            if init_t is not None:
                val = str(init_t)
                # Check for range pattern first
                match = re.search(r"^(\d{4}-?\d{2}-?\d{2}T\d{2})-(\d{4}-?\d{2}-?\d{2}T\d{2})$", val)
                if match:
                    start, end = match.groups()
                    if start == end:
                        title += f" ({start})"
                else:
                    # Single value or other format
                    if val and val.lower() not in ("none", "noinit"):
                        title += f" ({val})"

            ax.set_title(title)
            ax.grid(True, which="both", ls="--", alpha=0.4)

            # Add golden dotted line at 4*dx cutoff (k_max / 2)
            # We assume all models have roughly the same resolution/grid
            if wn is not None:
                k_max_inter = float(np.nanmax(wn))
                # Validation: only plot cutoff if k_max_inter is finite and > 0
                if np.isfinite(k_max_inter) and k_max_inter > 0:
                    k_cutoff_inter = k_max_inter / 2.0
                    ax.axvline(
                        k_cutoff_inter,
                        color="gold",
                        linestyle=":",
                        linewidth=2,
                        alpha=0.8,
                        label="4dx Cutoff",
                    )

            # Add secondary wavelength axis
            k_min, k_max = ax.get_xlim()
            add_wavelength_axis(ax, k_min, k_max)
            ax.legend(frameon=False)
            out_png = dst / base.replace(".npz", "_compare.png")
            plt.tight_layout()
            plt.savefig(out_png, bbox_inches="tight", dpi=200)
            plt.close(fig)
            c.success(f"Saved {out_png.relative_to(out_root)}")

            # Ratio Plot
            if wn is not None and spec_ds is not None and len(spec_ds) > 0:
                fig_r, ax_r = plt.subplots(figsize=(10, 6), dpi=160)
                for i, (lab, dat) in enumerate(zip(labels, datas, strict=False)):
                    specm = dat.get("spectrum_prediction")
                    if specm is None:
                        specm = dat.get("spectrum_pred")

                    # Try to get the matching target spectrum for this model
                    spec_t = dat.get("spectrum_target")
                    if spec_t is None:
                        spec_t = dat.get("spectrum_ds")
                    # Fallback to the common spec_ds if specific one is missing
                    if spec_t is None:
                        spec_t = spec_ds

                    wnm = dat.get("wavenumber")
                    if wnm is None:
                        wnm = dat.get("wavenumber_prediction")

                    if wnm is None or specm is None or spec_t is None:
                        continue

                    s_m = np.asarray(specm)
                    s_t = np.asarray(spec_t)

                    if s_m.shape != s_t.shape or len(s_m) == 0:
                        continue

                    try:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            ratio = s_m / s_t
                        ratio_pct = ratio * 100.0

                        # Determine model resolution from max wavenumber (Nyquist)
                        # k_max = 1 / (2 * dx)  =>  dx = 1 / (2 * k_max)
                        # We want to cut off wavelengths < 4 * dx
                        # lambda_cutoff = 4 * dx = 2 / k_max
                        # k_cutoff = 1 / lambda_cutoff = k_max / 2
                        k_max_model = np.nanmax(wnm)
                        # Validate k_max_model is finite and positive
                        if not np.isfinite(k_max_model) or k_max_model <= 0:
                            continue
                        k_cutoff = k_max_model / 2.0

                        mask = wnm <= k_cutoff

                        # Apply the same edge trimming as the main plot [2:-2]
                        if len(mask) > 4:
                            mask[:2] = False
                            mask[-2:] = False
                        else:
                            mask[:] = False

                        if not np.any(mask):
                            continue

                        ax_r.semilogx(
                            wnm[mask],
                            ratio_pct[mask],
                            label=lab,
                            color=colors[i],
                        )
                    except Exception:
                        continue

                ax_r.set_xlabel("Zonal Wavenumber (cycles/km)")
                ax_r.set_ylabel("Energy Density Ratio (%)")

                title_ratio = (
                    f"Energy Spectra Ratio — {format_variable_name(var)}"
                    f"{format_level_label(level_raw)}"
                )
                ax_r.set_title(title_ratio)
                ax_r.grid(True, which="both", ls="--", alpha=0.4)
                ax_r.legend(frameon=False)
                ax_r.axhline(100, color="k", linestyle="--", lw=1.0, alpha=0.5)
                ax_r.set_ylim(0, 200)

                # Add secondary top axis for Wavelength (km)
                k_min_plot, k_max_plot = ax_r.get_xlim()

                # Define wavelength candidates (km)
                add_wavelength_axis(ax_r, k_min_plot, k_max_plot)

                if ax_r.get_lines():
                    out_png_ratio = dst / base.replace(".npz", "_compare_ratio.png")
                    plt.tight_layout()
                    plt.savefig(out_png_ratio, bbox_inches="tight", dpi=200)
                    c.success(f"Saved {out_png_ratio.relative_to(out_root)}")
                else:
                    c.warn("No lines were added to the plot; no output saved.")
                plt.close(fig_r)

    # Call the helper to actually generate plots.
    # In multi-lead mode (bundles exist) individual per-spectrum plots are skipped
    # unless explicitly enabled via individual_plots=True.  For single-lead
    # datasets (no bundles) we always fall through to preserve existing behaviour.
    bundles = common_files(models, "energy_spectra/energy_spectra_per_lead_*_bundle*.npz")
    if spectra_files and (individual_plots or not bundles):
        _plot_group(spectra_files)
    elif spectra_files and not individual_plots and bundles:
        c.info(
            "[energy_spectra] Skipping individual spectrum/ratio plots "
            "(individual_plots=False; multi-lead dataset detected). "
            "Set metrics.energy_spectra.individual_plots: true to enable."
        )

    # 4) Plot Energy Spectra per Lead Time (from bundle NPZ)
    # Look for energy_spectra_per_lead_*_bundle.npz
    if bundles:
        print_file_list(f"Found {len(bundles)} common energy spectra bundles", bundles)

        for base in bundles:
            payloads = [load_npz(m / "energy_spectra" / base) for m in models]

            # Check if all have necessary keys
            if not all(
                "energy_prediction" in p and "lead_hours" in p and "wavenumber" in p
                for p in payloads
            ):
                continue

            # Assume lead_hours and wavenumber are same for all models
            lead_hours = payloads[0]["lead_hours"]
            wavenumber = payloads[0]["wavenumber"]
            variable = payloads[0].get("variable", "var")

            # Per-lead individual spectrum plots — only when explicitly enabled.
            if individual_plots:
                for i, h in enumerate(lead_hours):
                    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)

                    # Plot Target (from first model, assuming same target)
                    if "energy_target" in payloads[0]:
                        et = payloads[0]["energy_target"]
                        if et.ndim == 2 and et.shape[0] > i:
                            with contextlib.suppress(Exception):
                                ax.loglog(
                                    wavenumber[2:-2],
                                    et[i, 2:-2],
                                    color=COLOR_GROUND_TRUTH,
                                    lw=2.0,
                                    label="Target",
                                )

                    colors = sns.color_palette("tab10", n_colors=len(models))
                    for j, (lab, pay) in enumerate(zip(labels, payloads, strict=False)):
                        em = pay["energy_prediction"]
                        if em.ndim == 2 and em.shape[0] > i:
                            with contextlib.suppress(Exception):
                                ax.loglog(
                                    wavenumber[2:-2],
                                    em[i, 2:-2],
                                    label=lab,
                                    color=colors[j],
                                )

                    ax.set_xlabel("Zonal Wavenumber (cycles/km)")
                    ax.set_ylabel("Energy Density (weighted)")
                    ax.set_title(
                        f"Energy Spectra — {format_variable_name(str(variable))} (Lead {int(h)}h)"
                    )
                    ax.grid(True, which="both", ls="--", alpha=0.4)

                    # Add wavelength axis
                    k_min, k_max = ax.get_xlim()
                    add_wavelength_axis(ax, k_min, k_max)

                    ax.legend(frameon=False)

                    # Save plot
                    out_png = dst / f"energy_spectrum_{variable}_lead{int(h):03d}h_compare.png"
                    plt.tight_layout()
                    plt.savefig(out_png, bbox_inches="tight", dpi=200)
                    plt.close(fig)

            # Compact delta spectrogram — always generated for multi-lead bundles.
            # Optional compact compare: one delta spectrogram panel per model
            # Δlog10 energy = log10(model) - log10(target)
            if "energy_target" in payloads[0]:
                try:
                    x_hours = np.asarray(lead_hours, dtype=float)
                except Exception:
                    x_hours = np.arange(len(lead_hours), dtype=float)

                # Extract level from the bundle (None for 2D surface vars)
                level_raw = payloads[0].get("level")
                level_val = (
                    int(level_raw.item())
                    if hasattr(level_raw, "item")
                    else (int(level_raw) if level_raw is not None else None)
                )
                level_suffix = format_level_label(level_val) if level_val is not None else ""

                target_energy = np.asarray(payloads[0]["energy_target"])
                if target_energy.ndim == 2 and target_energy.shape[1] == len(wavenumber):
                    eps = 1e-10
                    diffs: list[np.ndarray] = []
                    for pay in payloads:
                        pred_energy = np.asarray(pay["energy_prediction"])
                        if pred_energy.shape != target_energy.shape:
                            continue
                        with np.errstate(divide="ignore", invalid="ignore"):
                            diff = np.log10(pred_energy + eps) - np.log10(target_energy + eps)
                        diffs.append(diff)

                    if diffs:
                        vmax = float(np.nanmax(np.abs(np.stack(diffs, axis=0))))
                        if np.isfinite(vmax) and vmax > 0:
                            ncols = len(diffs)
                            fig, axes = plt.subplots(
                                1,
                                ncols,
                                figsize=(max(5, 4 * ncols), 4.5),
                                dpi=160,
                                squeeze=False,
                                sharey=True,
                            )
                            plotted = False
                            for j, (ax, lab, diff) in enumerate(
                                zip(axes[0], labels, diffs, strict=False)
                            ):
                                im = ax.pcolormesh(
                                    x_hours,
                                    wavenumber,
                                    diff.T,
                                    shading="auto",
                                    cmap="coolwarm",
                                    vmin=-vmax,
                                    vmax=vmax,
                                )
                                ax.set_title(lab, fontsize=9)
                                ax.set_xlabel("Lead Time [h]")
                                if j == 0:
                                    ax.set_ylabel("Wavenumber (cycles/km)")
                                plotted = plotted or im is not None

                            if plotted:
                                cbar = fig.colorbar(
                                    im,
                                    ax=axes.ravel().tolist(),
                                    orientation="horizontal",
                                    location="bottom",
                                    pad=0.12,
                                    shrink=0.8,
                                )
                                cbar.set_label("Δ log10 energy (model - target)")
                                fig.suptitle(
                                    f"Energy Spectrogram Δ — "
                                    f"{format_variable_name(str(variable))}{level_suffix}",
                                    fontsize=11,
                                )
                                out_spec = dst / base.replace(
                                    ".npz", "_spectrogram_delta_compare.png"
                                )
                                plt.tight_layout()
                                plt.savefig(out_spec, bbox_inches="tight", dpi=200)
                                c.success(f"Saved {out_spec.relative_to(out_root)}")
                            plt.close(fig)

        c.success(f"Saved per-lead energy spectra plots to {dst.relative_to(out_root)}")
