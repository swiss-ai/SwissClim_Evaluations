from __future__ import annotations

import argparse
import contextlib
from collections.abc import Iterable
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import yaml

from .helpers import format_level_token

# Rich-style console utilities for consistent terminal output
try:  # pragma: no cover (console printing)
    from . import console as c
except ImportError:
    try:
        from swissclim_evaluations import console as c  # type: ignore[no-redef]
    except ImportError:
        try:
            import console as c  # type: ignore[no-redef]
        except ImportError:

            class _DummyConsole:
                def __getattr__(self, _name):
                    def _noop(*args, **kwargs):
                        # Fallback to basic print when console is not available
                        if args:
                            print(*args)

                    return _noop

            c = _DummyConsole()  # type: ignore


def _scan_model_sets(
    models: list[Path], rel_glob: str
) -> tuple[list[set[str]], set[str], set[str]]:
    """Return per-model sets, intersection and union for a relative glob pattern.

    rel_glob: e.g. "energy_spectra/*_spectrum*.npz", "maps/map_*.npz".
    """
    dir_part = rel_glob.split("/")[0]
    pat_part = "/".join(rel_glob.split("/")[1:]) if "/" in rel_glob else rel_glob
    per_model: list[set[str]] = []
    for m in models:
        base = (m / dir_part) if dir_part and dir_part != rel_glob else m
        files = list(base.glob(pat_part))
        per_model.append({f.name for f in files if f.is_file()})
    inter = set.intersection(*per_model) if per_model else set()
    uni = set().union(*per_model) if per_model else set()
    return per_model, inter, uni


def _report_missing(
    module: str,
    models: list[Path],
    labels: list[str],
    per_model: list[set[str]],
    union: set[str],
) -> None:
    """Pretty-print which basenames were missing per model for a module scan."""
    if not union:
        c.warn(f"[{module}] No files found in any model.")
        return

    intersection = set.intersection(*per_model) if per_model else set()

    rows: list[str] = []
    rows.append(f"Total unique files: {len(union)}")
    rows.append(f"Common files (in all models): {len(intersection)}")
    rows.append("")  # Spacer

    for lab, files in zip(labels, per_model, strict=False):
        missing = sorted(union - files)
        rows.append(f"• {lab}: present={len(files)} missing={len(missing)}")
        if missing:
            preview = ", ".join(missing[:8])
            if len(missing) > 8:
                preview += ", …"
            rows.append(f"  ↳ missing: {preview}")

    rows.append("")
    rows.append("(Missing counts are relative to the union of files found across all models)")

    c.panel(
        "\n".join(rows),
        title=f"Input Availability — {module}",
        style="yellow",
    )


def _report_checklist(module: str, results: dict[str, int]) -> None:
    """Print a checklist panel for the module with counts."""
    lines = []
    for label, count in results.items():
        if count > 0:
            if "(Ignored)" in label:
                lines.append(f"❌ {label} ({count})")
            else:
                lines.append(f"✅ {label} ({count})")
        else:
            clean_label = label.replace(" (Ignored)", "")
            lines.append(f"❌ {clean_label} (Missing)")

    c.panel(
        "\n".join(lines),
        title=f"Output Checklist — {module}",
        style="blue",
    )


def _print_file_list(msg: str, items: list[str]) -> None:
    """Print a list of files with a header message, using bullets on new lines."""
    if not items:
        return
    # Format: "Header:\n  • item1\n  • item2"
    bullet_list = "\n".join(f"  • {i}" for i in items)
    c.info(f"{msg}:\n{bullet_list}")


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
            list((m / rel_glob.split("/")[0]).glob("/".join(rel_glob.split("/")[1:])))
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


def _find_vertical_profile_files(models: list[Path]) -> list[str]:
    """Return common vertical profile NPZ basenames (current schema).

    New naming uses the standardized builder:
      vprof_nmae_<variable>_multi_combined[_init...][_lead...]_ens*.npz
    We locate files by the stable prefix and the "_combined" qualifier.
    Returns sorted list of basenames existing across all model folders.
    """
    vp_dir = Path("vertical_profiles")
    patterns = [
        "vprof_nmae_*_combined*.npz",  # current
        "*_pl_nmae_combined*.npz",  # legacy fallback
    ]
    # Build intersection over models; if multiple patterns match, union per model first
    sets: list[set[str]] = []
    for m in models:
        model_files: set[str] = set()
        for pat in patterns:
            model_files.update({f.name for f in (m / vp_dir).glob(pat) if f.is_file()})
        sets.append(model_files)
    if not sets:
        return []
    common = set.intersection(*sets) if len(sets) > 1 else sets[0]
    return sorted(common)


def intercompare_vertical_profiles(models: list[Path], labels: list[str], out_root: Path) -> None:
    """Overlay vertical profile NMAE (or legacy relative error) curves.

    For each variable present in all model folders we create per-lat-band figure
    (mirrors original 9 south + 9 north band layout => 9 rows x 2 cols) with DS
    (ground truth) not expressly stored. The NPZ files only contain metric curves
    already reduced vs. level; DS baseline is implicit (NMAE uses target stats).

    We therefore only plot model curves. If legacy rel_error files are used we
    label plots accordingly.
    """
    # Availability report
    per_model_vp: list[set[str]] = []
    for m in models:
        s: set[str] = set()
        for pat in ("vprof_nmae_*_combined*.npz", "*_pl_nmae_combined*.npz"):
            s.update({f.name for f in (m / "vertical_profiles").glob(pat) if f.is_file()})
        per_model_vp.append(s)
    union_vp = set().union(*per_model_vp) if per_model_vp else set()
    if union_vp:
        _report_missing("vertical_profiles", models, labels, per_model_vp, union_vp)

    results = {}
    all_vprof = _common_files(models, "vertical_profiles/vprof_*.npz")
    processed = _find_vertical_profile_files(models)

    results["Vertical Profiles"] = len(processed)

    processed_set = set(processed)
    ignored_count = sum(1 for f in all_vprof if f not in processed_set)
    if ignored_count > 0:
        results["Vertical Profiles (Ignored)"] = ignored_count

    _report_checklist("vertical_profiles", results)

    basenames = _find_vertical_profile_files(models)
    if not basenames:
        c.warn("No common vertical profile files found. Skipping plots.")
        return
    _print_file_list(f"Found {len(basenames)} common vertical profile files", basenames)
    dst = _ensure_dir(out_root / "vertical_profiles")
    color_palette = sns.color_palette("tab10", n_colors=len(models))
    for base in basenames:
        payloads = []
        for m in models:
            try:
                payloads.append(_load_npz(m / "vertical_profiles" / base))
            except Exception:
                payloads.append({})
        # Require at least two models with payload
        valid_models = [
            p for p in payloads if p.get("nmae_pos") is not None and p.get("nmae_neg") is not None
        ]
        if len(valid_models) < 2:
            continue
        key_neg = "nmae_neg"
        key_pos = "nmae_pos"
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
        fig, axs = plt.subplots(bands, 2, figsize=(14, 2.2 * bands), dpi=160, sharey=True)
        for j in range(bands):
            axn = axs[j, 0]
            for idx, (lab, pay) in enumerate(zip(labels, payloads, strict=False)):
                arr = np.asarray(pay.get(key_pos))
                if arr is None or arr.shape[0] <= j:
                    continue
                axn.plot(arr[j], level_values, label=lab, color=color_palette[idx])
            if pos_lat_min is not None and pos_lat_max is not None:
                axn.set_title(f"Lat {float(pos_lat_min[j])}° to {float(pos_lat_max[j])}° (North)")
            axn.invert_yaxis()
            axn.set_xlabel("NMAE (%)")
            axsou = axs[j, 1]
            for idx, (lab, pay) in enumerate(zip(labels, payloads, strict=False)):
                arr = np.asarray(pay.get(key_neg))
                if arr is None or arr.shape[0] <= j:
                    continue
                axsou.plot(arr[j], level_values, label=lab, color=color_palette[idx])
            if neg_lat_min is not None and neg_lat_max is not None:
                axsou.set_title(f"Lat {float(neg_lat_min[j])}° to {float(neg_lat_max[j])}° (South)")
            axsou.invert_yaxis()
            axsou.set_xlabel("NMAE (%)")
        handles, labels_leg = axs[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels_leg,
                loc="lower center",
                ncol=min(6, len(models)),
            )
        # Derive variable name from filename robustly
        var = base[:-4] if base.endswith(".npz") else base
        if var.startswith("vprof_nmae_"):
            # vprof_nmae_<variable>_multi_combined[...]
            tail = var[len("vprof_nmae_") :]
            if "_multi_combined" in tail:
                var = tail.split("_multi_combined", 1)[0]
            else:  # fallback: strip from first _combined
                var = tail.split("_combined", 1)[0]
        else:
            # legacy: <variable>_pl_nmae_combined[...]
            var = var.replace("_pl_nmae_combined", "").replace("_pl_rel_error_combined", "")
        fig.suptitle(f"Vertical Profiles — {var} (NMAE %)", y=1.02)
        plt.tight_layout(rect=(0, 0.04, 1, 1))
        out_png = dst / base.replace(".npz", "_compare.png")
        # Save only if at least two models contributed lines
        plt.savefig(out_png, bbox_inches="tight", dpi=200)
        c.success(f"Saved {out_png.relative_to(out_root)}")
        plt.close(fig)
        rows = []
        for lab, pay in zip(labels, payloads, strict=False):
            neg_arr = np.asarray(pay.get(key_neg))
            pos_arr = np.asarray(pay.get(key_pos))
            if neg_arr is None or pos_arr is None:
                continue
            for j in range(bands):
                with np.errstate(all="ignore"):
                    val_pos = np.nanmean(pos_arr[j]) if pos_arr[j].size else np.nan
                rows.append(
                    {
                        "variable": var,
                        "band_index": j,
                        "hemisphere": "north",
                        "model": lab,
                        "value": float(val_pos) if np.isfinite(val_pos) else np.nan,
                        "metric": "NMAE",
                    }
                )
                rows.append(
                    {
                        "variable": var,
                        "band_index": j,
                        "hemisphere": "south",
                        "model": lab,
                        "value": float(np.nanmean(neg_arr[j])) if neg_arr[j].size else np.nan,
                        "metric": "NMAE",
                    }
                )
        # Save summary only if we have at least two distinct models with values
        if rows:
            df = pd.DataFrame(rows)
            if df["model"].nunique() >= 2:
                out_csv = dst / base.replace(".npz", "_summary.csv")
                df.to_csv(out_csv, index=False)
                c.success(f"Saved {out_csv.relative_to(out_root)}")


def intercompare_energy_spectra(models: list[Path], labels: list[str], out_root: Path) -> None:
    src_rel = Path("energy_spectra")
    dst = _ensure_dir(out_root / "energy_spectra")

    # Availability report
    per_model, inter, uni = _scan_model_sets(models, "energy_spectra/*_spectrum*.npz")
    _report_missing("energy_spectra (spectra NPZ)", models, labels, per_model, uni)

    results = {}
    # Spectra: 1-to-1 mapping (each file -> one plot)
    spectra_files = _common_files(models, "energy_spectra/*_spectrum*.npz")
    # We generate 2 plots per file (standard + ratio)
    results["Energy Spectra Plots"] = len(spectra_files) * 2

    # LSD: Many-to-1 mapping (Combined CSVs)
    # We check for presence of inputs to determine if the Combined output
    # will be generated (1) or not (0).
    all_lsd = _common_files(models, "energy_spectra/lsd_*.csv")

    avg_no_bands = [f for f in all_lsd if "averaged" in f and "bands" not in f]
    lvl_no_bands = [f for f in all_lsd if "per_level" in f and "bands" not in f]
    avg_bands = [f for f in all_lsd if "averaged" in f and "bands" in f]
    lvl_bands = [f for f in all_lsd if "per_level" in f and "bands" in f]

    # Ignored files (present but not currently processed)
    init_time = [f for f in all_lsd if "init_time" in f]
    lead_time = [f for f in all_lsd if "per_lead_time" in f]

    def _count_2d_3d(files: list[str]) -> int:
        has_2d = any("2d" in f for f in files)
        has_3d = any("3d" in f for f in files)
        # If neither explicit tag is found but files exist, assume 1 generic output
        if not has_2d and not has_3d and files:
            return 1
        return (1 if has_2d else 0) + (1 if has_3d else 0)

    results["LSD Averaged Metrics"] = _count_2d_3d(avg_no_bands)
    results["LSD Banded Averaged Metrics"] = _count_2d_3d(avg_bands)
    results["LSD Per-Level Metrics"] = _count_2d_3d(lvl_no_bands)
    results["LSD Banded Per-Level Metrics"] = _count_2d_3d(lvl_bands)

    results["LSD Init Time Metrics (Ignored)"] = len(init_time)
    results["LSD Per Lead Time Metrics (Ignored)"] = len(lead_time)

    _report_checklist("energy_spectra", results)

    # Helper to plot a group of NPZ with baseline
    def _plot_group(basenames: list[str]) -> None:
        for base in basenames:
            datas = [_load_npz(m / src_rel / base) for m in models]
            # Use explicit fallback logic to avoid ambiguous truth-value evaluation on numpy arrays
            wn = datas[0].get("wavenumber")
            if wn is None:
                wn = datas[0].get("wavenumber_ds")
            spec_ds = datas[0].get("spectrum_target")
            if spec_ds is None:
                spec_ds = datas[0].get("spectrum_ds")

            # Determine surface vs level from metadata
            var = datas[0].get("variable") or "var"
            level_raw = datas[0].get("level")

            # Robust check for level
            is_surface = False
            if level_raw is None:
                is_surface = True
            elif isinstance(level_raw, str):
                if level_raw.lower() in ("surface", "", "none"):
                    is_surface = True
            elif isinstance(level_raw, (int | float)) and np.isnan(level_raw):
                is_surface = True

            if is_surface:
                surface = True
                level_val = None
            else:
                surface = False
                level_val = level_raw

            fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
            if wn is not None and spec_ds is not None and len(spec_ds) > 0:
                try:  # noqa: SIM105 (allow explicit clarity)
                    ax.loglog(
                        wn[2:-2],
                        np.asarray(spec_ds)[2:-2],
                        color="k",
                        lw=2.0,
                        label="Ground Truth",
                    )
                except Exception:  # pragma: no cover
                    pass
            colors = sns.color_palette("tab10", n_colors=len(models))
            for i, (lab, dat) in enumerate(zip(labels, datas, strict=False)):
                specm = dat.get("spectrum_prediction")
                if specm is None:
                    specm = dat.get("spectrum_ml")
                wnm = dat.get("wavenumber")
                if wnm is None:
                    wnm = dat.get("wavenumber_ml")
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

            if surface:
                title = f"Energy Spectra — {var} (sfc)"
            else:
                title = f"Energy Spectra — {var} {level_val} hPa"

            ax.set_title(title)
            ax.grid(True, which="both", ls="--", alpha=0.4)

            # Add golden dotted line at 4*dx cutoff (k_max / 2)
            # We assume all models have roughly the same resolution/grid
            if wn is not None:
                k_max_inter = float(np.nanmax(wn))
                k_cutoff_inter = k_max_inter / 2.0
                ax.axvline(
                    k_cutoff_inter,
                    color="gold",
                    linestyle=":",
                    linewidth=2,
                    alpha=0.8,
                    label="4dx Cutoff",
                )

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
                        specm = dat.get("spectrum_ml")

                    # Try to get the matching target spectrum for this model
                    spec_t = dat.get("spectrum_target")
                    if spec_t is None:
                        spec_t = dat.get("spectrum_ds")
                    # Fallback to the common spec_ds if specific one is missing
                    if spec_t is None:
                        spec_t = spec_ds

                    wnm = dat.get("wavenumber")
                    if wnm is None:
                        wnm = dat.get("wavenumber_ml")

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

                if surface:
                    title_ratio = f"Energy Spectra Ratio — {var} (sfc)"
                else:
                    title_ratio = (
                        f"Energy Spectra Ratio — {var} — level: {level_val} hPa"
                        if level_val is not None
                        else f"Energy Spectra Ratio — {var}"
                    )
                ax_r.set_title(title_ratio)
                ax_r.grid(True, which="both", ls="--", alpha=0.4)
                ax_r.legend(frameon=False)
                ax_r.axhline(100, color="k", linestyle="--", lw=1.0, alpha=0.5)

                # Add secondary top axis for Wavelength (km)
                k_min_plot, k_max_plot = ax_r.get_xlim()

                # Define wavelength candidates (km)
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

                wl_min_possible = 1.0 / k_max_plot if k_max_plot > 0 else 0
                wl_max_possible = 1.0 / k_min_plot if k_min_plot > 0 else float("inf")

                valid_wl = [
                    wl
                    for wl in wavelength_candidates
                    if wl_min_possible <= wl <= wl_max_possible * 1.01
                ]

                k_ticks = np.array([1.0 / wl for wl in valid_wl])
                # Filter ticks to be within plot limits
                k_ticks = k_ticks[(k_ticks >= k_min_plot) & (k_ticks <= k_max_plot)]

                if k_ticks.size == 0 and k_min_plot > 0 and k_max_plot > k_min_plot:
                    k_ticks = np.geomspace(k_min_plot, k_max_plot, num=6)

                ax_top = ax_r.twiny()
                ax_top.set_xscale("log")
                ax_top.set_xlim(k_min_plot, k_max_plot)
                ax_top.set_xticks(k_ticks)

                def _fmt_wl_from_k(k: float) -> str:
                    wl = 1.0 / k
                    if wl >= 1000:
                        return f"{wl / 1000:.0f}k"
                    if wl >= 100:
                        return f"{wl:.0f}"
                    if wl >= 10:
                        return f"{wl:.0f}"
                    if wl >= 1:
                        return f"{wl:.1f}"
                    return f"{wl:.2f}"

                ax_top.set_xticklabels([_fmt_wl_from_k(k) for k in k_ticks])
                ax_top.set_xlabel("Wavelength (km)")

                out_png_ratio = dst / base.replace(".npz", "_compare_ratio.png")
                plt.tight_layout()
                plt.savefig(out_png_ratio, bbox_inches="tight", dpi=200)
                plt.close(fig_r)
                c.success(f"Saved {out_png_ratio.relative_to(out_root)}")

    # Collect NPZ patterns (new first, fallback to legacy)
    # Adapt to new standardized naming: lsd_metric_variable_* files replaced by
    # build_output_filename outputs
    # Fallback to legacy glob if any remain
    # New simplified assumption: spectra NPZ basenames already uniform.
    # Retain backward compatibility not required; only support existing saved spectrum npz.
    # Spectrum files in current schema include an ensemble token after '_spectrum',
    # e.g. '..._spectrum_ensnone.npz'.
    surf = _common_files(models, str(src_rel / "*_spectrum*.npz"))
    if not surf:
        c.warn("No common energy spectra files found. Skipping plots.")
    if surf:
        _print_file_list(f"Found {len(surf)} common energy spectra files", surf)

    lsd_csv = _common_files(models, str(src_rel / "lsd_*.csv"))
    if lsd_csv:
        _print_file_list(f"Found {len(lsd_csv)} common LSD CSV files", lsd_csv)

    # Plot all energy spectra files (surface/level determined dynamically)
    if surf:
        _plot_group(surf)

    # Combine LSD summary across models
    # We separate 2D and 3D metrics to avoid confusing combined outputs
    lsd_2d_rows: list[pd.DataFrame] = []
    lsd_3d_rows: list[pd.DataFrame] = []
    lsd_2d_rows_lvl: list[pd.DataFrame] = []
    lsd_3d_rows_lvl: list[pd.DataFrame] = []

    lsd_banded_2d_rows: list[pd.DataFrame] = []
    lsd_banded_3d_rows: list[pd.DataFrame] = []
    lsd_banded_2d_rows_lvl: list[pd.DataFrame] = []
    lsd_banded_3d_rows_lvl: list[pd.DataFrame] = []

    for lab, m in zip(labels, models, strict=False):
        # Averaged (2D & 3D)
        for f in (m / src_rel).glob("lsd_*metrics_*averaged*.csv"):
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            if "variable" not in df.columns and "Unnamed: 0" in df.columns:
                df = df.rename(columns={"Unnamed: 0": "variable"})
            df.insert(0, "model", lab)
            df["source_file"] = f.name

            is_3d = "3d" in f.name
            if "bands" in f.name:
                if is_3d:
                    lsd_banded_3d_rows.append(df)
                else:
                    lsd_banded_2d_rows.append(df)
            else:
                if is_3d:
                    lsd_3d_rows.append(df)
                else:
                    lsd_2d_rows.append(df)

        # Per-level (2D & 3D)
        for f in (m / src_rel).glob("lsd_*metrics_*per_level*.csv"):
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            if "variable" not in df.columns and "Unnamed: 0" in df.columns:
                df = df.rename(columns={"Unnamed: 0": "variable"})
            df.insert(0, "model", lab)
            df["source_file"] = f.name

            is_3d = "3d" in f.name
            if "bands" in f.name:
                if is_3d:
                    lsd_banded_3d_rows_lvl.append(df)
                else:
                    lsd_banded_2d_rows_lvl.append(df)
            else:
                if is_3d:
                    lsd_3d_rows_lvl.append(df)
                else:
                    lsd_2d_rows_lvl.append(df)

    # Helper to save combined CSVs
    def _save_combined(rows: list[pd.DataFrame], name: str) -> None:
        if not rows:
            return
        dfc = pd.concat(rows, ignore_index=True)
        if dfc["model"].nunique() >= 2:
            out_csv = dst / name
            dfc.to_csv(out_csv, index=False)
            c.success(f"Saved {out_csv.relative_to(out_root)}")

    _save_combined(lsd_2d_rows, "lsd_2d_metrics_averaged_combined.csv")
    _save_combined(lsd_3d_rows, "lsd_3d_metrics_averaged_combined.csv")
    # Fallback for legacy files without 2d/3d distinction?
    # If neither 2d nor 3d is in name, they went to 2d list (is_3d=False).
    # We might want to rename the output if it's generic, but explicit is better.

    _save_combined(lsd_2d_rows_lvl, "lsd_2d_metrics_per_level_combined.csv")
    _save_combined(lsd_3d_rows_lvl, "lsd_3d_metrics_per_level_combined.csv")

    _save_combined(lsd_banded_2d_rows, "lsd_bands_2d_metrics_averaged_combined.csv")
    _save_combined(lsd_banded_3d_rows, "lsd_bands_3d_metrics_averaged_combined.csv")

    _save_combined(lsd_banded_2d_rows_lvl, "lsd_bands_2d_metrics_per_level_combined.csv")
    _save_combined(lsd_banded_3d_rows_lvl, "lsd_bands_3d_metrics_per_level_combined.csv")


def _plot_hist_counts(ax, edges: np.ndarray, counts: np.ndarray, label: str, color: str):
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
    # Availability report
    per_model, inter, uni = _scan_model_sets(models, "histograms/hist_*latbands_combined*.npz")
    _report_missing("histograms", models, labels, per_model, uni)

    results = {}
    # Plots: 1-to-1
    plots = _common_files(models, "histograms/hist_*latbands_combined*.npz")
    results["Histograms"] = len(plots)

    # Check for ignored
    all_hist = _common_files(models, "histograms/hist_*.npz")
    ignored = [f for f in all_hist if "latbands_combined" not in f]
    if ignored:
        results["Other Histograms (Ignored)"] = len(ignored)

    _report_checklist("histograms", results)

    common = _common_files(models, str(src_rel / "hist_*latbands_combined*.npz"))
    if not common:
        c.warn("No common histogram files found. Skipping plots.")
        return
    _print_file_list(f"Found {len(common)} common histogram files", common)

    colors = sns.color_palette("tab20", n_colors=max(12, len(models)))

    # --- Global Histograms ---
    per_model_g, inter_g, uni_g = _scan_model_sets(models, "histograms/hist_*global.npz")
    _report_missing("histograms (global)", models, labels, per_model_g, uni_g)
    _report_missing("histograms (global)", models, labels, per_model_g, uni_g)
    common_g = _common_files(models, str(src_rel / "hist_*global.npz"))

    for base in common_g:
        payloads = [_load_npz(m / src_rel / base) for m in models]
        fig, ax = plt.subplots(figsize=(10, 6), dpi=160)

        # Ground Truth (from first model)
        counts_ds = payloads[0]["counts_ds"]
        bins_ds = payloads[0]["bins"]
        _plot_hist_counts(ax, bins_ds, counts_ds, label="Ground Truth", color="k")

        # Models
        for i, (lab, pay) in enumerate(zip(labels, payloads, strict=False)):
            counts_ml = pay["counts_ml"]
            bins_ml = pay["bins"]
            _plot_hist_counts(ax, bins_ml, counts_ml, label=lab, color=colors[i])

        var = base.replace("hist_", "").replace("_global.npz", "")
        ax.set_title(f"Global Histogram — {var}")
        ax.set_ylabel("Frequency (log)")
        ax.set_yscale("log")
        ax.legend(frameon=False)
        ax.grid(True, which="both", ls="--", alpha=0.4)

        out_png = dst / base.replace(".npz", "_compare.png")
        plt.tight_layout()
        plt.savefig(out_png, bbox_inches="tight", dpi=200)
        plt.close(fig)
        c.success(f"Saved {out_png.relative_to(out_root)}")

        ax.set_title(f"Global Histogram - {base.replace('.npz', '')}")
        ax.legend()

        out_png = dst / base.replace(".npz", ".png")
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        c.success(f"Saved {out_png.relative_to(out_root)}")

    # --- Latitude Bands Histograms ---
    # Availability report (always display)
    per_model, inter, uni = _scan_model_sets(models, "histograms/hist_*latbands*.npz")
    # Filter out global histograms from this scan
    per_model = [{f for f in s if "global" not in f} for s in per_model]
    uni = {f for f in uni if "global" not in f}

    _report_missing("histograms (latbands)", models, labels, per_model, uni)
    _report_missing("histograms (latbands)", models, labels, per_model, uni)
    common = _common_files(models, str(src_rel / "hist_*latbands*.npz"))
    common = [f for f in common if "global" not in f]

    if common:
        for base in common:
            payloads = [_load_npz(m / src_rel / base) for m in models]
            # Layout: 9 rows x 2 columns (same as original)
            lat_neg_min = payloads[0].get("neg_lat_min")
            lat_neg_max = payloads[0].get("neg_lat_max")
            lat_pos_min = payloads[0].get("pos_lat_min")
            lat_pos_max = payloads[0].get("pos_lat_max")
            n_rows = len(lat_neg_min) if isinstance(lat_neg_min, (list | np.ndarray)) else 0
            fig, axs = plt.subplots(n_rows, 2, figsize=(16, 3 * n_rows), dpi=160)

            # Right column: southern hemisphere bands
            for j in range(n_rows):
                ax = axs[j, 1]
                # Baseline DS from first payload
                ds_ml_pairs = payloads[0]["neg_counts"][j]
                # Each element is (counts_ds, counts_ml)
                counts_ds = ds_ml_pairs[0]
                bins_ds = payloads[0]["neg_bins"][j]
                _plot_hist_counts(ax, bins_ds, counts_ds, label="Ground Truth", color="k")
                # Plot each model ML
                for i, (lab, pay) in enumerate(zip(labels, payloads, strict=False)):
                    counts_ml = pay["neg_counts"][j][1]
                    bins_ml = pay["neg_bins"][j]
                    _plot_hist_counts(ax, bins_ml, counts_ml, label=lab, color=colors[i])
                lat_min = (
                    float(lat_neg_min[j])
                    if isinstance(lat_neg_min, (list | np.ndarray))
                    else float("nan")
                )
                lat_max = (
                    float(lat_neg_max[j])
                    if isinstance(lat_neg_max, (list | np.ndarray))
                    else float("nan")
                )
                ax.set_title(f"Lat {lat_min}° to {lat_max}° (South)")

            # Left column: northern hemisphere bands
            for j in range(n_rows):
                ax = axs[j, 0]
                ds_ml_pairs = payloads[0]["pos_counts"][j]
                counts_ds = ds_ml_pairs[0]
                bins_ds = payloads[0]["pos_bins"][j]
                _plot_hist_counts(ax, bins_ds, counts_ds, label="Ground Truth", color="k")
                for i, (lab, pay) in enumerate(zip(labels, payloads, strict=False)):
                    counts_ml = pay["pos_counts"][j][1]
                    bins_ml = pay["pos_bins"][j]
                    _plot_hist_counts(ax, bins_ml, counts_ml, label=lab, color=colors[i])
                lat_min = (
                    float(lat_pos_min[j])
                    if isinstance(lat_pos_min, (list | np.ndarray))
                    else float("nan")
                )
                lat_max = (
                    float(lat_pos_max[j])
                    if isinstance(lat_pos_max, (list | np.ndarray))
                    else float("nan")
                )
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
            # Derive a variable/level label for the figure title.
            # Filename schema example: hist_temperature_850_latbands_ensnone.npz
            # We strip leading 'hist_' and everything from the first '_latbands' onwards.
            stem = base[:-4] if base.endswith(".npz") else base
            var_part = stem[len("hist_") :] if stem.startswith("hist_") else stem  # SIM108
            # Remove trailing ensemble token first (e.g., '_ensnone') to simplify pattern removal
            var_part_no_ens = (
                var_part.rsplit("_ens", 1)[0] if "_ens" in var_part else var_part
            )  # SIM108
            # Remove suffix beginning with '_latbands'
            if "_latbands" in var_part_no_ens:
                var_part_no_ens = var_part_no_ens.split("_latbands")[0]
            var = var_part_no_ens
            fig.suptitle(f"Distributions by Latitude Bands — {var}", y=1.02)
            out_png = dst / base.replace(".npz", "_compare.png")
            plt.savefig(out_png, bbox_inches="tight", dpi=200)
            c.success(f"Saved {out_png.relative_to(out_root)}")
            plt.close(fig)


def intercompare_wd_kde(models: list[Path], labels: list[str], out_root: Path) -> None:
    src_rel = Path("wd_kde")
    dst = _ensure_dir(out_root / "wd_kde")
    colors = sns.color_palette("tab10", n_colors=len(models))

    # --- Global KDE ---
    per_model_g, inter_g, uni_g = _scan_model_sets(models, "wd_kde/wd_kde_*global.npz")
    _report_missing("wd_kde (global)", models, labels, per_model_g, uni_g)
    common_g = _common_files(models, str(src_rel / "wd_kde_*global.npz"))

    for base in common_g:
        payloads = [_load_npz(m / src_rel / base) for m in models]
        fig, ax = plt.subplots(figsize=(10, 6), dpi=160)

        # Ground Truth (from first model)
        x_ds = payloads[0]["x"]
        kde_ds = payloads[0]["kde_ds"]
        ax.plot(x_ds, kde_ds, color="k", lw=2.0, label="Ground Truth")

        # Models
        for i, (lab, pay) in enumerate(zip(labels, payloads, strict=False)):
            x_ml = pay["x"]
            kde_ml = pay["kde_ml"]
            ax.plot(x_ml, kde_ml, color=colors[i], label=lab)

        ax.set_title(f"Global Normalized KDE - {base.replace('.npz', '')}")
        ax.legend()

        out_png = dst / base.replace(".npz", "_compare.png")
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        c.success(f"Saved {out_png.relative_to(out_root)}")

    # --- Latitude Bands KDE ---
    # Availability report (always display)
    per_model, inter, uni = _scan_model_sets(models, "wd_kde/wd_kde_*latbands*.npz")
    _report_missing("wd_kde (latbands)", models, labels, per_model, uni)
    common = _common_files(models, str(src_rel / "wd_kde_*latbands*.npz"))
    if not common:
        c.warn("No common WD KDE files found. Skipping plots.")
        return

    # colors already defined
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
            for i, (lab, pay) in enumerate(zip(labels, payloads, strict=False)):
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
            for i, (lab, pay) in enumerate(zip(labels, payloads, strict=False)):
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
        # Extract variable/level part.
        stem = base[:-4] if base.endswith(".npz") else base
        var_part = stem[len("wd_kde_") :] if stem.startswith("wd_kde_") else stem  # SIM108
        # Remove trailing ensemble token if present
        var_part_no_ens = (
            var_part.rsplit("_ens", 1)[0] if "_ens" in var_part else var_part
        )  # SIM108
        # Remove '_latbands' suffix
        if var_part_no_ens.endswith("_latbands"):
            var_part_no_ens = var_part_no_ens[: -len("_latbands")]
        var = var_part_no_ens
        fig.suptitle(f"Normalized KDE by Latitude Bands — {var}", y=1.02)
        out_png = dst / base.replace(".npz", "_compare.png")
        plt.savefig(out_png, bbox_inches="tight", dpi=200)
        c.success(f"Saved {out_png.relative_to(out_root)}")
        plt.close(fig)

    # Combine averaged Wasserstein summary across models if present
    frames_w: list[pd.DataFrame] = []
    for lab, m in zip(labels, models, strict=False):
        for f in (m / src_rel).glob("wd_kde_wasserstein_averaged_*.csv"):
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            if df.empty or (
                "wasserstein_mean" not in df.columns and "wasserstein" not in df.columns
            ):
                continue
            # Normalize column name
            if "wasserstein" in df.columns and "wasserstein_mean" not in df.columns:
                df = df.rename(columns={"wasserstein": "wasserstein_mean"})
            df.insert(0, "model", lab)
            df["source_file"] = f.name
            frames_w.append(df)
    if frames_w:
        combined = pd.concat(frames_w, ignore_index=True)
        if combined["model"].nunique() >= 2:
            out_csv = dst / "wd_kde_wasserstein_averaged_combined.csv"
            combined.to_csv(out_csv, index=False)
            c.success(f"Saved {out_csv.relative_to(out_root)}")


def _parse_map_filename(name: str) -> str:
    """Return base key without extension.

    New schema already omits placeholder tokens; we simply strip extension.
    """
    return name[:-4] if name.endswith(".npz") else name


def intercompare_maps(
    models: list[Path],
    labels: list[str],
    out_root: Path,
    max_panels: int = 4,
) -> None:
    src_rel = Path("maps")
    dst = _ensure_dir(out_root / "maps")
    # Availability report
    per_model, inter, uni = _scan_model_sets(models, "maps/map_*.npz")
    _report_missing("maps", models, labels, per_model, uni)

    results = {}
    maps = _common_files(models, "maps/map_*.npz")

    # Maps are 1-to-1, but limited by max_panels
    processed_count = min(len(maps), max_panels)
    ignored_count = max(0, len(maps) - max_panels)

    results["Maps"] = processed_count
    if ignored_count > 0:
        results["Maps (Ignored)"] = ignored_count

    _report_checklist("maps", results)

    # New schema: map_<var>[ _<level>][ _init...][ _lead...]_ens*.npz
    common = _common_files(models, str(src_rel / "map_*.npz"))
    if not common:
        c.warn("No common map files found. Skipping plots.")
        return
    _print_file_list(f"Found {len(common)} common map files", common)
    # Limit to first N common map artifacts to avoid huge outputs
    for base in common[:max_panels]:
        key = _parse_map_filename(base)
        payloads = [_load_npz(m / src_rel / f"{key}.npz") for m in models]
        # Extract DS from first payload
        nwp = payloads[0].get("nwp")
        mls = [p.get("ml") for p in payloads]
        if any(x is None for x in mls) or nwp is None:
            continue
        lats = payloads[0].get("latitude")
        lons = payloads[0].get("longitude")
        if lats is None or lons is None:
            continue

        def _is_3d(arr: np.ndarray) -> bool:
            return isinstance(arr, np.ndarray) and arr.ndim == 3

        n_levels = nwp.shape[0] if _is_3d(nwp) else 1
        if any(
            (_is_3d(nwp) and (not isinstance(m, np.ndarray) or m.ndim != 3))
            or ((not _is_3d(nwp)) and (not isinstance(m, np.ndarray) or m.ndim != 2))
            for m in mls
        ):
            c.warn(f"maps: shape mismatch for {key}; skipping")
            continue
        level_vals = payloads[0].get("level")
        for lvl in range(n_levels):
            nwp_slice = nwp[lvl] if n_levels > 1 else nwp
            ml_slices = [m[lvl] if n_levels > 1 else m for m in mls if isinstance(m, np.ndarray)]
            try:
                vmin = float(np.nanmin([np.nanmin(nwp_slice)] + [np.nanmin(x) for x in ml_slices]))
                vmax = float(np.nanmax([np.nanmax(nwp_slice)] + [np.nanmax(x) for x in ml_slices]))
            except ValueError:
                c.warn(f"maps: all-NaN data for {key} level {lvl}; skipping")
                continue
            ncols = 1 + len(models)
            fig, axes = plt.subplots(
                1,
                ncols,
                figsize=(6 * ncols, 4),
                dpi=160,
                subplot_kw={"projection": ccrs.PlateCarree()},
                constrained_layout=True,
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
                if isinstance(level_vals, np.ndarray) and len(level_vals) == n_levels:
                    level_token = format_level_token(level_vals[lvl])
                    title_base += f" (level {level_token})"
                else:
                    title_token = format_level_token(lvl)
                    title_base += f" (level {title_token})"
            axes[0].set_title(title_base)
            for ax, lab, ml_slice in zip(axes[1:], labels, ml_slices, strict=False):
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
            # Use a constrained-layout-friendly colorbar spanning all axes
            cbar = fig.colorbar(
                im0,
                ax=axes if isinstance(axes, (list | np.ndarray)) else [axes],
                orientation="horizontal",
                fraction=0.05,
                pad=0.08,
            )
            with contextlib.suppress(Exception):
                cbar.set_label("Value")
            # No tight_layout here; constrained_layout handles spacing
            suffix = ""
            if n_levels > 1:
                if isinstance(level_vals, np.ndarray) and len(level_vals) == n_levels:
                    suffix = f"_level{format_level_token(level_vals[lvl])}"
                else:
                    suffix = f"_level{format_level_token(lvl)}"
            out_png = dst / (key + suffix + "_compare.png")
            plt.savefig(out_png, bbox_inches="tight", dpi=200)
            c.success(f"Saved {out_png.relative_to(out_root)}")
            plt.close(fig)


def intercompare_deterministic_metrics(
    models: list[Path], labels: list[str], out_root: Path
) -> None:
    # Availability report
    per_model, inter, uni = _scan_model_sets(models, "deterministic/deterministic_metrics*.csv")
    _report_missing("deterministic_metrics", models, labels, per_model, uni)

    results = {}
    all_det = _common_files(models, "deterministic/deterministic_metrics*.csv")

    avg = [f for f in all_det if "averaged" in f]
    lvl = [f for f in all_det if "per_level" in f and "standardized" not in f]
    std = [f for f in all_det if "standardized" in f and "per_level" not in f]
    std_lvl = [f for f in all_det if "standardized" in f and "per_level" in f]

    results["Deterministic Averaged"] = 1 if avg else 0
    results["Deterministic Per Level"] = 1 if lvl else 0
    results["Deterministic Standardized"] = 1 if std else 0
    results["Deterministic Standardized Per Level"] = 1 if std_lvl else 0

    # Count potential plots by inspecting the first averaged file
    plot_count = 0
    if avg:
        try:
            # Read just the header of the first available file
            first_file = models[0] / "deterministic" / avg[0]
            if first_file.is_file():
                with open(first_file) as fh:
                    header = fh.readline()
                cols = [col.strip() for col in header.split(",")]
                # Exclude known non-metric columns
                excluded = {
                    "variable",
                    "model",
                    "level",
                    "lead_time",
                    "init_time",
                    "valid_time",
                    "Unnamed: 0",
                    "source_file",
                    "member",
                    "threshold",
                    "",
                }
                plot_count = sum(1 for col in cols if col not in excluded)
        except Exception:
            # If the file is missing or malformed, just skip plot counting.
            pass
    results["Deterministic Plots"] = plot_count

    # Ignored
    init_time = [f for f in all_det if "init_time" in f]
    lead_time = [f for f in all_det if "per_lead_time" in f]
    if init_time:
        results["Deterministic Init Time (Ignored)"] = len(init_time)
    if lead_time:
        results["Deterministic Lead Time (Ignored)"] = len(lead_time)

    _report_checklist("deterministic_metrics", results)

    # Report common files found (to match panel counts)
    det_csv = _common_files(models, "deterministic/deterministic_metrics*.csv")
    if det_csv:
        _print_file_list(f"Found {len(det_csv)} common deterministic metric files", det_csv)

    # Deterministic metrics
    dst_det = _ensure_dir(out_root / "deterministic")
    frames: list[pd.DataFrame] = []
    frames_std: list[pd.DataFrame] = []
    for lab, m in zip(labels, models, strict=False):
        # New schema: deterministic_metrics*.csv (may have qualifiers/time tokens)
        candidates = sorted((m / "deterministic").glob("deterministic_metrics*.csv"))
        # Prefer exact base (no averaged/time tokens) for primary combined table
        f = next(
            (
                cand
                for cand in candidates
                if "per_level" not in cand.name
                and (
                    cand.name.endswith("ensmean.csv")
                    or cand.name.endswith("ensnone.csv")
                    or cand.name.endswith("enspooled.csv")
                    or cand.name.endswith("ensprob.csv")
                    or (
                        "ens" in cand.name
                        and cand.name.split("_")[-1].startswith("ens")
                        and cand.name.split("_")[-1].replace("ens", "").split(".")[0].isdigit()
                    )
                )
            ),
            None,
        )
        if f is not None and f.is_file():
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
        fstd = next(
            (
                cand
                for cand in (m / "deterministic").glob("deterministic_metrics_standardized*.csv")
                if "per_level" not in cand.name
                and (
                    cand.name.endswith("ensmean.csv")
                    or cand.name.endswith("ensnone.csv")
                    or cand.name.endswith("enspooled.csv")
                    or cand.name.endswith("ensprob.csv")
                    or (
                        "ens" in cand.name
                        and cand.name.split("_")[-1].startswith("ens")
                        and cand.name.split("_")[-1].replace("ens", "").split(".")[0].isdigit()
                    )
                )
            ),
            None,
        )
        if fstd is not None and fstd.is_file():
            df = pd.read_csv(fstd)
            if "variable" not in df.columns:
                if "Unnamed: 0" in df.columns:
                    df = df.rename(columns={"Unnamed: 0": "variable"})
                else:
                    first = df.columns[0]
                    df = df.rename(columns={first: "variable"})
            df.insert(0, "model", lab)
            frames_std.append(df)

    # Per-level metrics (deterministic)
    frames_lvl: list[pd.DataFrame] = []
    frames_lvl_std: list[pd.DataFrame] = []
    for lab, m in zip(labels, models, strict=False):
        # Regular per-level
        candidates_lvl = sorted((m / "deterministic").glob("deterministic_metrics*per_level*.csv"))
        f_lvl = next(
            (
                cand
                for cand in candidates_lvl
                if "standardized" not in cand.name
                and (
                    cand.name.endswith("ensmean.csv")
                    or cand.name.endswith("ensnone.csv")
                    or cand.name.endswith("enspooled.csv")
                    or cand.name.endswith("ensprob.csv")
                    or (
                        "ens" in cand.name
                        and cand.name.split("_")[-1].startswith("ens")
                        and cand.name.split("_")[-1].replace("ens", "").split(".")[0].isdigit()
                    )
                )
            ),
            None,
        )
        if f_lvl and f_lvl.is_file():
            df = pd.read_csv(f_lvl)
            if "variable" not in df.columns:
                if "Unnamed: 0" in df.columns:
                    df = df.rename(columns={"Unnamed: 0": "variable"})
                else:
                    first = df.columns[0]
                    df = df.rename(columns={first: "variable"})
            df.insert(0, "model", lab)
            frames_lvl.append(df)

        # Standardized per-level
        f_lvl_std = next(
            (
                cand
                for cand in candidates_lvl
                if "standardized" in cand.name
                and (
                    cand.name.endswith("ensmean.csv")
                    or cand.name.endswith("ensnone.csv")
                    or cand.name.endswith("enspooled.csv")
                    or cand.name.endswith("ensprob.csv")
                    or (
                        "ens" in cand.name
                        and cand.name.split("_")[-1].startswith("ens")
                        and cand.name.split("_")[-1].replace("ens", "").split(".")[0].isdigit()
                    )
                )
            ),
            None,
        )
        if f_lvl_std and f_lvl_std.is_file():
            df = pd.read_csv(f_lvl_std)
            if "variable" not in df.columns:
                if "Unnamed: 0" in df.columns:
                    df = df.rename(columns={"Unnamed: 0": "variable"})
                else:
                    first = df.columns[0]
                    df = df.rename(columns={first: "variable"})
            df.insert(0, "model", lab)
            frames_lvl_std.append(df)

    if frames:
        comb = pd.concat(frames, ignore_index=True)
        if comb["model"].nunique() >= 2:
            comb.to_csv(dst_det / "metrics_combined.csv", index=False)
            c.success(f"Saved {(dst_det / 'metrics_combined.csv').relative_to(out_root)}")

    if frames_lvl:
        comb_lvl = pd.concat(frames_lvl, ignore_index=True)
        if comb_lvl["model"].nunique() >= 2:
            comb_lvl.to_csv(dst_det / "metrics_per_level_combined.csv", index=False)
            c.success(f"Saved {(dst_det / 'metrics_per_level_combined.csv').relative_to(out_root)}")

    if frames_lvl_std:
        comb_lvl_std = pd.concat(frames_lvl_std, ignore_index=True)
        if comb_lvl_std["model"].nunique() >= 2:
            out_path = dst_det / "metrics_standardized_per_level_combined.csv"
            comb_lvl_std.to_csv(out_path, index=False)
            c.success(f"Saved {out_path.relative_to(out_root)}")

    if frames:
        # Optional: simple bar plots; coerce to numeric and handle all-NaN gracefully
        excluded = {
            "variable",
            "model",
            "level",
            "lead_time",
            "init_time",
            "valid_time",
            "Unnamed: 0",
            "source_file",
            "member",
            "threshold",
            "",
        }
        metric_cols = [c for c in comb.columns if c not in excluded]

        for metric in metric_cols:
            if metric in comb.columns:
                tmp = comb.copy()
                tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
                pivot = tmp.pivot(index="variable", columns="model", values=metric)
                # Drop rows/columns that are entirely NaN
                pivot = pivot.dropna(axis=0, how="all").dropna(axis=1, how="all")
                out_png = dst_det / f"{metric}_compare.png"
                if pivot.empty or pivot.notna().sum().sum() == 0 or pivot.shape[1] < 2:
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
                    c.info(f"[intercompare] saved placeholder {out_png}")
                    continue
                ax = pivot.plot(kind="bar", figsize=(12, 6))
                ax.set_title(f"{metric} by variable and model")
                ax.set_ylabel(metric)
                plt.tight_layout()
                plt.savefig(out_png, bbox_inches="tight", dpi=200)
                c.success(f"Saved {out_png.relative_to(out_root)}")
                plt.close()

    if frames_std:
        combs = pd.concat(frames_std, ignore_index=True)
        if combs["model"].nunique() >= 2:
            combs.to_csv(dst_det / "metrics_standardized_combined.csv", index=False)
            c.success(
                f"Saved {(dst_det / 'metrics_standardized_combined.csv').relative_to(out_root)}"
            )


def intercompare_ets_metrics(models: list[Path], labels: list[str], out_root: Path) -> None:
    # Availability report
    results = {}
    all_ets = _common_files(models, "ets/ets_metrics*.csv")

    avg = [f for f in all_ets if "averaged" in f]
    lvl = [f for f in all_ets if "per_level" in f]

    results["ETS Averaged"] = 1 if avg else 0
    results["ETS Per Level"] = 1 if lvl else 0

    if len(avg) > 1:
        results["ETS Averaged (Ignored)"] = len(avg) - 1
    if len(lvl) > 1:
        results["ETS Per Level (Ignored)"] = len(lvl) - 1

    # Ignored
    init_time = [f for f in all_ets if "init_time" in f]
    lead_time = [f for f in all_ets if "per_lead_time" in f]
    if init_time:
        results["ETS Init Time (Ignored)"] = len(init_time)
    if lead_time:
        results["ETS Lead Time (Ignored)"] = len(lead_time)

    _report_checklist("ets_metrics", results)

    # Report common files found
    ets_csv = _common_files(models, "ets/ets_metrics*.csv")
    if ets_csv:
        _print_file_list(f"Found {len(ets_csv)} common ETS metric files", ets_csv)

    # Actually process and save ETS metrics (was missing!)
    dst_ets = _ensure_dir(out_root / "ets")
    frames: list[pd.DataFrame] = []
    frames_lvl: list[pd.DataFrame] = []

    for lab, m in zip(labels, models, strict=False):
        # Averaged
        candidates = sorted((m / "ets").glob("ets_metrics*.csv"))
        f = next(
            (
                c
                for c in candidates
                if "per_level" not in c.name
                and "init_time" not in c.name
                and "per_lead_time" not in c.name
            ),
            None,
        )
        # Fallback: try to find one that is averaged but might have init/lead tokens
        # if that's the only one
        if f is None:
            f = next((cand for cand in candidates if "averaged" in cand.name), None)

        if f is not None and f.is_file():
            try:
                df = pd.read_csv(f)
                if "variable" not in df.columns and "Unnamed: 0" in df.columns:
                    df = df.rename(columns={"Unnamed: 0": "variable"})
                df.insert(0, "model", lab)
                frames.append(df)
            except Exception:
                pass

        # Per Level
        f_lvl = next((c for c in candidates if "per_level" in c.name), None)
        if f_lvl is not None and f_lvl.is_file():
            try:
                df = pd.read_csv(f_lvl)
                if "variable" not in df.columns and "Unnamed: 0" in df.columns:
                    df = df.rename(columns={"Unnamed: 0": "variable"})
                df.insert(0, "model", lab)
                frames_lvl.append(df)
            except Exception:
                pass

    if frames:
        comb = pd.concat(frames, ignore_index=True)
        if comb["model"].nunique() >= 2:
            out_csv = dst_ets / "ets_metrics_combined.csv"
            comb.to_csv(out_csv, index=False)
            c.success(f"Saved {out_csv.relative_to(out_root)}")

    if frames_lvl:
        comb_lvl = pd.concat(frames_lvl, ignore_index=True)
        if comb_lvl["model"].nunique() >= 2:
            out_csv = dst_ets / "ets_metrics_per_level_combined.csv"
            comb_lvl.to_csv(out_csv, index=False)
            c.success(f"Saved {out_csv.relative_to(out_root)}")


def intercompare_probabilistic(
    models: list[Path],
    labels: list[str],
    out_root: Path,
    max_crps_map_panels: int = 4,
) -> None:
    src_rel = Path("probabilistic")
    dst = _ensure_dir(out_root / "probabilistic")

    # Availability report (single panel across all probabilistic artifacts)
    per_model_accum: list[set[str]] = [set() for _ in models]
    union_total: set[str] = set()
    for pattern in (
        "probabilistic/pit_hist_*.npz",
        "probabilistic/crps_map_*.npz",
        "probabilistic/prob_metrics_spatial*.nc",
        "probabilistic/prob_metrics_temporal*.nc",
    ):
        per_model, inter, uni = _scan_model_sets(models, pattern)
        union_total |= uni
        for i, s in enumerate(per_model):
            per_model_accum[i] |= s
    _report_missing("probabilistic", models, labels, per_model_accum, union_total)

    # Availability report
    results = {}

    # CSVs
    all_csv = _common_files(models, "probabilistic/*.csv")
    crps_sum = [f for f in all_csv if "crps_summary" in f]
    crps_sum_avg = [f for f in crps_sum if "per_level" not in f]
    crps_sum_lvl = [f for f in crps_sum if "per_level" in f]

    spread = [f for f in all_csv if "spread_skill_ratio" in f]
    crps_ens = [f for f in all_csv if "crps_ensemble" in f]

    results["CRPS Summary"] = 1 if crps_sum_avg else 0
    results["CRPS Summary Per Level"] = 1 if crps_sum_lvl else 0
    results["Spread Skill Ratio"] = 1 if spread else 0
    results["CRPS Ensemble"] = 1 if crps_ens else 0

    # PIT Histograms
    pit = _common_files(models, "probabilistic/pit_hist_*.npz")
    results["PIT Histograms"] = len(pit)

    # CRPS Maps
    maps = _common_files(models, "probabilistic/crps_map_*.npz")
    processed_maps = min(len(maps), max_crps_map_panels)
    ignored_maps = max(0, len(maps) - max_crps_map_panels)
    results["CRPS Maps"] = processed_maps
    if ignored_maps > 0:
        results["CRPS Maps (Ignored)"] = ignored_maps

    # NC files
    spatial = _common_files(models, "probabilistic/prob_metrics_spatial*.nc")
    temporal = _common_files(models, "probabilistic/prob_metrics_temporal*.nc")

    results["Spatial Metrics"] = 1 if spatial else 0
    results["Temporal Metrics"] = 1 if temporal else 0

    _report_checklist("probabilistic", results)

    # 1) Combine CRPS summary (non-WBX) across models
    frames_crps: list[pd.DataFrame] = []
    frames_crps_lvl: list[pd.DataFrame] = []
    for lab, m in zip(labels, models, strict=False):
        for f in (m / src_rel).glob("crps_summary*.csv"):
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            df.insert(0, "model", lab)
            if "per_level" in f.name:
                frames_crps_lvl.append(df)
            else:
                frames_crps.append(df)
    if frames_crps:
        comb = pd.concat(frames_crps, ignore_index=True)
        if comb["model"].nunique() >= 2:
            comb.to_csv(dst / "crps_summary_combined.csv", index=False)
    if frames_crps_lvl:
        comb_lvl = pd.concat(frames_crps_lvl, ignore_index=True)
        if comb_lvl["model"].nunique() >= 2:
            comb_lvl.to_csv(dst / "crps_summary_per_level_combined.csv", index=False)

    # 2) Combine WBX CSV summaries if present
    for basename, outname in (
        ("spread_skill_ratio.csv", "spread_skill_ratio_combined.csv"),
        ("crps_ensemble.csv", "crps_ensemble_combined.csv"),
    ):
        frames: list[pd.DataFrame] = []
        for lab, m in zip(labels, models, strict=False):
            f = m / src_rel / basename
            if f.is_file():
                df = pd.read_csv(f)
                df.insert(0, "model", lab)
                frames.append(df)
        if frames:
            dfc = pd.concat(frames, ignore_index=True)
            if dfc["model"].nunique() >= 2:
                dfc.to_csv(dst / outname, index=False)

    # 3) Overlay PIT histograms by variable
    common_pit = _common_files(models, str(src_rel / "pit_hist_*.npz"))
    if common_pit:
        _print_file_list(f"Found {len(common_pit)} common PIT histogram files", common_pit)
    colors = sns.color_palette("tab10", n_colors=len(models))
    for base in common_pit:
        payloads = [_load_npz(m / src_rel / base) for m in models]
        fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
        # Uniform reference line at y=1
        ax.axhline(1.0, color="brown", linestyle="--", linewidth=1, label="Uniform")
        contributed = 0
        for i, (lab, pay) in enumerate(zip(labels, payloads, strict=False)):
            counts = pay.get("counts")
            edges = pay.get("edges")
            if counts is None or edges is None:
                continue
            contributed += 1
            _plot_hist_counts(ax, edges, counts, label=lab, color=colors[i])
        if contributed < 2:
            plt.close(fig)
            continue
        # Extract variable token from standardized filename: pit_hist_<var>_..._ens*.npz
        stem = base[:-4] if base.endswith(".npz") else base
        var = stem
        if stem.startswith("pit_hist_"):
            rest = stem[len("pit_hist_") :]
            # trim ensemble token and optional time tokens
            if "_ens" in rest:
                rest = rest.split("_ens", 1)[0]
            # remove optional _init... and _lead... segments if present
            for tok in ("_init", "_lead"):
                if tok in rest:
                    rest = rest.split(tok, 1)[0]
            var = rest
        ax.set_title(f"PIT histogram — {var}")
        ax.set_xlabel("PIT value")
        ax.set_ylabel("Density")
        ax.legend()
        plt.tight_layout()
        out_png = dst / base.replace(".npz", "_compare.png")
        plt.savefig(out_png, bbox_inches="tight", dpi=200)
        c.success(f"Saved {out_png.relative_to(out_root)}")
        plt.close(fig)

    # 4) Panel CRPS maps from saved NPZ (if available)
    common_crps_map_npz = _common_files(models, str(src_rel / "crps_map_*.npz"))
    if common_crps_map_npz:
        _print_file_list(
            f"Found {len(common_crps_map_npz)} common CRPS map files", common_crps_map_npz
        )

    # Report other common probabilistic files (CSVs, NCs)
    prob_csv = _common_files(models, str(src_rel / "*.csv"))
    if prob_csv:
        _print_file_list(f"Found {len(prob_csv)} common probabilistic CSV files", prob_csv)

    prob_nc = _common_files(models, str(src_rel / "*.nc"))
    if prob_nc:
        _print_file_list(f"Found {len(prob_nc)} common probabilistic NC files", prob_nc)

    for base in common_crps_map_npz[:max_crps_map_panels]:
        payloads = [_load_npz(m / src_rel / base) for m in models]
        # Compute global vmin/vmax across models for consistent color scale
        arrays = [p.get("crps") for p in payloads]
        # Require at least two models with arrays
        if sum(1 for a in arrays if a is not None) < 2:
            continue
        vmin = float(np.nanmin([np.nanmin(a) for a in arrays if a is not None]))
        vmax = float(np.nanmax([np.nanmax(a) for a in arrays if a is not None]))
        lats = payloads[0].get("latitude")
        lons = payloads[0].get("longitude")
        ncols = len(models)
        fig, axes = plt.subplots(
            1,
            ncols,
            figsize=(6 * ncols, 4),
            dpi=160,
            subplot_kw={"projection": ccrs.PlateCarree()},
            constrained_layout=True,
        )
        if ncols == 1:
            axes = [axes]
        for ax, lab, arr in zip(axes, labels, arrays, strict=False):
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
        # Use a constrained-layout-friendly colorbar spanning all axes
        cbar = fig.colorbar(
            mesh,
            ax=axes if isinstance(axes, (list | np.ndarray)) else [axes],
            orientation="horizontal",
            fraction=0.05,
            pad=0.08,
        )
        with contextlib.suppress(Exception):
            cbar.set_label("CRPS")
        # No tight_layout here; constrained_layout handles spacing
        out_png = dst / base.replace(".npz", "_compare.png")
        plt.savefig(out_png, bbox_inches="tight", dpi=200)
        c.success(f"Saved {out_png.relative_to(out_root)}")
        plt.close(fig)

    # 5) Combine spatial/temporal WBX NetCDF aggregates into tidy CSVs and simple plots
    # Spatial aggregates
    spatial_rows: list[pd.DataFrame] = []
    for lab, m in zip(labels, models, strict=False):
        # New naming: prob_metrics_spatial*.nc (fallback to legacy probabilistic_metrics_spatial.nc)
        nc_candidates = list((m / src_rel).glob("prob_metrics_spatial*.nc"))
        if not nc_candidates:
            legacy = m / src_rel / "probabilistic_metrics_spatial.nc"
            nc_candidates = [legacy] if legacy.is_file() else []
        for f in nc_candidates:
            try:
                ds = xr.open_dataset(f)
                df = ds.to_dataframe().reset_index()
                # Keep only metric variables we know (columns like 'CRPS.<var>' or 'SSR.<var>')
                value_cols = [
                    c
                    for c in df.columns
                    if isinstance(c, str) and (c.startswith("CRPS.") or c.startswith("SSR."))
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
        # Only save/output if at least two models contributed
        if spatial_df["model"].nunique() >= 2:
            out_csv = dst / "spatial_metrics_combined.csv"
            spatial_df.to_csv(out_csv, index=False)
            c.success(f"Saved {out_csv.relative_to(out_root)}")
            # Simple plot: if a region-like column exists, average across
            # variables and plot by region
            region_col = None
            # Prefer canonical 'region' column; else pick the first object-type column among dims
            cand_cols = [c for c in spatial_df.columns if c.lower() == "region"]
            if cand_cols:
                region_col = cand_cols[0]
            else:
                obj_cols = [
                    c
                    for c in spatial_df.columns
                    if spatial_df[c].dtype == object and c not in ("metric", "variable", "model")
                ]
                region_col = obj_cols[0] if obj_cols else None
            if region_col:
                for metric in sorted(spatial_df["metric"].unique()):
                    tmp = spatial_df[spatial_df["metric"] == metric].copy()
                    tmp = tmp.groupby([region_col, "model"], as_index=False)["value"].mean()
                    pivot = tmp.pivot(
                        index=region_col, columns="model", values="value"
                    ).sort_index()
                    if not pivot.empty and pivot.notna().sum().sum() > 0 and pivot.shape[1] >= 2:
                        ax = pivot.plot(kind="bar", figsize=(12, 6))
                        ax.set_title(f"{metric} (spatial aggregates)")
                        ax.set_ylabel(metric)
                        plt.tight_layout()
                        out_png = dst / f"spatial_{metric}_compare.png"
                        plt.savefig(out_png, bbox_inches="tight", dpi=200)
                        c.success(f"Saved {out_png.relative_to(out_root)}")
                        plt.close()

    # Temporal aggregates
    temporal_rows: list[pd.DataFrame] = []
    for lab, m in zip(labels, models, strict=False):
        nc_candidates = list((m / src_rel).glob("prob_metrics_temporal*.nc"))
        if not nc_candidates:
            legacy = m / src_rel / "probabilistic_metrics_temporal.nc"
            nc_candidates = [legacy] if legacy.is_file() else []
        for f in nc_candidates:
            try:
                ds = xr.open_dataset(f)
                df = ds.to_dataframe().reset_index()
                value_cols = [
                    c
                    for c in df.columns
                    if isinstance(c, str) and (c.startswith("CRPS.") or c.startswith("SSR."))
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
        # Only save/output if at least two models contributed
        if temporal_df["model"].nunique() >= 2:
            out_csv = dst / "temporal_metrics_combined.csv"
            temporal_df.to_csv(out_csv, index=False)
            c.success(f"Saved {out_csv.relative_to(out_root)}")
            # Pick a time-bin column to plot if present (e.g., 'season'); else skip plotting
            timebin_col = None
            pref_cols = ["season", "month", "time_bin"]
            for col in pref_cols:
                if col in temporal_df.columns:
                    timebin_col = col
                    break
            if timebin_col is None:
                # Try any object-like dim besides variable/metric/model
                obj_cols = [
                    c
                    for c in temporal_df.columns
                    if temporal_df[c].dtype == object and c not in ("metric", "variable", "model")
                ]
                timebin_col = obj_cols[0] if obj_cols else None
            if timebin_col:
                for metric in sorted(temporal_df["metric"].unique()):
                    tmp = temporal_df[temporal_df["metric"] == metric].copy()
                    tmp = tmp.groupby([timebin_col, "model"], as_index=False)["value"].mean()
                    # Ensure categorical ordering if seasons
                    if timebin_col == "season":
                        order = ["DJF", "MAM", "JJA", "SON"]
                        tmp[timebin_col] = pd.Categorical(
                            tmp[timebin_col], categories=order, ordered=True
                        )
                    piv = tmp.pivot(index=timebin_col, columns="model", values="value").sort_index()
                    if not piv.empty and piv.notna().sum().sum() > 0 and piv.shape[1] >= 2:
                        ax = piv.plot(kind="line", marker="o", figsize=(10, 4))
                        ax.set_title(f"{metric} (temporal aggregates)")
                        ax.set_ylabel(metric)
                        ax.set_xlabel(timebin_col.capitalize())
                        plt.tight_layout()
                        out_png = dst / f"temporal_{metric}_compare.png"
                        plt.savefig(out_png, bbox_inches="tight", dpi=200)
                        c.success(f"Saved {out_png.relative_to(out_root)}")
                        plt.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SwissClim Evaluations — Intercomparison runner (YAML-configured)"
    )
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to intercomparison YAML config (models, labels, output_root, modules)",
    )
    return p


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def run_from_config(cfg: dict) -> None:
    # Resolve models
    model_strs = cfg.get("models") or []
    if not isinstance(model_strs, list) or len(model_strs) < 2:
        raise ValueError("config.models must be a list with at least two model directories")
    models = _as_paths(model_strs)
    # Labels
    labels = cfg.get("labels") or []
    if not labels or len(labels) != len(models):
        labels = [_model_label(p) for p in models]
    # Output root
    out_root = _ensure_dir(Path(cfg.get("output_root", "output/intercomparison")).resolve())
    c.info(f"Output directory: {out_root}")
    # Modules
    modules = cfg.get("modules") or ["spectra", "hist", "kde", "maps", "metrics", "prob", "vprof"]
    modules = [str(m).lower() for m in modules]
    # Other options
    max_map_panels = int(cfg.get("max_map_panels", 4))
    max_crps_map_panels = int(cfg.get("max_crps_map_panels", 4))

    # Light validation: warn on missing model dirs
    for m in models:
        if not m.exists():
            print(f"[intercompare] WARNING: model folder does not exist: {m}")

    mods = set(modules)
    if "spectra" in mods:
        intercompare_energy_spectra(models, labels, out_root)
    if "hist" in mods:
        intercompare_histograms(models, labels, out_root)
    if "kde" in mods:
        intercompare_wd_kde(models, labels, out_root)
    if "maps" in mods:
        intercompare_maps(models, labels, out_root, max_panels=max_map_panels)
    if "metrics" in mods:
        intercompare_deterministic_metrics(models, labels, out_root)
        intercompare_ets_metrics(models, labels, out_root)
    if "prob" in mods:
        intercompare_probabilistic(
            models, labels, out_root, max_crps_map_panels=max_crps_map_panels
        )
    if "vprof" in mods:
        intercompare_vertical_profiles(models, labels, out_root)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = _load_yaml(args.config)
    run_from_config(cfg)


if __name__ == "__main__":
    main()
