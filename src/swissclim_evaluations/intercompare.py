from __future__ import annotations

import argparse
import contextlib
import re
from collections.abc import Iterable
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from swissclim_evaluations.plots.energy_spectra import add_wavelength_axis

from .helpers import (
    COLOR_GROUND_TRUTH,
    extract_date_from_filename,
    format_level_label,
    format_level_token,
    format_variable_name,
)

# Global flag for quiet mode (can be overridden if needed)
quiet = False

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

    Returns a sorted list of basenames present in all model folders that match the pattern.
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
    (target) not expressly stored. The NPZ files only contain metric curves
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

        # Extract date info from filename if possible
        date_suffix = extract_date_from_filename(base)

        fig.suptitle(f"Vertical Profiles — {var} (NMAE %){date_suffix}", y=1.02)
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
    per_model, _, uni = _scan_model_sets(models, "energy_spectra/*_spectrum*.npz")
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
            title = f"Energy Spectra — {var}{format_level_label(level_raw)}"

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

                title_ratio = f"Energy Spectra Ratio — {var}{format_level_label(level_raw)}"
                ax_r.set_title(title_ratio)
                ax_r.grid(True, which="both", ls="--", alpha=0.4)
                ax_r.legend(frameon=False)
                ax_r.axhline(100, color="k", linestyle="--", lw=1.0, alpha=0.5)

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

    # 4) Plot Energy Spectra per Lead Time (from bundle NPZ)
    # Look for energy_spectrogram_*_bundle.npz
    bundles = _common_files(models, str(src_rel / "energy_spectrogram_*_bundle.npz"))
    if bundles:
        _print_file_list(f"Found {len(bundles)} common energy spectra bundles", bundles)

        for base in bundles:
            payloads = [_load_npz(m / src_rel / base) for m in models]

            # Check if all have necessary keys
            if not all(
                "energy_model" in p and "lead_hours" in p and "wavenumber" in p for p in payloads
            ):
                continue

            # Assume lead_hours and wavenumber are same for all models
            lead_hours = payloads[0]["lead_hours"]
            wavenumber = payloads[0]["wavenumber"]
            variable = payloads[0].get("variable", "var")

            # Iterate over lead times
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
                    em = pay["energy_model"]
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

        c.success(f"Saved per-lead energy spectra plots to {dst.relative_to(out_root)}")


def intercompare_ets_metrics(models: list[Path], labels: list[str], out_root: Path) -> None:
    """Combine ETS metrics from multiple models and plot vs lead time."""
    dst_ets = _ensure_dir(out_root / "ets")

    # Availability report
    per_model, _, uni = _scan_model_sets(models, "ets/ets_metrics_by_lead_wide.csv")
    _report_missing("ets_metrics", models, labels, per_model, uni)

    results = {}
    all_ets = _common_files(models, "ets/ets_metrics_by_lead_wide.csv")
    results["ETS Metrics (Wide)"] = 1 if all_ets else 0
    _report_checklist("ets_metrics", results)

    if not all_ets:
        c.warn("No common ETS wide metrics files found. Skipping plots.")
        return

    # Combine wide CSVs
    frames: list[pd.DataFrame] = []
    for lab, m in zip(labels, models, strict=False):
        f = m / "ets" / "ets_metrics_by_lead_wide.csv"
        if f.is_file():
            try:
                df = pd.read_csv(f)
                df.insert(0, "model", lab)
                frames.append(df)
            except Exception:
                pass

    if not frames:
        return

    combined = pd.concat(frames, ignore_index=True)
    if combined["model"].nunique() >= 2:
        out_csv = dst_ets / "ets_metrics_combined.csv"
        combined.to_csv(out_csv, index=False)
        c.success(f"Saved {out_csv.relative_to(out_root)}")

        # Plot ETS vs Lead Time per (Variable, Threshold)
        # Columns are: model, lead_time_hours, <var>_ETS <thresh>%
        # We need to identify metric columns
        meta_cols = {"model", "lead_time_hours", "Unnamed: 0"}
        metric_cols = [c for c in combined.columns if c not in meta_cols]

        # Group by variable and threshold
        # Expected format: "{var}_ETS {thresh}%"
        # We can parse this
        for col in metric_cols:
            if "_ETS " not in col:
                continue

            # Extract variable and threshold for title
            # This is a bit heuristic, assuming var doesn't contain "_ETS "
            try:
                var_part, thresh_part = col.split("_ETS ", 1)
            except ValueError:
                var_part = col
                thresh_part = ""

            pivot = combined.pivot(
                index="lead_time_hours", columns="model", values=col
            ).sort_index()

            if not pivot.empty and pivot.notna().sum().sum() > 0 and pivot.shape[1] >= 2:
                fig, ax = plt.subplots(figsize=(10, 6))
                pivot.plot(kind="line", ax=ax, marker="o", markersize=4)

                title = f"ETS vs Lead Time — {format_variable_name(var_part)}"
                if thresh_part:
                    title += f" ({thresh_part})"

                ax.set_title(title)
                ax.set_ylabel("ETS")
                ax.set_xlabel("Lead Time (h)")
                ax.grid(True, linestyle="--", alpha=0.6)
                plt.tight_layout()

                safe_col = col.replace(" ", "_").replace("%", "pct")
                out_png = dst_ets / f"ets_{safe_col}_compare.png"
                plt.savefig(out_png, bbox_inches="tight", dpi=200)
                c.success(f"Saved {out_png.relative_to(out_root)}")
                plt.close(fig)


def _clean_var_from_filename(filename: str, prefix: str = "") -> str:
    """Clean variable name from filename for plot titles."""
    stem = filename[:-4] if filename.endswith(".npz") else filename
    if prefix and stem.startswith(prefix):
        stem = stem[len(prefix) :]

    # Remove common suffixes/tokens
    for token in ["_global", "_latbands", "_combined"]:
        stem = stem.replace(token, "")

    # Remove ensemble token
    if "_ens" in stem:
        stem = stem.rsplit("_ens", 1)[0]

    # Remove init/lead time patterns
    # initYYYY-MM-DDTHH-YYYY-MM-DDTHH
    stem = re.sub(r"_init\d{4}-?\d{2}-?\d{2}T\d{2}-\d{4}-?\d{2}-?\d{2}T\d{2}", "", stem)
    # leadXXXh-YYYh
    stem = re.sub(r"_lead\d+h-\d+h", "", stem)

    return format_variable_name(stem)


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
    per_model, _, uni = _scan_model_sets(models, "histograms/hist_*latbands_combined*.npz")
    _report_missing("histograms", models, labels, per_model, uni)

    results = {}
    # Plots: 1-to-1
    plots = _common_files(models, "histograms/hist_*latbands_combined*.npz")
    results["Histograms (Latbands)"] = len(plots)

    # Check for ignored
    all_hist = _common_files(models, "histograms/hist_*.npz")
    ignored = [f for f in all_hist if "latbands_combined" not in f and "global" not in f]
    if ignored:
        results["Other Histograms (Ignored)"] = len(ignored)

    common = _common_files(models, str(src_rel / "hist_*latbands_combined*.npz"))
    if common:
        _print_file_list(f"Found {len(common)} common latbands histogram files", common)

    colors = sns.color_palette("tab10", n_colors=len(models))

    # --- Global Histograms ---
    per_model_g, inter_g, uni_g = _scan_model_sets(models, "histograms/hist_*global*.npz")
    _report_missing("histograms (global)", models, labels, per_model_g, uni_g)
    common_g = _common_files(models, str(src_rel / "hist_*global*.npz"))

    results["Histograms (Global)"] = len(common_g)
    _report_checklist("histograms", results)

    if not common and not common_g:
        c.warn("No common histogram files found (latbands or global). Skipping plots.")
        return

    for base in common_g:
        payloads = [_load_npz(m / src_rel / base) for m in models]
        fig, ax = plt.subplots(figsize=(10, 6), dpi=160)

        # Target (from first model)
        counts_ds = payloads[0]["counts_ds"]
        bins_ds = payloads[0]["bins"]
        _plot_hist_counts(ax, bins_ds, counts_ds, label="Target", color=COLOR_GROUND_TRUTH)

        # Models
        for i, (lab, pay) in enumerate(zip(labels, payloads, strict=False)):
            counts_ml = pay["counts_ml"]
            bins_ml = pay["bins"]
            _plot_hist_counts(ax, bins_ml, counts_ml, label=lab, color=colors[i])

        var = _clean_var_from_filename(base, prefix="hist_")
        date_suffix = extract_date_from_filename(base)
        ax.set_title(f"Global Histogram — {var}{date_suffix}")
        ax.set_ylabel("Frequency (log)")
        ax.set_yscale("log")
        ax.legend(frameon=False)
        ax.grid(True, which="both", ls="--", alpha=0.4)

        out_png = dst / base.replace(".npz", "_compare.png")
        plt.tight_layout()
        plt.savefig(out_png, bbox_inches="tight", dpi=200)
        plt.close(fig)
        c.success(f"Saved {out_png.relative_to(out_root)}")

    # --- Latitude Bands Histograms ---
    # Availability report (always display)
    per_model, _, uni = _scan_model_sets(models, "histograms/hist_*latbands*.npz")
    # Filter out global histograms from this scan
    per_model = [{f for f in s if "global" not in f} for s in per_model]
    uni = {f for f in uni if "global" not in f}

    _report_missing("histograms (latbands)", models, labels, per_model, uni)
    common = _common_files(models, str(src_rel / "hist_*latbands*.npz"))
    common = [f for f in common if "global" not in f]

    if common:
        for base in common:
            payloads = [_load_npz(m / src_rel / base) for m in models]
            units = payloads[0].get("units")
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

                _plot_hist_counts(ax, bins_ds, counts_ds, label="Target", color=COLOR_GROUND_TRUTH)

                # Plot each prediction model
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
                if units:
                    ax.set_xlabel(str(units))

            # Left column: northern hemisphere bands
            for j in range(n_rows):
                ax = axs[j, 0]
                ds_ml_pairs = payloads[0]["pos_counts"][j]
                counts_ds = ds_ml_pairs[0]
                bins_ds = payloads[0]["pos_bins"][j]

                _plot_hist_counts(ax, bins_ds, counts_ds, label="Target", color=COLOR_GROUND_TRUTH)

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
                if units:
                    ax.set_xlabel(str(units))

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
            var = _clean_var_from_filename(base, prefix="hist_")
            date_suffix = extract_date_from_filename(base)

            fig.suptitle(f"Distributions by Latitude Bands — {var}{date_suffix}", y=1.02)
            out_png = dst / base.replace(".npz", "_compare.png")
            plt.savefig(out_png, bbox_inches="tight", dpi=200)
            c.success(f"Saved {out_png.relative_to(out_root)}")
            plt.close(fig)


def intercompare_wd_kde(models: list[Path], labels: list[str], out_root: Path) -> None:
    src_rel = Path("wd_kde")
    dst = _ensure_dir(out_root / "wd_kde")
    colors = sns.color_palette("tab10", n_colors=len(models))

    # --- Global KDE ---
    per_model_g, inter_g, uni_g = _scan_model_sets(models, "wd_kde/wd_kde_*global*.npz")
    _report_missing("wd_kde (global)", models, labels, per_model_g, uni_g)
    common_g = _common_files(models, str(src_rel / "wd_kde_*global*.npz"))

    for base in common_g:
        payloads = [_load_npz(m / src_rel / base) for m in models]
        units = payloads[0].get("units")
        fig, ax = plt.subplots(figsize=(10, 6), dpi=160)

        # Target (from first model)
        x_ds = payloads[0]["x"]
        kde_ds = payloads[0]["kde_ds"]
        ax.plot(x_ds, kde_ds, color=COLOR_GROUND_TRUTH, lw=2.0, label="Target")

        # Models
        for i, (lab, pay) in enumerate(zip(labels, payloads, strict=False)):
            x_ml = pay["x"]
            kde_ml = pay["kde_ml"]
            ax.plot(x_ml, kde_ml, color=colors[i], label=lab)

        var = _clean_var_from_filename(base, prefix="wd_kde_")
        date_suffix = extract_date_from_filename(base)
        ax.set_title(f"Global Normalized KDE — {var}{date_suffix}")
        if units:
            ax.set_xlabel(str(units))
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
        units = payloads[0].get("units")
        # Assume each payload carries arrays of object dtype per band
        pos_x0 = payloads[0]["pos_x"]
        n_rows = len(pos_x0)
        fig, axs = plt.subplots(n_rows, 2, figsize=(16, 3 * n_rows), dpi=160)
        # South (right)
        for j in range(n_rows):
            ax = axs[j, 1]
            x_ds = payloads[0]["neg_x"][j]
            kde_ds = payloads[0]["neg_kde_ds"][j]
            ax.plot(x_ds, kde_ds, color=COLOR_GROUND_TRUTH, lw=2.0, label="Target")

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
            if units:
                ax.set_xlabel(str(units))

        # North (left)
        for j in range(n_rows):
            ax = axs[j, 0]
            x_ds = payloads[0]["pos_x"][j]
            kde_ds = payloads[0]["pos_kde_ds"][j]
            ax.plot(x_ds, kde_ds, color=COLOR_GROUND_TRUTH, lw=2.0, label="Target")

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
            if units:
                ax.set_xlabel(str(units))

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
        var = _clean_var_from_filename(base, prefix="wd_kde_")
        date_suffix = extract_date_from_filename(base)

        fig.suptitle(f"Normalized KDE by Latitude Bands — {var}{date_suffix}", y=1.02)
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
    per_model, _, uni = _scan_model_sets(models, "maps/map_*.npz")
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
        target = payloads[0].get("target")
        if target is None:
            target = payloads[0].get("nwp")

        predictions = []
        for p in payloads:
            val = p.get("prediction")
            if val is None:
                val = p.get("ml")
            predictions.append(val)

        if any(x is None for x in predictions) or target is None:
            continue
        lats = payloads[0].get("latitude")
        lons = payloads[0].get("longitude")
        var_name = payloads[0].get("variable")
        if not var_name and key.startswith("map_"):
            # Fallback: try to extract variable from filename key
            # key format: map_<var>_init... or map_<var>_ens...
            # Remove 'map_' prefix
            rest = key[4:]
            # Split by known tokens
            for token in ("_init", "_lead", "_ens", "_level"):
                if token in rest:
                    rest = rest.split(token, 1)[0]
                    break
            var_name = rest

        units = payloads[0].get("units")
        if lats is None or lons is None:
            continue

        def _is_3d(arr: np.ndarray) -> bool:
            return isinstance(arr, np.ndarray) and arr.ndim == 3

        n_levels = target.shape[0] if _is_3d(target) else 1
        if any(
            (_is_3d(target) and (not isinstance(m, np.ndarray) or m.ndim != 3))
            or ((not _is_3d(target)) and (not isinstance(m, np.ndarray) or m.ndim != 2))
            for m in predictions
        ):
            c.warn(f"maps: shape mismatch for {key}; skipping")
            continue
        level_vals = payloads[0].get("level")
        for lvl in range(n_levels):
            target_slice = target[lvl] if n_levels > 1 else target
            pred_slices = [
                m[lvl] if n_levels > 1 else m for m in predictions if isinstance(m, np.ndarray)
            ]
            try:
                vmin = float(
                    np.nanmin([np.nanmin(target_slice)] + [np.nanmin(x) for x in pred_slices])
                )
                vmax = float(
                    np.nanmax([np.nanmax(target_slice)] + [np.nanmax(x) for x in pred_slices])
                )
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
                target_slice,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree(),
            )
            axes[0].coastlines(linewidth=0.5)
            # Extract date info from filename if possible
            date_suffix = extract_date_from_filename(key)

            title_base = "Target"
            if var_name:
                title_base = f"{format_variable_name(str(var_name))} — {title_base}{date_suffix}"
            if n_levels > 1:
                if isinstance(level_vals, np.ndarray) and len(level_vals) == n_levels:
                    level_token = format_level_token(level_vals[lvl])
                    title_base += f" (level {level_token})"
                else:
                    title_token = format_level_token(lvl)
                    title_base += f" (level {title_token})"
            axes[0].set_title(title_base)
            for ax, lab, pred_slice in zip(axes[1:], labels, pred_slices, strict=False):
                ax.pcolormesh(
                    lons,
                    lats,
                    pred_slice,
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                    transform=ccrs.PlateCarree(),
                )
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
                cbar.set_label(str(units) if units else "Value")
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
    per_model, _, uni = _scan_model_sets(models, "deterministic/deterministic_metrics*.csv")
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
            frames_std.append(df)

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
                ax.set_xlabel("")
                plt.xticks(rotation=45, ha="right")
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

    # 2) Combine per-lead-time CSV metrics (e.g. RMSE vs lead_time)
    temporal_rows: list[dict] = []
    for lab, m in zip(labels, models, strict=False):
        # Find all per-lead CSVs
        csv_files = list((m / "deterministic").glob("deterministic_metrics*per_lead_time*.csv"))
        csv_files.extend((m / "deterministic").glob("deterministic_metrics*by_lead_long*.csv"))

        for f in csv_files:
            try:
                df = pd.read_csv(f)

                excluded = {
                    "variable",
                    "model",
                    "level",
                    "lead_time",
                    "lead_time_hours",
                    "init_time",
                    "valid_time",
                    "Unnamed: 0",
                    "source_file",
                    "member",
                    "",
                }
                metric_cols = [c for c in df.columns if c not in excluded]

                if not metric_cols:
                    continue

                lt_col = "lead_time_hours" if "lead_time_hours" in df.columns else "lead_time"
                if lt_col not in df.columns:
                    continue

                df["model"] = lab
                df = df.rename(columns={lt_col: "lead_time"})

                melted = df.melt(
                    id_vars=["model", "variable", "lead_time"],
                    value_vars=metric_cols,
                    var_name="metric",
                    value_name="value",
                )

                temporal_rows.extend(melted.to_dict("records"))
            except Exception:
                pass

    if temporal_rows:
        temporal_df = pd.DataFrame(temporal_rows)
        if len(temporal_df) > 0 and temporal_df["model"].nunique() >= 2:
            out_csv = dst_det / "temporal_metrics_combined.csv"
            temporal_df.to_csv(out_csv, index=False)
            c.success(f"Saved {out_csv.relative_to(out_root)}")

            pairs = temporal_df[["metric", "variable"]].drop_duplicates().values

            for metric, variable in pairs:
                subset = temporal_df[
                    (temporal_df["metric"] == metric) & (temporal_df["variable"] == variable)
                ].copy()

                pivot = subset.pivot(
                    index="lead_time", columns="model", values="value"
                ).sort_index()

                if not pivot.empty and pivot.notna().sum().sum() > 0 and pivot.shape[1] >= 2:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    pivot.plot(kind="line", ax=ax, marker="o", markersize=4)

                    ax.set_title(f"{metric} vs Lead Time — {format_variable_name(variable)}")
                    ax.set_ylabel(metric)
                    ax.set_xlabel("Lead Time (h)")
                    ax.grid(True, linestyle="--", alpha=0.6)
                    plt.tight_layout()

                    out_png = dst_det / f"temporal_{metric}_{variable}_compare.png"
                    plt.savefig(out_png, bbox_inches="tight", dpi=200)
                    c.success(f"Saved {out_png.relative_to(out_root)}")
                    plt.close(fig)


def intercompare_probabilistic(
    models: list[Path], labels: list[str], out_root: Path, max_crps_map_panels: int = 4
) -> None:
    """Combine Probabilistic metrics (CRPS, Spread/Error) from multiple models."""
    dst_prob = _ensure_dir(out_root / "probabilistic")

    # Availability report
    per_model, _, uni = _scan_model_sets(models, "probabilistic/crps_summary*.csv")
    _report_missing("probabilistic_metrics", models, labels, per_model, uni)

    results = {}
    all_prob = _common_files(models, "probabilistic/crps_summary*.csv")
    results["Probabilistic Metrics"] = 1 if all_prob else 0

    # Ignored
    init_time = [f for f in all_prob if "init_time" in f]
    lead_time = [f for f in all_prob if "per_lead_time" in f]
    if init_time:
        results["Probabilistic Init Time (Ignored)"] = len(init_time)
    if lead_time:
        results["Probabilistic Lead Time (Ignored)"] = len(lead_time)

    _report_checklist("probabilistic_metrics", results)

    # Report common files found
    prob_csv = _common_files(models, "probabilistic/crps_summary*.csv")
    if prob_csv:
        _print_file_list(f"Found {len(prob_csv)} common probabilistic metric files", prob_csv)

    # 1) Combine standard Probabilistic metrics (averaged)
    frames: list[pd.DataFrame] = []
    frames_lvl: list[pd.DataFrame] = []

    for lab, m in zip(labels, models, strict=False):
        # Standard (averaged)
        candidates = sorted((m / "probabilistic").glob("crps_summary*.csv"))
        f = next(
            (c for c in candidates if "per_lead_time" not in c.name and "per_level" not in c.name),
            None,
        )

        if f is not None and f.is_file():
            try:
                df = pd.read_csv(f)
                if "variable" not in df.columns:
                    if "Unnamed: 0" in df.columns:
                        df = df.rename(columns={"Unnamed: 0": "variable"})
                    else:
                        first = df.columns[0]
                        df = df.rename(columns={first: "variable"})

                df.insert(0, "model", lab)
                frames.append(df)
            except Exception:
                pass

        # Per-level
        f_lvl = next(
            (c for c in candidates if "per_level" in c.name),
            None,
        )
        if f_lvl is not None and f_lvl.is_file():
            try:
                df = pd.read_csv(f_lvl)
                if "variable" not in df.columns:
                    if "Unnamed: 0" in df.columns:
                        df = df.rename(columns={"Unnamed: 0": "variable"})
                    else:
                        first = df.columns[0]
                        df = df.rename(columns={first: "variable"})

                df.insert(0, "model", lab)
                frames_lvl.append(df)
            except Exception:
                pass

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        if combined["model"].nunique() >= 2:
            out_csv = dst_prob / "crps_summary_combined.csv"
            combined.to_csv(out_csv, index=False)
            c.success(f"Saved {out_csv.relative_to(out_root)}")

    if frames_lvl:
        combined_lvl = pd.concat(frames_lvl, ignore_index=True)
        if combined_lvl["model"].nunique() >= 2:
            out_csv = dst_prob / "crps_summary_per_level_combined.csv"
            combined_lvl.to_csv(out_csv, index=False)
            c.success(f"Saved {out_csv.relative_to(out_root)}")

    # 2) Combine per-lead-time CSV metrics (e.g. CRPS vs lead_time)
    temporal_rows: list[dict] = []
    for lab, m in zip(labels, models, strict=False):
        # Find all per-lead CSVs
        # Look for both legacy summary files and new line-plot data files
        csv_files = list((m / "probabilistic").glob("crps_summary*per_lead_time*.csv"))
        csv_files.extend((m / "probabilistic").glob("crps_line*by_lead*.csv"))

        for f in csv_files:
            try:
                df = pd.read_csv(f)
                # Expected columns: lead_time_hours, CRPS, Spread, RMSE, variable

                excluded = {
                    "variable",
                    "model",
                    "level",
                    "lead_time",
                    "lead_time_hours",
                    "init_time",
                    "valid_time",
                    "Unnamed: 0",
                    "source_file",
                    "member",
                    "",
                }
                metric_cols = [c for c in df.columns if c not in excluded]

                if not metric_cols:
                    continue

                lt_col = "lead_time_hours" if "lead_time_hours" in df.columns else "lead_time"
                if lt_col not in df.columns:
                    continue

                df["model"] = lab
                df = df.rename(columns={lt_col: "lead_time"})

                melted = df.melt(
                    id_vars=["model", "variable", "lead_time"],
                    value_vars=metric_cols,
                    var_name="metric",
                    value_name="value",
                )

                temporal_rows.extend(melted.to_dict("records"))
            except Exception:
                pass

    if temporal_rows:
        temporal_df = pd.DataFrame(temporal_rows)
        if len(temporal_df) > 0 and temporal_df["model"].nunique() >= 2:
            out_csv = dst_prob / "temporal_metrics_combined.csv"
            temporal_df.to_csv(out_csv, index=False)
            c.success(f"Saved {out_csv.relative_to(out_root)}")

            # Plot: group by lead_time and model
            pairs = temporal_df[["metric", "variable"]].drop_duplicates().values

            for metric, variable in pairs:
                subset = temporal_df[
                    (temporal_df["metric"] == metric) & (temporal_df["variable"] == variable)
                ].copy()

                pivot = subset.pivot(
                    index="lead_time", columns="model", values="value"
                ).sort_index()

                if not pivot.empty and pivot.notna().sum().sum() > 0 and pivot.shape[1] >= 2:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    pivot.plot(kind="line", ax=ax, marker="o", markersize=4)

                    ax.set_title(f"{metric} vs Lead Time — {format_variable_name(variable)}")
                    ax.set_ylabel(metric)
                    ax.set_xlabel("Lead Time (h)")
                    ax.grid(True, linestyle="--", alpha=0.6)
                    plt.tight_layout()

                    out_png = dst_prob / f"temporal_{metric}_{variable}_compare.png"
                    plt.savefig(out_png, bbox_inches="tight", dpi=200)
                    c.success(f"Saved {out_png.relative_to(out_root)}")
                    plt.close(fig)

    # 3) CRPS Maps (if available)
    # Pattern: crps_map_*.npz
    # We reuse intercompare_maps logic but specific to CRPS maps
    # Availability report
    per_model_map, _, uni_map = _scan_model_sets(models, "probabilistic/crps_map_*.npz")
    _report_missing("crps_maps", models, labels, per_model_map, uni_map)

    common_maps = _common_files(models, "probabilistic/crps_map_*.npz")
    if common_maps:
        _print_file_list(f"Found {len(common_maps)} common CRPS map files", common_maps)

        # Limit panels
        for base in common_maps[:max_crps_map_panels]:
            key = _parse_map_filename(base)
            payloads = [_load_npz(m / "probabilistic" / base) for m in models]

            # CRPS maps usually have 'crps' key
            predictions = []
            for p in payloads:
                val = p.get("crps")
                predictions.append(val)

            if any(x is None for x in predictions):
                continue

            lats = payloads[0].get("latitude")
            lons = payloads[0].get("longitude")
            var_name = payloads[0].get("variable")
            units = payloads[0].get("units")

            if lats is None or lons is None:
                continue

            # Plotting
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

            # Determine common vmin/vmax
            try:
                vmin = float(np.nanmin([np.nanmin(x) for x in predictions]))
                vmax = float(np.nanmax([np.nanmax(x) for x in predictions]))
            except ValueError:
                continue

            im0 = None
            for ax, lab, pred in zip(axes, labels, predictions, strict=False):
                im0 = ax.pcolormesh(
                    lons,
                    lats,
                    pred,
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                    transform=ccrs.PlateCarree(),
                )
                ax.coastlines(linewidth=0.5)
                ax.set_title(f"{lab}")

            # Colorbar
            if im0:
                cbar = fig.colorbar(
                    im0,
                    ax=axes if isinstance(axes, (list | np.ndarray)) else [axes],
                    orientation="horizontal",
                    fraction=0.05,
                    pad=0.08,
                )
                if units:
                    cbar.set_label(f"CRPS ({units})")
                else:
                    cbar.set_label("CRPS")

            if var_name:
                title_base = f"CRPS Map — {format_variable_name(str(var_name))}"
            else:
                title_base = "CRPS Map"
            date_suffix = extract_date_from_filename(key)
            fig.suptitle(f"{title_base}{date_suffix}")

            out_png = dst_prob / (key + "_compare.png")
            plt.savefig(out_png, bbox_inches="tight", dpi=200)
            c.success(f"Saved {out_png.relative_to(out_root)}")
            plt.close(fig)


def intercompare_ssim(models: list[Path], labels: list[str], out_root: Path) -> None:
    """Combine SSIM metrics from multiple models."""
    # Availability report
    per_model, _, uni = _scan_model_sets(models, "ssim/ssim_ssim_*.csv")
    _report_missing("ssim_ssim", models, labels, per_model, uni)

    results = {}
    all_multi = _common_files(models, "ssim/ssim_ssim_*.csv")

    # Check for any files
    results["SSIM Summary"] = 1 if all_multi else 0

    _report_checklist("ssim_ssim", results)

    # Report common files found
    multi_csv = _common_files(models, "ssim/ssim_ssim_*.csv")
    if multi_csv:
        _print_file_list(f"Found {len(multi_csv)} common SSIM metric files", multi_csv)

    # SSIM metrics
    dst_multi = _ensure_dir(out_root / "ssim")
    frames: list[pd.DataFrame] = []

    for lab, m in zip(labels, models, strict=False):
        candidates = sorted((m / "ssim").glob("ssim_ssim_*.csv"))
        # Prefer ensmean or det
        f = next(
            (c for c in candidates if "ensmean" in c.name or "det" in c.name),
            next(iter(candidates), None),
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

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        out_csv = dst_multi / "ssim_combined.csv"
        combined.to_csv(out_csv, index=False)
        c.success(f"[SSIM] Saved combined metrics to {out_csv}")

        # Plot SSIM comparison
        if "SSIM" in combined.columns:
            # Filter for AVERAGE_SSIM
            df_avg = combined[combined["variable"] == "AVERAGE_SSIM"]
            if not df_avg.empty:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(data=df_avg, x="model", y="SSIM", ax=ax)
                ax.set_title("SSIM Comparison")
                plt.tight_layout()
                out_png = dst_multi / "ssim_comparison.png"
                fig.savefig(out_png, dpi=150)
                plt.close(fig)
                c.success(f"[SSIM] Saved comparison plot to {out_png}")


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
    if "ets" in mods:
        intercompare_ets_metrics(models, labels, out_root)
    if "ssim" in mods:
        intercompare_ssim(models, labels, out_root)
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
