from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from swissclim_evaluations.helpers import (
    COLOR_GROUND_TRUTH,
    extract_date_from_filename,
    format_variable_name,
    get_variable_units,
)
from swissclim_evaluations.intercomparison.core import (
    c,
    clean_var_from_filename,
    common_files,
    ensure_dir,
    load_npz,
    print_file_list,
    report_checklist,
    report_missing,
    scan_model_sets,
)


def _plot_hist_counts(ax, edges: np.ndarray, counts: np.ndarray, label: str | None, color: str):
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
    dst = ensure_dir(out_root / "histograms")
    # Availability report
    patterns = [
        "histograms/hist_*latbands_combined*.npz",
        "histograms/hist_*global*.npz",
    ]
    total_per_model: list[set[str]] = [set() for _ in models]
    total_union = set()

    for pat in patterns:
        pm, _, uni = scan_model_sets(models, pat)
        for i, s in enumerate(pm):
            total_per_model[i].update(s)
        total_union.update(uni)

    report_missing("histograms", models, labels, total_per_model, total_union)

    results = {}
    # Plots: 1-to-1
    plots = common_files(models, "histograms/hist_*latbands_combined*.npz")
    results["Histograms (Latbands)"] = len(plots)

    # Check for ignored
    all_hist = common_files(models, "histograms/hist_*.npz")
    ignored = [f for f in all_hist if "latbands" not in f and "global" not in f]
    if ignored:
        results["Other Histograms (Ignored)"] = len(ignored)

    # --- Latbands Histograms ---
    common = common_files(models, str(src_rel / "hist_*latbands*.npz"))
    # Filter out global if it matches latbands (unlikely but safe)
    common = [f for f in common if "global" not in f]
    results["Histograms (Latbands)"] = len(common)

    # --- Global Histograms ---
    common_g = common_files(models, str(src_rel / "hist_*global*.npz"))
    results["Histograms (Global)"] = len(common_g)

    report_checklist("histograms", results)

    if not common and not common_g:
        c.warn("No common histogram files found (latbands or global). Skipping plots.")
        return

    colors = sns.color_palette("tab10", n_colors=len(models))

    # Process Global
    if common_g:
        print_file_list(f"Found {len(common_g)} common global histogram files", common_g)
        for base in common_g:
            payloads = [load_npz(m / src_rel / base) for m in models]
            fig, ax = plt.subplots(figsize=(10, 6), dpi=160)

            # Target (from first model)
            counts_target = payloads[0].get("counts_target", payloads[0].get("counts_true"))
            if counts_target is None:
                counts_target = payloads[0].get("densities_true")

            bins_target = payloads[0].get("bins", payloads[0].get("edges"))

            if counts_target is not None:
                counts_target = np.asarray(counts_target)
            if bins_target is not None:
                bins_target = np.asarray(bins_target)

            if counts_target is not None and bins_target is not None:
                if counts_target.ndim > 1:
                    for idx in range(len(counts_target)):
                        _plot_hist_counts(
                            ax,
                            bins_target[idx],
                            counts_target[idx],
                            label="Target" if idx == 0 else None,
                            color=COLOR_GROUND_TRUTH,
                        )
                else:
                    _plot_hist_counts(
                        ax, bins_target, counts_target, label="Target", color=COLOR_GROUND_TRUTH
                    )

            # Models
            for i, (lab, pay) in enumerate(zip(labels, payloads, strict=False)):
                counts_prediction = pay.get("counts_prediction", pay.get("counts_pred"))
                if counts_prediction is None:
                    counts_prediction = pay.get("densities_pred")

                bins_prediction = pay.get("bins", pay.get("edges"))

                if counts_prediction is not None:
                    counts_prediction = np.asarray(counts_prediction)
                if bins_prediction is not None:
                    bins_prediction = np.asarray(bins_prediction)

                if counts_prediction is not None and bins_prediction is not None:
                    if counts_prediction.ndim > 1:
                        for idx in range(len(counts_prediction)):
                            _plot_hist_counts(
                                ax,
                                bins_prediction[idx],
                                counts_prediction[idx],
                                label=lab if idx == 0 else None,
                                color=colors[i],
                            )
                    else:
                        _plot_hist_counts(
                            ax,
                            bins_prediction,
                            counts_prediction,
                            label=lab,
                            color=colors[i],
                        )

            raw_var = clean_var_from_filename(base, prefix="hist_", format=False)
            var = format_variable_name(raw_var)
            date_suffix = extract_date_from_filename(base)
            ax.set_title(f"Global Histogram — {var}{date_suffix}", fontsize=16)
            ax.set_ylabel("Frequency (log)")
            ax.set_yscale("log")

            units_val = payloads[0].get("units")
            if not units_val:
                units_val = get_variable_units(None, raw_var)

            label = var
            if units_val:
                label += f" [{units_val}]"
            ax.set_xlabel(label)
            ax.legend(frameon=False)
            ax.grid(True, which="both", ls="--", alpha=0.4)

            out_png = dst / base.replace(".npz", "_compare.png")
            plt.tight_layout()
            plt.savefig(out_png, bbox_inches="tight", dpi=200)
            plt.close(fig)
            c.success(f"Saved {out_png.relative_to(out_root)}")

    # Process Latbands
    if common:
        print_file_list(f"Found {len(common)} common latbands histogram files", common)

        for base in common:
            payloads = [load_npz(m / src_rel / base) for m in models]

            raw_var = clean_var_from_filename(base, prefix="hist_", format=False)
            var_fmt = format_variable_name(raw_var)

            units = payloads[0].get("units")
            if not units:
                units = get_variable_units(None, raw_var)

            xlabel = var_fmt
            if units:
                xlabel += f" [{units}]"

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
                ds_prediction_pairs = payloads[0]["neg_counts"][j]
                # Each element is (counts_ds, counts_prediction)
                counts_ds = ds_prediction_pairs[0]
                bins_ds = payloads[0]["neg_bins"][j]

                _plot_hist_counts(ax, bins_ds, counts_ds, label="Target", color=COLOR_GROUND_TRUTH)

                # Plot each prediction model
                for i, (lab, pay) in enumerate(zip(labels, payloads, strict=False)):
                    counts_prediction = pay["neg_counts"][j][1]
                    bins_prediction = pay["neg_bins"][j]
                    _plot_hist_counts(
                        ax, bins_prediction, counts_prediction, label=lab, color=colors[i]
                    )
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
                ax.set_xlabel(xlabel)

            # Left column: northern hemisphere bands
            for j in range(n_rows):
                ax = axs[j, 0]
                ds_prediction_pairs = payloads[0]["pos_counts"][j]
                counts_ds = ds_prediction_pairs[0]
                bins_ds = payloads[0]["pos_bins"][j]

                _plot_hist_counts(ax, bins_ds, counts_ds, label="Target", color=COLOR_GROUND_TRUTH)

                for i, (lab, pay) in enumerate(zip(labels, payloads, strict=False)):
                    counts_prediction = pay["pos_counts"][j][1]
                    bins_prediction = pay["pos_bins"][j]
                    _plot_hist_counts(
                        ax, bins_prediction, counts_prediction, label=lab, color=colors[i]
                    )
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
                ax.set_xlabel(xlabel)

            # Legends: add a single shared legend
            handles, labels_leg = axs[0, 0].get_legend_handles_labels()
            if handles:
                fig.legend(
                    handles[: 1 + len(models)],
                    labels_leg[: 1 + len(models)],
                    loc="lower center",
                    ncol=min(6, 1 + len(models)),
                    bbox_to_anchor=(0.5, 0.02),
                )
            plt.tight_layout(rect=(0, 0.08, 1, 1))
            # Derive a variable/level label for the figure title.
            var = clean_var_from_filename(base, prefix="hist_")
            date_suffix = extract_date_from_filename(base)

            fig.suptitle(
                f"Distributions by Latitude Bands — {var}{date_suffix}", y=1.02, fontsize=20
            )
            out_png = dst / base.replace(".npz", "_compare.png")
            plt.savefig(out_png, bbox_inches="tight", dpi=200)
            c.success(f"Saved {out_png.relative_to(out_root)}")
            plt.close(fig)
