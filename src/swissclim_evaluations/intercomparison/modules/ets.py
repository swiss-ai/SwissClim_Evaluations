from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from swissclim_evaluations.helpers import format_variable_name
from swissclim_evaluations.intercomparison.core import (
    c,
    common_files,
    ensure_dir,
    print_file_list,
    report_checklist,
    report_missing,
    scan_model_sets,
)


def intercompare_ets_metrics(models: list[Path], labels: list[str], out_root: Path) -> None:
    """Combine ETS metrics from multiple models and plot vs lead time."""
    dst_ets = ensure_dir(out_root / "ets")

    # Availability report
    # Look for any ETS metrics CSV
    per_model, _, uni = scan_model_sets(models, "ets/ets_metrics_*.csv")
    report_missing("ets_metrics", models, labels, per_model, uni)

    results = {}
    all_ets = common_files(models, "ets/ets_metrics_*.csv")
    results["ETS Metrics"] = len(all_ets)
    report_checklist("ets_metrics", results)

    if not all_ets:
        c.warn("No common ETS metrics files found. Skipping plots.")
        return

    print_file_list(f"Found {len(all_ets)} common ETS metrics files", all_ets)

    # Combine wide CSVs if they exist (legacy)
    frames: list[pd.DataFrame] = []
    # Also look for per-lead or averaged files that might contain ETS data
    # We'll try to find a file that has 'ets' in the name and looks like a metric file

    # Strategy: Look for specific patterns
    patterns = [
        "ets_metrics_by_lead_wide.csv",
        "ets_metrics_averaged*.csv",
        "ets_metrics_per_lead*.csv",
    ]

    for lab, m in zip(labels, models, strict=False):
        found_df = None
        for pat in patterns:
            candidates = list((m / "ets").glob(pat))
            # Prefer 'wide' if exists, else averaged/per_lead
            if not candidates:
                continue
            # Pick the first one that works
            for cand in candidates:
                try:
                    df = pd.read_csv(cand)
                    # Check if it has ETS-like columns (e.g. ETS, EDI, FBI, or threshold columns)
                    # or if it's in the 'wide' format
                    if "ETS" in df.columns or any("ETS" in c for c in df.columns):
                        found_df = df
                        break
                except Exception:
                    continue
            if found_df is not None:
                break

        if found_df is not None:
            found_df.insert(0, "model", lab)
            frames.append(found_df)

    if not frames:
        return

    combined = pd.concat(frames, ignore_index=True)
    if combined["model"].nunique() >= 2:
        out_csv = dst_ets / "ets_metrics_combined.csv"
        combined.to_csv(out_csv, index=False)
        c.success(f"Saved {out_csv.relative_to(out_root)}")

        # Generate bar plots for ETS metrics
        # We expect columns like 'threshold', 'ETS', 'model', 'variable'
        # If 'threshold' is present, we can plot ETS vs threshold for each model
        if "threshold" in combined.columns and "ETS" in combined.columns:
            # Group by variable
            for var, group in combined.groupby("variable"):
                # Check if we have multiple thresholds
                if group["threshold"].nunique() > 1:
                    # Pivot to get models side-by-side for each threshold
                    pivot = group.pivot(index="threshold", columns="model", values="ETS")
                    if not pivot.empty:
                        fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
                        pivot.plot(kind="bar", ax=ax, width=0.8)
                        ax.set_title(f"ETS by Threshold — {format_variable_name(str(var))}")
                        ax.set_ylabel("ETS")
                        ax.set_xlabel("Threshold")
                        plt.xticks(rotation=45)
                        ax.grid(True, axis="y", linestyle="--", alpha=0.7)
                        plt.tight_layout()
                        out_png = dst_ets / f"ets_barplot_{var}_compare.png"
                        plt.savefig(out_png, bbox_inches="tight", dpi=200)
                        c.success(f"Saved {out_png.relative_to(out_root)}")
                        plt.close(fig)

        # Also handle wide format where thresholds are in column names (e.g. "tp_ETS 1mm")
        wide_metrics: dict[str, list[tuple[str, str]]] = {}  # (variable, threshold) -> col_name
        for col in combined.columns:
            if "_ETS " in col:
                try:
                    var_part, thresh_part = col.split("_ETS ", 1)
                    wide_metrics.setdefault(var_part, []).append((thresh_part, col))
                except ValueError:
                    pass

        for var, items in wide_metrics.items():
            if len(items) > 1:
                # We have multiple thresholds for this variable
                plot_data: dict[str, dict[str, float]] = {}
                thresholds = []

                for thresh, col in items:
                    thresholds.append(thresh)
                    # Group by model and take mean of this column
                    means = combined.groupby("model")[col].mean()
                    for model, val in means.items():
                        if model not in plot_data:
                            plot_data[model] = {}
                        plot_data[model][thresh] = val

                if plot_data:
                    df_plot = pd.DataFrame(plot_data).T  # models as index, thresholds as columns
                    # We want thresholds as index for bar plot
                    df_plot = df_plot.T

                    # Sort thresholds if they are numeric-like
                    try:
                        # Try to parse "1mm", "5mm" etc.
                        def parse_thresh(t):
                            return float(re.sub(r"[^\d\.]", "", t))

                        sorted_thresh = sorted(df_plot.index, key=parse_thresh)
                        df_plot = df_plot.reindex(sorted_thresh)
                    except Exception:
                        pass

                    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
                    df_plot.plot(kind="bar", ax=ax, width=0.8)
                    ax.set_title(
                        f"ETS by Threshold (Mean over Leads) — {format_variable_name(str(var))}"
                    )
                    ax.set_ylabel("ETS")
                    ax.set_xlabel("Threshold")
                    plt.xticks(rotation=45)
                    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
                    plt.tight_layout()
                    out_png = dst_ets / f"ets_barplot_{var}_wide_compare.png"
                    plt.savefig(out_png, bbox_inches="tight", dpi=200)
                    c.success(f"Saved {out_png.relative_to(out_root)}")
                    plt.close(fig)

        # Plot ETS vs Lead Time per (Variable, Threshold)
        # Columns are: model, lead_time_hours, <var>_ETS <thresh>%
        meta_cols = {"model", "lead_time_hours", "Unnamed: 0"}
        metric_cols = [c for c in combined.columns if c not in meta_cols]

        # Group by variable and threshold
        # Expected format: "{var}_ETS {thresh}%"
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
