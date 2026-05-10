from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from swissclim_evaluations.helpers import format_variable_name
from swissclim_evaluations.intercomparison.core import (
    c,
    common_files,
    ensure_dir,
    model_color_map,
    print_file_list,
    reorder_pivot_columns,
    report_checklist,
    report_missing,
    scan_model_sets,
)


def intercompare_fss_metrics(models: list[Path], labels: list[str], out_root: Path) -> None:
    """Combine FSS metrics from multiple models and plot vs lead time."""
    dst_fss = ensure_dir(out_root / "fss")
    _cmap = model_color_map(labels)

    per_model, _, uni = scan_model_sets(models, "fss/fss_metrics_*.csv")
    report_missing("fss_metrics", models, labels, per_model, uni)

    all_fss = common_files(models, "fss/fss_metrics_*.csv")
    results = {"FSS Metrics": len(all_fss)}
    report_checklist("fss_metrics", results)

    if not all_fss:
        c.warn("No common FSS metrics files found. Skipping plots.")
        return

    print_file_list(f"Found {len(all_fss)} common FSS metrics files", all_fss)

    # Collect per-member-per-lead CSVs for line plots
    frames_lead: list[pd.DataFrame] = []
    for lab, m in zip(labels, models, strict=False):
        candidates = list((m / "fss").glob("fss_metrics_per_member_per_lead*.csv"))
        for cand in candidates:
            try:
                df = pd.read_csv(cand)
                df.insert(0, "model", lab)
                frames_lead.append(df)
            except Exception:
                continue

    # Collect averaged/overall CSVs
    frames_avg: list[pd.DataFrame] = []
    for lab, m in zip(labels, models, strict=False):
        for pat in ["fss_metrics_averaged*.csv", "fss_metrics_members_mean*.csv"]:
            candidates = list((m / "fss").glob(pat))
            for cand in candidates:
                try:
                    df = pd.read_csv(cand)
                    if any("FSS" in str(col) for col in df.columns):
                        df.insert(0, "model", lab)
                        frames_avg.append(df)
                        break
                except Exception:
                    continue
            if frames_avg and frames_avg[-1]["model"].iloc[0] == lab:
                break

    # Save combined averaged metrics
    if frames_avg:
        combined_avg = pd.concat(frames_avg, ignore_index=True)
        if combined_avg["model"].nunique() >= 2:
            out_csv = dst_fss / "fss_metrics_combined.csv"
            combined_avg.to_csv(out_csv, index=False)
            c.success(f"Saved {out_csv.relative_to(out_root)}")

    # Per-member per-lead line plots
    if frames_lead:
        combined_lead = pd.concat(frames_lead, ignore_index=True)
        out_csv = dst_fss / "fss_per_member_per_lead_combined.csv"
        combined_lead.to_csv(out_csv, index=False)
        c.success(f"Saved {out_csv.relative_to(out_root)}")

        # Plot: mean over members, one line per model, per (variable, threshold)
        fss_cols = [col for col in combined_lead.columns if col.startswith("FSS ")]
        if "variable" in combined_lead.columns and "lead_time_hours" in combined_lead.columns:
            for fss_col in fss_cols:
                for var, grp in combined_lead.groupby("variable"):
                    # Average over members
                    pivot = grp.pivot_table(
                        index="lead_time_hours",
                        columns="model",
                        values=fss_col,
                        aggfunc="mean",
                    ).sort_index()

                    if pivot.empty or pivot.notna().sum().sum() == 0 or pivot.shape[1] < 2:
                        continue

                    pivot = reorder_pivot_columns(pivot, labels)
                    _colors = [_cmap[c] for c in pivot.columns if c in _cmap]

                    fig, ax = plt.subplots(figsize=(10, 6))
                    pivot.plot(
                        kind="line", ax=ax, marker="o", markersize=4, color=_colors or None
                    )
                    thresh_label = fss_col.replace("FSS ", "")
                    ax.set_title(
                        f"FSS vs Lead Time — {format_variable_name(str(var))} ({thresh_label})"
                    )
                    ax.set_ylabel("FSS")
                    ax.set_xlabel("Lead Time (h)")
                    ax.set_ylim(0, 1)
                    ax.grid(True, linestyle="--", alpha=0.6)
                    plt.tight_layout()

                    safe_name = f"fss_{var}_{thresh_label.replace('%', 'pct')}_compare.png"
                    out_png = dst_fss / safe_name
                    plt.savefig(out_png, bbox_inches="tight", dpi=200)
                    c.success(f"Saved {out_png.relative_to(out_root)}")
                    plt.close(fig)

            # Also plot per-member spread (envelope) for each model
            for fss_col in fss_cols:
                for var, grp in combined_lead.groupby("variable"):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    has_data = False

                    for lab in labels:
                        model_data = grp[grp["model"] == lab]
                        if model_data.empty:
                            continue
                        stats = model_data.groupby("lead_time_hours")[fss_col].agg(
                            ["mean", "min", "max"]
                        )
                        if stats.empty:
                            continue
                        has_data = True
                        color = _cmap.get(lab)
                        ax.plot(
                            stats.index, stats["mean"], marker="o", markersize=3,
                            label=lab, color=color,
                        )
                        ax.fill_between(
                            stats.index, stats["min"], stats["max"],
                            alpha=0.15, color=color,
                        )

                    if not has_data:
                        plt.close(fig)
                        continue

                    thresh_label = fss_col.replace("FSS ", "")
                    ax.set_title(
                        f"FSS Member Spread — {format_variable_name(str(var))} ({thresh_label})"
                    )
                    ax.set_ylabel("FSS")
                    ax.set_xlabel("Lead Time (h)")
                    ax.set_ylim(0, 1)
                    ax.legend()
                    ax.grid(True, linestyle="--", alpha=0.6)
                    plt.tight_layout()

                    safe_name = f"fss_{var}_{thresh_label.replace('%', 'pct')}_spread.png"
                    out_png = dst_fss / safe_name
                    plt.savefig(out_png, bbox_inches="tight", dpi=200)
                    c.success(f"Saved {out_png.relative_to(out_root)}")
                    plt.close(fig)
