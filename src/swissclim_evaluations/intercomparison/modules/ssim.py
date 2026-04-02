from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from swissclim_evaluations import console as c
from swissclim_evaluations.helpers import format_variable_name
from swissclim_evaluations.intercomparison.core import (
    common_files,
    ensure_dir,
    model_color_map,
    print_file_list,
    reorder_pivot_columns,
    report_checklist,
    report_missing,
    scan_model_sets,
)


def _normalize_variable_col(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a 'variable' column exists, renaming the first column if needed."""
    if "variable" not in df.columns:
        first = "Unnamed: 0" if "Unnamed: 0" in df.columns else df.columns[0]
        df = df.rename(columns={first: "variable"})
    return df


def _pick_csv(model_dir: Path, pattern: str) -> Path | None:
    """Return the best matching CSV for a model directory and glob pattern."""
    candidates = sorted(model_dir.glob(pattern))
    # Prefer ensmean, fall back to first available
    return next(
        (c for c in candidates if "ensmean" in c.name),
        next(iter(candidates), None),
    )


def intercompare_ssim(models: list[Path], labels: list[str], out_root: Path) -> None:
    """Combine SSIM metrics from multiple models and produce comparison outputs."""
    _cmap = model_color_map(labels)
    dst_ssim = ensure_dir(out_root / "ssim")

    # ── Availability report ───────────────────────────────────────────────────
    per_model, _, uni = scan_model_sets(models, "ssim/ssim_ssim_*.csv")
    report_missing("ssim_ssim", models, labels, per_model, uni)

    results = {
        "SSIM Overall": 1 if common_files(models, "ssim/ssim_ssim_*.csv") else 0,
        "SSIM Per Level": (1 if common_files(models, "ssim/ssim_ssim_per_level_*.csv") else 0),
        "SSIM By Lead": (1 if common_files(models, "ssim/ssim_ssim_by_lead_*.csv") else 0),
        "SSIM Per Level By Lead": (
            1 if common_files(models, "ssim/ssim_ssim_per_level_by_lead_*.csv") else 0
        ),
    }
    report_checklist("ssim_ssim", results)

    if common_files(models, "ssim/ssim_ssim_*.csv"):
        print_file_list(
            "Found common SSIM metric files",
            common_files(models, "ssim/ssim_ssim_*.csv"),
        )

    # ── Overall CSV ───────────────────────────────────────────────────────────
    frames_overall: list[pd.DataFrame] = []
    for lab, m in zip(labels, models, strict=False):
        # Exclude per_level / by_lead / per_level_by_lead files
        candidates = [
            p
            for p in sorted((m / "ssim").glob("ssim_ssim_*.csv"))
            if "per_level" not in p.name and "by_lead" not in p.name
        ]
        f = next((p for p in candidates if "ensmean" in p.name), next(iter(candidates), None))
        if f and f.is_file():
            df = _normalize_variable_col(pd.read_csv(f))
            df.insert(0, "model", lab)
            frames_overall.append(df)

    if frames_overall:
        combined = pd.concat(frames_overall, ignore_index=True)
        combined.to_csv(dst_ssim / "ssim_combined.csv", index=False)
        c.success("[SSIM] saved ssim_combined.csv")

        if "SSIM" in combined.columns:
            # Average SSIM bar chart (one bar per model)
            df_avg = combined[combined["variable"] == "AVERAGE_SSIM"]
            if not df_avg.empty and df_avg["model"].nunique() >= 2:
                colors = [_cmap.get(m, None) for m in df_avg["model"]]
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.bar(df_avg["model"], df_avg["SSIM"], color=colors)
                ax.set_title("Average SSIM per Model")
                ax.set_ylabel("SSIM")
                ax.set_ylim(0, 1)
                ax.grid(True, axis="y", linestyle="--", alpha=0.6)
                plt.tight_layout()
                fig.savefig(dst_ssim / "ssim_average_compare.png", dpi=150)
                plt.close(fig)
                c.success("[SSIM] saved ssim_average_compare.png")

            # Per-variable grouped bar chart (one group per variable)
            df_vars = combined[combined["variable"] != "AVERAGE_SSIM"]
            if not df_vars.empty and df_vars["model"].nunique() >= 2:
                try:
                    pivot = df_vars.pivot(index="variable", columns="model", values="SSIM")
                    pivot = reorder_pivot_columns(pivot, labels)
                    bar_colors = [_cmap.get(m, None) for m in pivot.columns]
                    fig, ax = plt.subplots(figsize=(max(8, len(pivot) * 1.5), 5))
                    pivot.plot(kind="bar", ax=ax, color=bar_colors or None, width=0.8)
                    ax.set_title("SSIM per Variable")
                    ax.set_ylabel("SSIM")
                    ax.set_ylim(0, 1)
                    ax.set_xlabel("")
                    plt.xticks(rotation=30, ha="right")
                    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
                    plt.tight_layout()
                    fig.savefig(dst_ssim / "ssim_per_variable_compare.png", dpi=150)
                    plt.close(fig)
                    c.success("[SSIM] saved ssim_per_variable_compare.png")
                except Exception:
                    pass

    # ── Per-level CSV ─────────────────────────────────────────────────────────
    frames_level: list[pd.DataFrame] = []
    for lab, m in zip(labels, models, strict=False):
        f = _pick_csv(m / "ssim", "ssim_ssim_per_level_*.csv")
        if f and f.is_file():
            df = _normalize_variable_col(pd.read_csv(f))
            df.insert(0, "model", lab)
            frames_level.append(df)

    if frames_level:
        combined_lvl = pd.concat(frames_level, ignore_index=True)
        combined_lvl.to_csv(dst_ssim / "ssim_per_level_combined.csv", index=False)
        c.success("[SSIM] saved ssim_per_level_combined.csv")

        # Line chart: SSIM vs level, one line per model, one figure per variable
        if "level" in combined_lvl.columns and "SSIM" in combined_lvl.columns:
            for var, grp in combined_lvl.groupby("variable"):
                pivot = grp.pivot_table(
                    index="level", columns="model", values="SSIM", aggfunc="mean"
                ).sort_index(ascending=False)
                if pivot.empty or pivot.shape[1] < 2:
                    continue
                pivot = reorder_pivot_columns(pivot, labels)
                line_colors = [_cmap.get(m, None) for m in pivot.columns]
                fig, ax = plt.subplots(figsize=(7, 5))
                pivot.plot(kind="line", ax=ax, marker="o", markersize=5, color=line_colors or None)
                ax.set_title(f"SSIM per Level — {format_variable_name(str(var))}")
                ax.set_ylabel("SSIM")
                ax.set_xlabel("Pressure Level (hPa)")
                ax.grid(True, linestyle="--", alpha=0.6)
                plt.tight_layout()
                safe_var = str(var).replace(" ", "_")
                fig.savefig(dst_ssim / f"ssim_per_level_{safe_var}_compare.png", dpi=150)
                plt.close(fig)
                c.success(f"[SSIM] saved ssim_per_level_{safe_var}_compare.png")

    # ── By-lead CSV ───────────────────────────────────────────────────────────
    frames_lead: list[pd.DataFrame] = []
    for lab, m in zip(labels, models, strict=False):
        f = _pick_csv(m / "ssim", "ssim_ssim_by_lead_*.csv")
        if f and f.is_file():
            df = _normalize_variable_col(pd.read_csv(f))
            df.insert(0, "model", lab)
            frames_lead.append(df)

    if frames_lead:
        combined_lead = pd.concat(frames_lead, ignore_index=True)
        combined_lead.to_csv(dst_ssim / "ssim_by_lead_combined.csv", index=False)
        c.success("[SSIM] saved ssim_by_lead_combined.csv")

        # Line chart: SSIM vs lead_time_hours, one line per model, one figure per variable
        lead_col = "lead_time_hours" if "lead_time_hours" in combined_lead.columns else "lead_time"
        if lead_col in combined_lead.columns and "SSIM" in combined_lead.columns:
            for var, grp in combined_lead.groupby("variable"):
                pivot = grp.pivot_table(
                    index=lead_col, columns="model", values="SSIM", aggfunc="mean"
                ).sort_index()
                if pivot.empty or pivot.shape[1] < 2:
                    continue
                pivot = reorder_pivot_columns(pivot, labels)
                line_colors = [_cmap.get(m, None) for m in pivot.columns]
                fig, ax = plt.subplots(figsize=(8, 5))
                pivot.plot(kind="line", ax=ax, marker="o", markersize=5, color=line_colors or None)
                ax.set_title(f"SSIM by Lead Time — {format_variable_name(str(var))}")
                ax.set_ylabel("SSIM")
                ax.set_xlabel("Lead Time (h)")
                ax.grid(True, linestyle="--", alpha=0.6)
                plt.tight_layout()
                safe_var = str(var).replace(" ", "_")
                fig.savefig(dst_ssim / f"ssim_by_lead_{safe_var}_compare.png", dpi=150)
                plt.close(fig)
                c.success(f"[SSIM] saved ssim_by_lead_{safe_var}_compare.png")

    # ── Per-level-by-lead CSV ─────────────────────────────────────────────────
    frames_lvl_lead: list[pd.DataFrame] = []
    for lab, m in zip(labels, models, strict=False):
        f = _pick_csv(m / "ssim", "ssim_ssim_per_level_by_lead_*.csv")
        if f and f.is_file():
            df = _normalize_variable_col(pd.read_csv(f))
            df.insert(0, "model", lab)
            frames_lvl_lead.append(df)

    if frames_lvl_lead:
        combined_lvl_lead = pd.concat(frames_lvl_lead, ignore_index=True)
        combined_lvl_lead.to_csv(dst_ssim / "ssim_per_level_by_lead_combined.csv", index=False)
        c.success("[SSIM] saved ssim_per_level_by_lead_combined.csv")
