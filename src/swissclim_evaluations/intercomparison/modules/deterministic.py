from __future__ import annotations

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


def intercompare_deterministic_metrics(
    models: list[Path], labels: list[str], out_root: Path
) -> None:
    # Availability report
    per_model, _, uni = scan_model_sets(models, "deterministic/deterministic_metrics*.csv")
    report_missing("deterministic_metrics", models, labels, per_model, uni)

    results = {}
    all_det = common_files(models, "deterministic/deterministic_metrics*.csv")

    avg = [f for f in all_det if "averaged" in f]
    lvl = [f for f in all_det if "per_level" in f and "standardized" not in f]
    std = [f for f in all_det if "standardized" in f and "per_level" not in f]
    std_lvl = [f for f in all_det if "standardized" in f and "per_level" in f]
    lead = [f for f in all_det if "per_lead_time" in f or "by_lead" in f]

    results["Deterministic Averaged"] = 1 if avg else 0
    results["Deterministic Per Level"] = 1 if lvl else 0
    results["Deterministic Standardized"] = 1 if std else 0
    results["Deterministic Standardized Per Level"] = 1 if std_lvl else 0
    if lead:
        results["Deterministic Temporal"] = len(lead)

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

    report_checklist("deterministic_metrics", results)

    # Report common files found (to match panel counts)
    det_csv = common_files(models, "deterministic/deterministic_metrics*.csv")
    if det_csv:
        print_file_list(f"Found {len(det_csv)} common deterministic metric files", det_csv)

    # Deterministic metrics
    dst_det = ensure_dir(out_root / "deterministic")
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
            "lead_time_hours",
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
                # If we have multiple entries per variable/model (e.g. multiple lead times),
                # average them to produce a summary scalar metric.
                if tmp.duplicated(subset=["variable", "model"]).any():
                    tmp = tmp.groupby(["variable", "model"], as_index=False)[metric].mean()
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
                ax.set_title(f"{metric} by variable and model".title())
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

                # If variable column is missing, try to infer from filename
                if "variable" not in df.columns:
                    # Try to extract variable from filename
                    # Pattern: crps_line_<var>_by_lead...
                    fname = f.name
                    if fname.startswith("crps_line_"):
                        # Remove prefix
                        rest = fname[len("crps_line_") :]
                        # Remove suffix starting with _by_lead
                        if "_by_lead" in rest:
                            var_name = rest.split("_by_lead")[0]
                            df["variable"] = var_name
                    elif fname.startswith("crps_summary_"):
                        # Remove prefix
                        rest = fname[len("crps_summary_") :]
                        # Remove suffix starting with _per_lead
                        if "_per_lead" in rest:
                            var_name = rest.split("_per_lead")[0]
                            df["variable"] = var_name

                if "variable" not in df.columns:
                    continue

                id_vars = ["model", "variable", "lead_time"]
                if "level" in df.columns:
                    id_vars.append("level")

                melted = df.melt(
                    id_vars=id_vars,
                    value_vars=metric_cols,
                    var_name="metric",
                    value_name="value",
                )

                temporal_rows.extend(melted.to_dict("records"))
            except Exception:
                # If a single CSV is malformed or cannot be read, skip it and continue
                # processing the remaining files so that aggregation can still proceed.
                pass

    if temporal_rows:
        temporal_df = pd.DataFrame(temporal_rows)
        if len(temporal_df) > 0 and temporal_df["model"].nunique() >= 2:
            out_csv = dst_det / "temporal_metrics_combined.csv"
            temporal_df.to_csv(out_csv, index=False)
            c.success(f"Saved {out_csv.relative_to(out_root)}")

            pair_cols = ["metric", "variable"]
            has_level = "level" in temporal_df.columns
            if has_level:
                pair_cols.append("level")

            pairs = temporal_df[pair_cols].drop_duplicates().to_dict("records")

            for pair in pairs:
                metric = pair["metric"]
                variable = pair["variable"]
                subset = temporal_df[
                    (temporal_df["metric"] == metric) & (temporal_df["variable"] == variable)
                ].copy()
                level_token = ""
                if has_level:
                    level_val = pair.get("level")
                    if pd.notna(level_val):
                        subset = subset[subset["level"] == level_val]
                        level_int = int(level_val)
                        level_token = f"_level{level_int}"
                    else:
                        subset = subset[subset["level"].isna()]

                pivot = subset.pivot_table(
                    index="lead_time", columns="model", values="value", aggfunc="mean"
                ).sort_index()

                if not pivot.empty and pivot.notna().sum().sum() > 0 and pivot.shape[1] >= 2:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    pivot.plot(kind="line", ax=ax, marker="o", markersize=4)

                    level_has_value = has_level and pd.notna(pair.get("level"))
                    level_title = f" @ {int(level_val)}" if level_has_value else ""
                    ax.set_title(
                        f"{metric} vs Lead Time — {format_variable_name(variable)}{level_title}"
                    )
                    ax.set_ylabel(metric)
                    ax.set_xlabel("Lead Time (h)")
                    ax.grid(True, linestyle="--", alpha=0.6)
                    ax.legend(title=None)
                    plt.tight_layout()

                    out_png = dst_det / f"temporal_{metric}_{variable}{level_token}_compare.png"
                    plt.savefig(out_png, bbox_inches="tight", dpi=200)
                    c.success(f"Saved {out_png.relative_to(out_root)}")
                    plt.close(fig)
