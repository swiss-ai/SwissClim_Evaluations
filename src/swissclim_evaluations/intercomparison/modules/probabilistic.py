from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from swissclim_evaluations.helpers import (
    extract_date_from_filename,
    format_variable_name,
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


def _parse_map_filename(name: str) -> str:
    return name[:-4] if name.endswith(".npz") else name


def intercompare_probabilistic(
    models: list[Path],
    labels: list[str],
    out_root: Path,
    max_crps_map_panels: int = 4,
) -> None:
    dst_prob = ensure_dir(out_root / "probabilistic")

    # Availability report
    # Scan broadly for any CSV to catch CRPS, SSR, etc.
    per_model, _, uni = scan_model_sets(models, "probabilistic/*.csv")
    report_missing("probabilistic", models, labels, per_model, uni)

    results = {}
    all_prob = common_files(models, "probabilistic/crps_summary*.csv")

    # Check for different types
    avg = [f for f in all_prob if "per_lead_time" not in f and "per_level" not in f]
    lvl = [f for f in all_prob if "per_level" in f]
    lead = [f for f in all_prob if "per_lead_time" in f]

    results["CRPS Summary"] = 1 if avg else 0
    results["CRPS Summary (Per Level)"] = 1 if lvl else 0
    if lead:
        results["CRPS Summary (Per Lead)"] = len(lead)

    # Maps
    maps = common_files(models, "probabilistic/crps_map_*.npz")
    results["CRPS Maps"] = len(maps)

    # SSR
    ssr = common_files(models, "probabilistic/spread_skill_ratio*.csv")
    ssr_line = common_files(models, "probabilistic/ssr_line_*.csv")
    results["Spread Skill Ratio"] = 1 if (ssr or ssr_line) else 0
    if ssr_line:
        results["SSR Line Plots"] = len(ssr_line)

    # Ensemble
    ens = common_files(models, "probabilistic/crps_ensemble*.csv")
    results["CRPS Ensemble"] = 1 if ens else 0

    # PIT
    pit = common_files(models, "probabilistic/pit_hist*.npz")
    results["PIT Histograms"] = len(pit)

    report_checklist("probabilistic", results)

    # Report common files found
    prob_csv = common_files(models, "probabilistic/crps_summary*.csv")
    if prob_csv:
        print_file_list(f"Found {len(prob_csv)} common probabilistic metric files", prob_csv)

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

    # --- 3. Temporal Metrics ---
    temporal_rows: list[dict] = []
    for lab, m in zip(labels, models, strict=False):
        # Find all per-lead CSVs
        # Look for both legacy summary files and new line-plot data files
        csv_files = list((m / "probabilistic").glob("crps_summary*per_lead_time*.csv"))
        csv_files.extend((m / "probabilistic").glob("crps_line*by_lead*.csv"))
        csv_files.extend((m / "probabilistic").glob("pit_hist*by_lead*.csv"))

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
                    elif fname.startswith("pit_hist_"):
                        # Remove prefix
                        rest = fname[len("pit_hist_") :]
                        # Remove suffix starting with _by_lead
                        if "_by_lead" in rest:
                            var_name = rest.split("_by_lead")[0]
                            if "_uniform_diff" in var_name:
                                var_name = var_name.split("_uniform_diff")[0]
                            df["variable"] = var_name

                if "variable" not in df.columns:
                    continue

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

                pivot = subset.pivot_table(
                    index="lead_time", columns="model", values="value", aggfunc="mean"
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

    # --- 4. CRPS Maps ---
    common_maps = common_files(models, "probabilistic/crps_map_*.npz")
    if common_maps:
        print_file_list(f"Found {len(common_maps)} common CRPS map files", common_maps)

        # Limit panels
        seen_vars: set[str] = set()
        for base in common_maps:
            if len(seen_vars) >= max_crps_map_panels:
                break

            key = _parse_map_filename(base)
            # Check if we already processed this variable (ignoring date/ens suffix)
            var_part = clean_var_from_filename(base, prefix="crps_map_")
            if var_part in seen_vars:
                continue

            payloads = [load_npz(m / "probabilistic" / base) for m in models]

            # CRPS maps usually have 'crps' key
            predictions = []
            for p in payloads:
                val = p.get("crps")
                predictions.append(val)

            if any(x is None for x in predictions):
                continue

            lats = payloads[0].get("latitude")
            lons = payloads[0].get("longitude")
            map_var_name = payloads[0].get("variable")
            units = payloads[0].get("units")

            if lats is None or lons is None or map_var_name is None:
                continue

            seen_vars.add(var_part)

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

            if map_var_name:
                title_base = f"CRPS Map — {format_variable_name(str(map_var_name))}"
            else:
                title_base = "CRPS Map"
            date_suffix = extract_date_from_filename(key)
            fig.suptitle(f"{title_base}{date_suffix}")

            out_png = dst_prob / (key + "_compare.png")
            plt.savefig(out_png, bbox_inches="tight", dpi=200)
            c.success(f"Saved {out_png.relative_to(out_root)}")
            plt.close(fig)

    # --- 5. Spread Skill Ratio (SSR) ---
    frames_ssr: list[pd.DataFrame] = []
    for lab, m in zip(labels, models, strict=False):
        candidates = sorted((m / "probabilistic").glob("spread_skill_ratio*.csv"))
        if candidates:
            try:
                df = pd.read_csv(candidates[0])
                df.insert(0, "model", lab)
                frames_ssr.append(df)
            except Exception:
                pass

    if frames_ssr:
        combined_ssr = pd.concat(frames_ssr, ignore_index=True)
        if combined_ssr["model"].nunique() >= 2:
            out_csv = dst_prob / "spread_skill_ratio_combined.csv"
            combined_ssr.to_csv(out_csv, index=False)
            c.success(f"Saved {out_csv.relative_to(out_root)}")

    # --- 5b. SSR Line Plots (Multi-Lead) ---
    # Look for ssr_line_<var>_by_lead_*.csv
    ssr_line_files = common_files(models, "probabilistic/ssr_line_*.csv")
    if ssr_line_files:
        print_file_list(f"Found {len(ssr_line_files)} common SSR line files", ssr_line_files)
        colors = sns.color_palette("tab10", n_colors=len(models))

        for base in ssr_line_files:
            # Extract variable name from filename: ssr_line_<var>_by_lead...
            # We can use the helper but need to be careful with the prefix
            var_name = base.replace("ssr_line_", "").split("_by_lead")[0]

            fig, ax = plt.subplots(figsize=(10, 6))
            has_data = False

            for idx, (lab, m) in enumerate(zip(labels, models, strict=False)):
                ssr_path = m / "probabilistic" / base
                if ssr_path.exists():
                    try:
                        df = pd.read_csv(ssr_path)
                        if "lead_time_hours" in df.columns and "SSR" in df.columns:
                            df = df.sort_values("lead_time_hours")
                            ax.plot(
                                df["lead_time_hours"],
                                df["SSR"],
                                label=lab,
                                color=colors[idx],
                                marker="o",
                            )
                            has_data = True
                    except Exception:
                        pass

            if has_data:
                ax.set_xlabel("Lead Time [h]")
                ax.set_ylabel("SSR")
                ax.set_title(f"SSR Evolution — {format_variable_name(var_name)}")
                ax.legend()
                ax.grid(True, linestyle=":", alpha=0.6)

                # Add ideal line
                ax.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="Ideal")

                out_png = dst_prob / base.replace(".csv", "_compare.png")
                fig.savefig(out_png, dpi=150)
                plt.close(fig)
                c.success(f"Saved {out_png.relative_to(out_root)}")
            else:
                plt.close(fig)

    # --- 6. CRPS Ensemble ---
    frames_ens: list[pd.DataFrame] = []
    for lab, m in zip(labels, models, strict=False):
        candidates = sorted((m / "probabilistic").glob("crps_ensemble*.csv"))
        if candidates:
            try:
                df = pd.read_csv(candidates[0])
                df.insert(0, "model", lab)
                frames_ens.append(df)
            except Exception:
                pass

    if frames_ens:
        combined_ens = pd.concat(frames_ens, ignore_index=True)
        if combined_ens["model"].nunique() >= 2:
            out_csv = dst_prob / "crps_ensemble_combined.csv"
            combined_ens.to_csv(out_csv, index=False)
            c.success(f"Saved {out_csv.relative_to(out_root)}")

    # --- 7. PIT Histograms ---
    common_pit = common_files(models, "probabilistic/pit_hist*.npz")
    if common_pit:
        print_file_list(f"Found {len(common_pit)} common PIT histogram files", common_pit)
        for base in common_pit:
            # Skip if it's a grid file or data file that we don't want to plot directly
            # Also skip per-lead files to avoid duplicates
            if "grid" in base or "data" in base or "lead" in base:
                continue

            # Special check for geopotential which might be named differently or skipped
            # If base contains 'geopotential', ensure we process it

            payloads = [load_npz(m / "probabilistic" / base) for m in models]

            # Check if we have counts and bins/edges
            if not all("counts" in p and ("bins" in p or "edges" in p) for p in payloads):
                continue

            fig, ax = plt.subplots(figsize=(10, 6), dpi=160)

            # Plot ideal uniform line (dashed)
            # Assuming normalized counts or we normalize them
            # Usually PIT histograms are normalized so that area is 1 or mean height is 1

            colors = sns.color_palette("tab10", n_colors=len(models))

            # Calculate width for side-by-side bars
            n_models = len(models)

            # First pass to get bins and check consistency
            bins0 = payloads[0].get("bins", payloads[0].get("edges"))
            width_total = np.diff(bins0)
            # Bar width for each model
            bar_width = width_total / n_models

            for i, (lab, pay) in enumerate(zip(labels, payloads, strict=False)):
                counts = pay["counts"]
                bins = pay.get("bins", pay.get("edges"))

                # Normalize if needed? Usually they are raw counts.
                # Let's plot density
                width = np.diff(bins)
                density = counts / (counts.sum() * width)

                # Plot as filled bars side-by-side
                # Shift x position based on model index
                x_pos = bins[:-1] + (i * bar_width)

                ax.bar(
                    x_pos,
                    density,
                    width=bar_width,
                    align="edge",
                    label=lab,
                    color=colors[i],
                    alpha=0.8,
                )

            # Add reference line at 1.0
            ax.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="Ideal")

            var = clean_var_from_filename(base, prefix="pit_hist_")
            ax.set_title(f"PIT Histogram — {var}")
            ax.set_ylabel("Probability Density")
            ax.set_xlabel("PIT Value")
            ax.set_ylim(bottom=0)
            ax.legend(frameon=False)
            ax.grid(True, linestyle="--", alpha=0.4)

            out_png = dst_prob / base.replace(".npz", "_compare.png")
            plt.savefig(out_png, bbox_inches="tight", dpi=200)
            c.success(f"Saved {out_png.relative_to(out_root)}")
            plt.close(fig)
