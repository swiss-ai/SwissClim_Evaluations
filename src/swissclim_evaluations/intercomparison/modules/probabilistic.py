from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    model_color_map,
    print_file_list,
    reorder_pivot_columns,
    report_checklist,
    report_missing,
    scan_model_sets,
)


def _parse_map_filename(name: str) -> str:
    return name[:-4] if name.endswith(".npz") else name


def _lead_time_to_hours(series: pd.Series) -> pd.Series:
    vals = pd.to_timedelta(series, errors="coerce")
    if vals.notna().any():
        hours = vals.dt.total_seconds() / 3600.0
        return hours.round().astype("Int64")
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def _normalize_summary_lead_time(df: pd.DataFrame) -> pd.DataFrame:
    if "lead_time" not in df.columns:
        return df
    out = df.copy()
    out["lead_time"] = _lead_time_to_hours(out["lead_time"])
    return out


def _intercompare_spaghetti(
    models: list[Path],
    labels: list[str],
    out_root: Path,
    dst_prob: Path,
    cmap: dict[str, tuple],
    common_spaghetti: list[str],
) -> None:
    """Generate gridded spaghetti comparison plots.

    For each common spaghetti NPZ file (one per variable/level), produces a
    figure with **models as columns** and **lead-time subsets as rows**.  Each
    panel shows all ensemble members as thin lines in the model's assigned
    colour and the shared ground-truth target as a thick black line.

    Because spaghetti plots already show the full lead-time range on the x-axis,
    the "rows" here represent individual model panels rather than slicing by
    lead time.  This gives a compact side-by-side view of ensemble spread
    across models.
    """
    if not common_spaghetti:
        return

    from swissclim_evaluations.helpers import (
        COLOR_GROUND_TRUTH,
        format_level_token,
    )

    print_file_list(
        f"Found {len(common_spaghetti)} common spaghetti NPZ files",
        common_spaghetti,
    )

    for base in common_spaghetti:
        payloads = []
        for m in models:
            p = load_npz(m / "probabilistic" / base)
            payloads.append(p)

        # Validate all payloads have required keys
        required_keys = {"lead_hours", "member_values", "target_values"}
        if not all(required_keys.issubset(p.keys()) for p in payloads):
            continue

        # Extract metadata from first model
        variable = str(payloads[0].get("variable", ""))
        units = str(payloads[0].get("units", ""))
        level = payloads[0].get("level")
        # NPZ stores scalars wrapped in 0-d arrays
        if hasattr(variable, "item"):
            variable = variable.item()
        if hasattr(units, "item"):
            units = units.item()
        if level is not None and hasattr(level, "item"):
            level = level.item()

        n_models = len(models)
        lead_hours_ref = np.asarray(payloads[0]["lead_hours"])

        # --- Gridded figure: 1 row, n_models columns ---
        fig, axes = plt.subplots(
            1,
            n_models,
            figsize=(5.5 * n_models, 4),
            dpi=160,
            sharey=True,
            constrained_layout=True,
        )
        if n_models == 1:
            axes = [axes]

        for mi, (ax, lab, pay) in enumerate(zip(axes, labels, payloads, strict=False)):
            lead_h = np.asarray(pay["lead_hours"])
            members = np.asarray(pay["member_values"])
            target = np.asarray(pay["target_values"])

            if members.ndim == 1:
                members = members.reshape(1, -1)
            n_mem = members.shape[0]
            model_color = cmap.get(lab, "#D55E00")

            # Ensemble member lines
            for em in range(n_mem):
                ax.plot(
                    lead_h,
                    members[em],
                    color=model_color,
                    alpha=max(0.15, min(0.6, 3.0 / n_mem)),
                    linewidth=0.8,
                    label="Members" if em == 0 else None,
                )

            # Target line (shared across models)
            ax.plot(
                lead_h,
                target,
                color=COLOR_GROUND_TRUTH,
                linewidth=2.0,
                label="Target",
                zorder=10,
            )

            ax.set_title(lab, fontsize=10)
            ax.set_xlabel("Lead Time [h]")
            if mi == 0:
                y_label = f"Spatial Mean [{units}]" if units else "Spatial Mean"
                ax.set_ylabel(y_label)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="best", fontsize=7, frameon=False)

        lvl_str = f" @ {format_level_token(level)}" if level is not None else ""
        fig.suptitle(
            f"Ensemble Spaghetti — " f"{format_variable_name(variable)}{lvl_str}",
            fontsize=12,
            y=1.02,
        )

        stem = base.replace(".npz", "")
        out_png = dst_prob / f"{stem}_compare.png"
        plt.savefig(out_png, bbox_inches="tight", dpi=200)
        c.success(f"Saved {out_png.relative_to(out_root)}")
        plt.close(fig)


def intercompare_probabilistic(
    models: list[Path],
    labels: list[str],
    out_root: Path,
    max_crps_map_panels: int = 4,
) -> None:
    dst_prob = ensure_dir(out_root / "probabilistic")
    _cmap = model_color_map(labels)

    # Availability report
    # Scan broadly for any CSV to catch CRPS, SSR, etc.
    per_model, _, uni = scan_model_sets(models, "probabilistic/*.csv")
    report_missing("probabilistic", models, labels, per_model, uni)

    results = {}
    all_prob = common_files(models, "probabilistic/crps_summary*.csv")

    # Check for different types
    avg = [f for f in all_prob if "per_lead_time" not in f and "per_level" not in f]
    lvl = [f for f in all_prob if "per_level" in f]
    lead = common_files(models, "probabilistic/crps_line*by_lead*.csv")

    results["CRPS Summary"] = 1 if avg else 0
    results["CRPS Summary (Per Level)"] = 1 if lvl else 0
    if lead:
        results["CRPS Line (Per Lead)"] = len(lead)

    # Maps
    maps = common_files(models, "probabilistic/crps_spatial_*.npz")
    results["CRPS Maps"] = len(maps)

    # SSR
    ssr = common_files(models, "probabilistic/ssr*.csv")
    ssr_lvl = [f for f in ssr if "per_level" in f]
    ssr_avg = [f for f in ssr if "per_level" not in f]
    results["SSR"] = 1 if ssr_avg else 0
    if ssr_lvl:
        results["SSR (Per Level)"] = 1

    # Ensemble
    ens = common_files(models, "probabilistic/crps_ensemble*.csv")
    results["CRPS Ensemble"] = 1 if ens else 0

    # PIT
    pit = common_files(models, "probabilistic/pit_hist*.npz")
    results["PIT Histograms"] = len(pit)

    # Spaghetti
    spaghetti = common_files(models, "probabilistic/spaghetti_*.npz")
    results["Spaghetti Timeseries"] = len(spaghetti)

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
        combined = _normalize_summary_lead_time(combined)
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
        # Find all per-lead CSVs (CRPS + SSR)
        csv_files = list((m / "probabilistic").glob("crps_line*by_lead*.csv"))
        csv_files.extend((m / "probabilistic").glob("ssr_line*by_lead*.csv"))
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
                    if fname.startswith("crps_line_") or fname.startswith("ssr_line_"):
                        prefix = "crps_line_" if fname.startswith("crps_line_") else "ssr_line_"
                        # Remove prefix
                        rest = fname[len(prefix) :]
                        # Remove suffix starting with _by_lead
                        if "_by_lead" in rest:
                            var_name = rest.split("_by_lead")[0]
                            level_token = var_name.rsplit("_", 1)[-1]
                            if level_token.isdigit():
                                df["level"] = int(level_token)
                                var_name = var_name.rsplit("_", 1)[0]
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
                pass

    if temporal_rows:
        temporal_df = pd.DataFrame(temporal_rows)
        if len(temporal_df) > 0 and temporal_df["model"].nunique() >= 2:
            out_csv = dst_prob / "temporal_metrics_combined.csv"
            temporal_df.to_csv(out_csv, index=False)
            c.success(f"Saved {out_csv.relative_to(out_root)}")

            # Plot: group by lead_time and model
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
                    pivot = reorder_pivot_columns(pivot, labels)
                    _line_colors = [_cmap[c] for c in pivot.columns if c in _cmap]
                    fig, ax = plt.subplots(figsize=(10, 6))
                    pivot.plot(
                        kind="line", ax=ax, marker="o", markersize=4, color=_line_colors or None
                    )

                    level_has_value = has_level and pd.notna(pair.get("level"))
                    level_title = f" @ {int(level_val)}" if level_has_value else ""
                    ax.set_title(
                        f"{metric} vs Lead Time — {format_variable_name(variable)}{level_title}"
                    )
                    ax.set_ylabel(metric)
                    ax.set_xlabel("Lead Time (h)")
                    ax.grid(True, linestyle="--", alpha=0.6)
                    plt.tight_layout()

                    out_png = dst_prob / f"temporal_{metric}_{variable}{level_token}_compare.png"
                    plt.savefig(out_png, bbox_inches="tight", dpi=200)
                    c.success(f"Saved {out_png.relative_to(out_root)}")
                    plt.close(fig)

    # --- 4. CRPS Maps ---
    common_maps = common_files(models, "probabilistic/crps_spatial_*.npz")
    if common_maps:
        print_file_list(f"Found {len(common_maps)} common CRPS map files", common_maps)

        # Limit panels
        seen_vars: set[str] = set()
        for base in common_maps:
            if len(seen_vars) >= max_crps_map_panels:
                break

            key = _parse_map_filename(base)
            # Check if we already processed this variable (ignoring date/ens suffix)
            var_part = clean_var_from_filename(base, prefix="crps_spatial_")
            if var_part in seen_vars:
                continue

            payloads = [load_npz(m / "probabilistic" / base) for m in models]

            # CRPS maps saved by CLI under 'data' key
            predictions = []
            for p in payloads:
                val = p.get("data")
                predictions.append(val)

            if any(x is None for x in predictions):
                continue

            lats = payloads[0].get("latitude")
            lons = payloads[0].get("longitude")
            lead_times = payloads[0].get("lead_time")
            map_var_name = payloads[0].get("variable")
            units = payloads[0].get("units")

            # Fallback: extract variable name from filename when NPZ lacks it
            if map_var_name is None:
                map_var_name = var_part if var_part else None

            if lats is None or lons is None or map_var_name is None:
                continue

            seen_vars.add(var_part)

            n_lead_times = len(lead_times) if lead_times is not None else 0

            # Normalise predictions to (lead_time, lat, lon) canonical order.
            # Old NPZ files may have been saved with arbitrary dim ordering (e.g.
            # latitude, longitude, lead_time from WBX). Detect which axis matches
            # n_lead_times and move it to position 0.
            if lead_times is not None and n_lead_times > 0:
                normalised = []
                for p in predictions:
                    if p.ndim == 3 and p.shape[0] != n_lead_times:
                        for ax in (1, 2):
                            if p.shape[ax] == n_lead_times:
                                p = np.moveaxis(p, ax, 0)
                                break
                    normalised.append(p)
                predictions = normalised

            # Determine lead-time labels (hours) if present
            has_lead = lead_times is not None and n_lead_times > 1 and predictions[0].ndim == 3

            if has_lead:
                # Convert lead_time to hours for labelling
                try:
                    lt_hours = lead_times.astype("timedelta64[h]").astype(float)
                except (TypeError, ValueError):
                    lt_hours = np.asarray(lead_times, dtype=float)
                n_leads = n_lead_times

                # ── Mean map (averaged over lead_time) ──
                mean_preds = [p.mean(axis=0) for p in predictions]
                ncols = len(models)
                fig_m, axes_m = plt.subplots(
                    1,
                    ncols,
                    figsize=(6 * ncols, 4),
                    dpi=160,
                    subplot_kw={"projection": ccrs.PlateCarree()},
                    constrained_layout=True,
                )
                if ncols == 1:
                    axes_m = [axes_m]
                try:
                    vmin = float(np.nanmin([np.nanmin(x) for x in mean_preds]))
                    vmax = float(np.nanmax([np.nanmax(x) for x in mean_preds]))
                except ValueError:
                    continue
                im0 = None
                for ax, lab, pred in zip(axes_m, labels, mean_preds, strict=False):
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
                    ax.set_title(lab)
                if im0:
                    cbar = fig_m.colorbar(
                        im0,
                        ax=axes_m if isinstance(axes_m, (list | np.ndarray)) else [axes_m],
                        orientation="horizontal",
                        fraction=0.05,
                        pad=0.08,
                    )
                    cbar.set_label(f"CRPS ({units})" if units else "CRPS")
                title_base = (
                    f"CRPS Map (mean) — {format_variable_name(str(map_var_name))}"
                    if map_var_name
                    else "CRPS Map (mean)"
                )
                date_suffix = extract_date_from_filename(key)
                fig_m.suptitle(f"{title_base}{date_suffix}", y=1.02)
                out_mean = dst_prob / (key + "_mean_compare.png")
                plt.savefig(out_mean, bbox_inches="tight", dpi=200)
                c.success(f"Saved {out_mean.relative_to(out_root)}")
                plt.close(fig_m)

                # ── Per-lead-time grid (rows=lead_time, cols=models) ──
                ncols = len(models)
                nrows = n_leads
                fig, axes = plt.subplots(
                    nrows,
                    ncols,
                    figsize=(5 * ncols, 3.5 * nrows),
                    dpi=160,
                    subplot_kw={"projection": ccrs.PlateCarree()},
                    constrained_layout=True,
                )
                axes = np.atleast_2d(axes)
                # Global vmin/vmax across all leads and models
                try:
                    vmin = float(np.nanmin([np.nanmin(x) for x in predictions]))
                    vmax = float(np.nanmax([np.nanmax(x) for x in predictions]))
                except ValueError:
                    continue
                im0 = None
                for li in range(nrows):
                    lt_label = f"(+{int(lt_hours[li])}h)" if li < len(lt_hours) else f"lead {li}"
                    for mi, (lab, pred) in enumerate(zip(labels, predictions, strict=False)):
                        ax = axes[li, mi]
                        im0 = ax.pcolormesh(
                            lons,
                            lats,
                            pred[li],
                            cmap="viridis",
                            vmin=vmin,
                            vmax=vmax,
                            transform=ccrs.PlateCarree(),
                        )
                        ax.coastlines(linewidth=0.5)
                        if li == 0:
                            ax.set_title(lab, fontsize=9)
                        if mi == 0:
                            ax.text(
                                -0.08,
                                0.5,
                                lt_label,
                                transform=ax.transAxes,
                                rotation=90,
                                va="center",
                                ha="right",
                                fontsize=9,
                                fontweight="bold",
                            )
                if im0:
                    cbar = fig.colorbar(
                        im0,
                        ax=axes.ravel().tolist(),
                        orientation="horizontal",
                        fraction=0.03,
                        pad=0.04,
                    )
                    cbar.set_label(f"CRPS ({units})" if units else "CRPS")
                title_base = (
                    f"CRPS Map per Lead — {format_variable_name(str(map_var_name))}"
                    if map_var_name
                    else "CRPS Map per Lead"
                )
                date_suffix = extract_date_from_filename(key)
                fig.suptitle(f"{title_base}{date_suffix}", y=1.02)
                out_lead = dst_prob / (key + "_per_lead_compare.png")
                plt.savefig(out_lead, bbox_inches="tight", dpi=200)
                c.success(f"Saved {out_lead.relative_to(out_root)}")
                plt.close(fig)
            else:
                # Data is 2D or single lead — plot as before
                # If 3D with single lead, squeeze
                preds_2d = []
                for p in predictions:
                    if p.ndim == 3 and p.shape[0] == 1:
                        preds_2d.append(p[0])
                    else:
                        preds_2d.append(p)
                predictions = preds_2d

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
                fig.suptitle(f"{title_base}{date_suffix}", y=1.02)

                out_png = dst_prob / (key + "_compare.png")
                plt.savefig(out_png, bbox_inches="tight", dpi=200)
                c.success(f"Saved {out_png.relative_to(out_root)}")
                plt.close(fig)

    # --- 5. Spread Skill Ratio (SSR) ---
    frames_ssr: list[pd.DataFrame] = []
    frames_ssr_lvl: list[pd.DataFrame] = []
    for lab, m in zip(labels, models, strict=False):
        candidates_avg = sorted(
            c
            for c in (m / "probabilistic").glob("ssr*.csv")
            if "per_level" not in c.name and not c.name.startswith("ssr_line_")
        )
        candidates_lvl = sorted(
            c for c in (m / "probabilistic").glob("ssr*per_level*.csv") if "per_level" in c.name
        )
        if candidates_avg:
            try:
                df = pd.read_csv(candidates_avg[0])
                df.insert(0, "model", lab)
                frames_ssr.append(df)
            except Exception:
                pass
        if candidates_lvl:
            try:
                df = pd.read_csv(candidates_lvl[0])
                df.insert(0, "model", lab)
                frames_ssr_lvl.append(df)
            except Exception:
                pass

    if frames_ssr:
        combined_ssr = pd.concat(frames_ssr, ignore_index=True)
        combined_ssr = _normalize_summary_lead_time(combined_ssr)
        if combined_ssr["model"].nunique() >= 2:
            out_csv = dst_prob / "ssr_combined.csv"
            combined_ssr.to_csv(out_csv, index=False)
            c.success(f"Saved {out_csv.relative_to(out_root)}")

    if frames_ssr_lvl:
        combined_ssr_lvl = pd.concat(frames_ssr_lvl, ignore_index=True)
        if combined_ssr_lvl["model"].nunique() >= 2:
            out_csv = dst_prob / "ssr_per_level_combined.csv"
            combined_ssr_lvl.to_csv(out_csv, index=False)
            c.success(f"Saved {out_csv.relative_to(out_root)}")

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

        # Group grid files by variable so we can aggregate per-lead counts
        # Grid NPZs: pit_hist_<var>_grid_ensprob.npz  (counts shape = n_leads × n_bins)
        # Non-grid NPZs: pit_hist_<var>_ensprob.npz   (counts shape = n_bins)
        processed_vars: set[str] = set()

        for base in common_pit:
            # Skip auxiliary files we can't plot
            if "data" in base:
                continue

            is_grid = "grid" in base
            # Derive the variable key without the "grid" qualifier so we can
            # check whether a non-grid file already covered this variable.
            var_key = base.replace("_grid", "") if is_grid else base

            if var_key in processed_vars:
                continue
            processed_vars.add(var_key)

            payloads = [load_npz(m / "probabilistic" / base) for m in models]

            # For grid files the counts array is 2-D (n_leads × n_bins).
            # Aggregate across leads to get a single histogram per model.
            if is_grid:
                agg_payloads = []
                for p in payloads:
                    counts_raw = p.get("counts")
                    edges = p.get("edges", p.get("bins"))
                    if counts_raw is None or edges is None:
                        agg_payloads.append(p)
                        continue
                    counts_arr = np.asarray(counts_raw)
                    if counts_arr.ndim == 2:
                        # Sum across leads, then re-normalise
                        counts_sum = counts_arr.sum(axis=0)
                        width = np.diff(edges)
                        total = counts_sum.sum()
                        density = counts_sum / (total * width.mean()) if total > 0 else counts_sum
                        agg_payloads.append({**p, "counts": density, "edges": edges})
                    else:
                        agg_payloads.append(p)
                payloads = agg_payloads

            # Check if we have counts and bins/edges
            if not all("counts" in p and ("bins" in p or "edges" in p) for p in payloads):
                continue

            fig, ax = plt.subplots(figsize=(10, 6), dpi=160)

            # Plot ideal uniform line (dashed)
            # Assuming normalized counts or we normalize them
            # Usually PIT histograms are normalized so that area is 1 or mean height is 1

            colors = [_cmap[lab] for lab in labels if lab in _cmap]

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

                # Guard: skip model if bin count doesn't match reference
                if len(bins) != len(bins0):
                    c.warn(f"Skipping {lab}: bin count {len(bins)} != {len(bins0)}")
                    continue

                # Normalize to density; guard against empty histogram
                width = np.diff(bins)
                total = counts.sum()
                if total == 0:
                    density = np.zeros_like(counts, dtype=float)
                else:
                    density = counts / (total * width)

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

    # --- 8. Spaghetti Timeseries Comparison ---
    _intercompare_spaghetti(models, labels, out_root, dst_prob, _cmap, spaghetti)
