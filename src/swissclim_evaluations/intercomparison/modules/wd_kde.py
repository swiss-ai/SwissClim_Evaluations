from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from swissclim_evaluations.helpers import (
    COLOR_GROUND_TRUTH,
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


def intercompare_wd_kde(
    models: list[Path],
    labels: list[str],
    out_root: Path,
    max_models_in_legend: int = 12,
) -> None:
    src_rel = Path("wd_kde")
    dst = ensure_dir(out_root / "wd_kde")
    colors = sns.color_palette("tab10", n_colors=len(models))

    # Availability report
    patterns = [
        "wd_kde/wd_kde_*global*.npz",
        "wd_kde/wd_kde_*latbands*.npz",
        "wd_kde/wd_kde_evolve_*ridgeline_data*.npz",
        "wd_kde/wd_kde_wasserstein_averaged_*.csv",
    ]
    total_per_model: list[set[str]] = [set() for _ in models]
    total_union = set()

    for pat in patterns:
        pm, _, uni = scan_model_sets(models, pat)
        for i, s in enumerate(pm):
            total_per_model[i].update(s)
        total_union.update(uni)

    report_missing("wd_kde", models, labels, total_per_model, total_union)

    results = {}

    # --- Global KDE ---
    common_g = common_files(models, str(src_rel / "wd_kde_*global*.npz"))
    results["KDE Plots (Global)"] = len(common_g)

    # --- Latitude Bands KDE ---
    common = common_files(models, str(src_rel / "wd_kde_*latbands*.npz"))
    results["KDE Plots (Latbands)"] = len(common)

    # Wasserstein
    common_w = common_files(models, str(src_rel / "wd_kde_wasserstein_averaged_*.csv"))
    results["Wasserstein Metrics"] = 1 if common_w else 0

    # KDE evolution ridgeline data
    common_ridge = common_files(models, str(src_rel / "wd_kde_evolve_*ridgeline_data*.npz"))
    results["KDE Evolution (Ridgeline)"] = len(common_ridge)

    report_checklist("wd_kde", results)

    if not common and not common_g and not common_ridge:
        c.warn("No common WD KDE files found. Skipping plots.")
        return

    # Process Ridgeline evolution compare (minimal output: one compare PNG per common file)
    if common_ridge:
        print_file_list(f"Found {len(common_ridge)} common ridgeline KDE files", common_ridge)
        for base in common_ridge:
            payloads = [load_npz(m / src_rel / base) for m in models]

            lead_hours = np.asarray(payloads[0].get("lead_hours", []), dtype=float)
            y_eval = np.asarray(payloads[0].get("y_eval", []), dtype=float)
            density_target = np.asarray(payloads[0].get("density_target", []), dtype=float)

            if lead_hours.size == 0 or y_eval.size == 0 or density_target.size == 0:
                continue

            # Compute a stable vertical spacing from all available curves
            max_vals = [float(np.nanmax(density_target)) if density_target.size else 0.0]
            for pay in payloads:
                dm = np.asarray(pay.get("density_model", []), dtype=float)
                if dm.size:
                    max_vals.append(float(np.nanmax(dm)))
            offset = 1.05 * max(1e-6, max(max_vals))

            fig, ax = plt.subplots(figsize=(10, 6), dpi=160, constrained_layout=True)

            for i, h in enumerate(lead_hours.tolist()):
                base_y = i * offset

                # Target from first payload in black
                if i < density_target.shape[0]:
                    ax.plot(y_eval, base_y + density_target[i], color=COLOR_GROUND_TRUTH, lw=1.1)

                # Models in color with transparency
                for mi, (_, pay) in enumerate(zip(labels, payloads, strict=False)):
                    dm = np.asarray(pay.get("density_model", []), dtype=float)
                    if dm.ndim == 2 and i < dm.shape[0]:
                        ax.plot(
                            y_eval,
                            base_y + dm[i],
                            color=colors[mi],
                            alpha=0.35,
                            lw=0.9,
                        )

                ax.text(y_eval[-1] + (y_eval[1] - y_eval[0]) * 0.5, base_y + 0.02, f"{int(h)}h")

            var = clean_var_from_filename(base, prefix="wd_kde_evolve_")
            date_suffix = extract_date_from_filename(base)
            ax.set_title(
                f"KDE Evolution (Ridgeline) — {format_variable_name(var)}{date_suffix}",
                fontsize=14,
            )
            ax.set_xlabel("Standardized value")
            ax.set_yticks([])

            # Legend: target + models
            n_legend_models = min(len(labels), max_models_in_legend)
            if n_legend_models > 0:
                target_proxy = Line2D([0], [0], color=COLOR_GROUND_TRUTH, lw=1.1)
                model_proxies = [
                    Line2D([0], [0], color=colors[i], lw=1.2, alpha=0.8)
                    for i in range(n_legend_models)
                ]
                legend_handles = [target_proxy, *model_proxies]
                legend_labels = ["Target", *labels[:n_legend_models]]
                ax.legend(legend_handles, legend_labels, loc="upper right", frameon=False)

            out_png = dst / base.replace("_ridgeline_data.npz", "_ridgeline_compare.png")
            fig.savefig(out_png, bbox_inches="tight")
            plt.close(fig)
            c.success(f"Saved {out_png.relative_to(out_root)}")

    # Process Global
    if common_g:
        print_file_list(f"Found {len(common_g)} common global KDE files", common_g)
        for base in common_g:
            payloads = [load_npz(m / src_rel / base) for m in models]
            fig, ax = plt.subplots(figsize=(10, 6), dpi=160)

            # Target (from first model)
            x_ds = payloads[0]["x"]
            kde_ds = payloads[0]["kde_ds"]
            ax.plot(x_ds, kde_ds, color=COLOR_GROUND_TRUTH, lw=2.0, label="Target")

            # Models
            for i, (lab, pay) in enumerate(zip(labels, payloads, strict=False)):
                x_prediction = pay["x"]
                kde_prediction = pay["kde_prediction"]
                ax.plot(x_prediction, kde_prediction, color=colors[i], label=lab)

            var = clean_var_from_filename(base, prefix="wd_kde_")
            date_suffix = extract_date_from_filename(base)
            ax.set_title(
                f"Global Normalized KDE — {format_variable_name(var)}{date_suffix}", fontsize=16
            )
            # Note: units for the x-axis are available via payloads[0].get("units") if needed.
            ax.legend()

            out_png = dst / base.replace(".npz", "_compare.png")
            fig.savefig(out_png, bbox_inches="tight")
            plt.close(fig)
            c.success(f"Saved {out_png.relative_to(out_root)}")

    # Process Latbands
    if common:
        print_file_list(f"Found {len(common)} common latbands KDE files", common)
        # colors already defined
        for base in common:
            payloads = [load_npz(m / src_rel / base) for m in models]
            # units = payloads[0].get("units")
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
                        pay["neg_kde_prediction"][j],
                        color=colors[i],
                        label=lab,
                    )
                lat_min = float(payloads[0]["neg_lat_min"][j])
                lat_max = float(payloads[0]["neg_lat_max"][j])
                ax.set_title(f"Lat {lat_min}° to {lat_max}° (South)")
                # if units:
                #     ax.set_xlabel(str(units))

            # North (left)
            for j in range(n_rows):
                ax = axs[j, 0]
                x_ds = payloads[0]["pos_x"][j]
                kde_ds = payloads[0]["pos_kde_ds"][j]
                ax.plot(x_ds, kde_ds, color=COLOR_GROUND_TRUTH, lw=2.0, label="Target")

                for i, (lab, pay) in enumerate(zip(labels, payloads, strict=False)):
                    ax.plot(
                        pay["pos_x"][j],
                        pay["pos_kde_prediction"][j],
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
                    bbox_to_anchor=(0.5, 0.02),
                )
            plt.tight_layout(rect=(0, 0.08, 1, 1))
            # Extract variable/level part.
            var = clean_var_from_filename(base, prefix="wd_kde_")
            date_suffix = extract_date_from_filename(base)

            fig.suptitle(
                f"Normalized KDE by Latitude Bands — {format_variable_name(var)}{date_suffix}",
                y=1.02,
                fontsize=20,
            )
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
