from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

    report_checklist("wd_kde", results)

    if not common and not common_g:
        c.warn("No common WD KDE files found. Skipping plots.")
        return

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
            # if units := payloads[0].get("units"):
            #     ax.set_xlabel(str(units))
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
                # if units:
                #     ax.set_xlabel(str(units))

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
