from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from swissclim_evaluations import console as c
from swissclim_evaluations.intercomparison.core import (
    common_files,
    ensure_dir,
    print_file_list,
    report_checklist,
    report_missing,
    scan_model_sets,
)


def intercompare_ssim(models: list[Path], labels: list[str], out_root: Path) -> None:
    """Combine SSIM metrics from multiple models."""
    # Availability report
    per_model, _, uni = scan_model_sets(models, "ssim/ssim_ssim_*.csv")
    report_missing("ssim_ssim", models, labels, per_model, uni)

    results = {}
    all_multi = common_files(models, "ssim/ssim_ssim_*.csv")

    # Check for any files
    results["SSIM Summary"] = 1 if all_multi else 0

    report_checklist("ssim_ssim", results)

    # Report common files found
    multi_csv = common_files(models, "ssim/ssim_ssim_*.csv")
    if multi_csv:
        print_file_list(f"Found {len(multi_csv)} common SSIM metric files", multi_csv)

    # SSIM metrics
    dst_multi = ensure_dir(out_root / "ssim")
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
