from __future__ import annotations

import contextlib
from pathlib import Path

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
    common_files,
    ensure_dir,
    load_npz,
    print_file_list,
    report_checklist,
    report_missing,
)


def _find_vertical_profile_files(models: list[Path]) -> list[str]:
    """Return common vertical profile NPZ basenames (current schema).

    New naming uses the standardized builder:
      vertical_profiles_nmae_<variable>_multi_combined[_init...][_lead...]_ens*.npz
    We locate files by the stable prefix and the "_combined" qualifier.
    Returns sorted list of basenames existing across all model folders.
    """
    vp_dir = Path("vertical_profiles")
    patterns = [
        "vertical_profiles_nmae_*_combined*.npz",  # current
        "vprof_nmae_*_combined*.npz",  # legacy
        "*_pl_nmae_combined*.npz",  # legacy fallback
    ]
    # Build intersection over models; if multiple patterns match, union per model first
    sets: list[set[str]] = []
    for m in models:
        model_files: set[str] = set()
        for pat in patterns:
            model_files.update({f.name for f in (m / vp_dir).glob(pat) if f.is_file()})
        sets.append(model_files)
    if not sets:
        return []
    common = set.intersection(*sets) if len(sets) > 1 else sets[0]
    return sorted(common)


def _find_vertical_profile_global_files(models: list[Path]) -> list[str]:
    """Return common global vertical profile NPZ basenames."""
    vp_dir = Path("vertical_profiles")
    patterns = [
        "vertical_profiles_nmae_*_global_profile*.npz",
    ]
    sets: list[set[str]] = []
    for m in models:
        model_files: set[str] = set()
        for pat in patterns:
            model_files.update({f.name for f in (m / vp_dir).glob(pat) if f.is_file()})
        sets.append(model_files)
    if not sets:
        return []
    common = set.intersection(*sets) if len(sets) > 1 else sets[0]
    return sorted(common)


def intercompare_vertical_profiles(models: list[Path], labels: list[str], out_root: Path) -> None:
    """Overlay vertical profile NMAE (or legacy relative error) curves.

    For each variable present in all model folders we create per-lat-band figure
    (mirrors original 9 south + 9 north band layout => 9 rows x 2 cols) with DS
    (target) not expressly stored. The NPZ files only contain metric curves
    already reduced vs. level; DS baseline is implicit (NMAE uses target stats).

    We therefore only plot model curves. If legacy rel_error files are used we
    label plots accordingly.
    """
    # Availability report
    per_model_vp: list[set[str]] = []
    for m in models:
        s: set[str] = set()
        for pat in (
            "vertical_profiles_nmae_*_combined*.npz",
            "vprof_nmae_*_combined*.npz",
            "*_pl_nmae_combined*.npz",
        ):
            s.update({f.name for f in (m / "vertical_profiles").glob(pat) if f.is_file()})
        per_model_vp.append(s)
    union_vp = set().union(*per_model_vp) if per_model_vp else set()
    if union_vp:
        report_missing("vertical_profiles", models, labels, per_model_vp, union_vp)

    results = {}
    all_vprof = common_files(models, "vertical_profiles/vprof_*.npz")
    processed = _find_vertical_profile_files(models)

    results["Vertical Profiles"] = len(processed)

    processed_set = set(processed)
    ignored_count = sum(1 for f in all_vprof if f not in processed_set)
    if ignored_count > 0:
        results["Vertical Profiles (Ignored)"] = ignored_count

    report_checklist("vertical_profiles", results)

    basenames = _find_vertical_profile_files(models)
    if not basenames:
        c.warn("No common vertical profile files found. Skipping plots.")
        return
    print_file_list(f"Found {len(basenames)} common vertical profile files", basenames)
    dst = ensure_dir(out_root / "vertical_profiles")
    color_palette = sns.color_palette("tab10", n_colors=len(models))
    for base in basenames:
        payloads = []
        for m in models:
            try:
                payloads.append(load_npz(m / "vertical_profiles" / base))
            except Exception:
                payloads.append({})
        # Require at least two models with payload
        valid_models = [
            p for p in payloads if p.get("nmae_pos") is not None and p.get("nmae_neg") is not None
        ]
        if len(valid_models) < 2:
            continue
        key_neg = "nmae_neg"
        key_pos = "nmae_pos"
        if key_neg not in payloads[0] or key_pos not in payloads[0]:
            continue
        neg_arr0 = np.asarray(payloads[0][key_neg])
        bands = neg_arr0.shape[0]
        neg_lat_min = payloads[0].get("neg_lat_min")
        neg_lat_max = payloads[0].get("neg_lat_max")
        pos_lat_min = payloads[0].get("pos_lat_min")
        pos_lat_max = payloads[0].get("pos_lat_max")
        level_values = payloads[0].get("level")
        if level_values is None:
            continue

        # vertical_profiles.py does:
        # Row 0: South
        # Row 1: North
        fig, axs = plt.subplots(2, bands, figsize=(24, 10), dpi=160, sharey=True, sharex=True)

        for j in range(bands):
            # South (Row 0)
            axsou = axs[0, j]
            for idx, (lab, pay) in enumerate(zip(labels, payloads, strict=False)):
                arr = np.asarray(pay.get(key_neg))
                if arr is None or arr.shape[0] <= j:
                    continue
                axsou.plot(arr[j], level_values, label=lab, color=color_palette[idx])
            if neg_lat_min is not None and neg_lat_max is not None:
                axsou.set_title(f"Lat {float(neg_lat_min[j])}° to {float(neg_lat_max[j])}° (South)")

            if j == 0:
                axsou.set_ylabel("Level")
                # Only invert once because axes are shared
                axsou.invert_yaxis()

            # North (Row 1)
            axn = axs[1, j]
            for idx, (lab, pay) in enumerate(zip(labels, payloads, strict=False)):
                arr = np.asarray(pay.get(key_pos))
                if arr is None or arr.shape[0] <= j:
                    continue
                axn.plot(arr[j], level_values, label=lab, color=color_palette[idx])
            if pos_lat_min is not None and pos_lat_max is not None:
                axn.set_title(f"Lat {float(pos_lat_min[j])}° to {float(pos_lat_max[j])}° (North)")

            axn.set_xlabel("NMAE (%)")
            if j == 0:
                axn.set_ylabel("Level")

        handles, labels_leg = axs[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels_leg,
                loc="lower center",
                ncol=min(6, len(models)),
            )
        # Derive variable name from filename robustly
        var = base[:-4] if base.endswith(".npz") else base
        if var.startswith("vertical_profiles_nmae_"):
            # vertical_profiles_nmae_<variable>_multi_combined[...]
            tail = var[len("vertical_profiles_nmae_") :]
            if "_multi_combined" in tail:
                var = tail.split("_multi_combined", 1)[0]
            else:
                var = tail.split("_combined", 1)[0]
        elif var.startswith("vprof_nmae_"):
            # vprof_nmae_<variable>_multi_combined[...]
            tail = var[len("vprof_nmae_") :]
            if "_multi_combined" in tail:
                var = tail.split("_multi_combined", 1)[0]
            else:  # fallback: strip from first _combined
                var = tail.split("_combined", 1)[0]
        else:
            # legacy: <variable>_pl_nmae_combined[...]
            var = var.replace("_pl_nmae_combined", "").replace("_pl_rel_error_combined", "")

        # Extract date info from filename if possible
        date_suffix = extract_date_from_filename(base)

        fig.suptitle(
            f"Vertical Profiles of NMAE for {format_variable_name(var)} (band-wise){date_suffix}",
            y=1.02,
            fontsize=24,
        )
        plt.tight_layout(rect=(0, 0.04, 1, 1))
        out_png = dst / base.replace(".npz", "_compare.png")
        # Save only if at least two models contributed lines
        plt.savefig(out_png, bbox_inches="tight", dpi=200)
        c.success(f"Saved {out_png.relative_to(out_root)}")
        plt.close(fig)
        rows = []
        for lab, pay in zip(labels, payloads, strict=False):
            neg_arr = np.asarray(pay.get(key_neg))
            pos_arr = np.asarray(pay.get(key_pos))
            if neg_arr is None or pos_arr is None:
                continue
            for j in range(bands):
                with np.errstate(all="ignore"):
                    val_pos = np.nanmean(pos_arr[j]) if pos_arr[j].size else np.nan
                rows.append(
                    {
                        "variable": var,
                        "band_index": j,
                        "hemisphere": "north",
                        "model": lab,
                        "value": float(val_pos) if np.isfinite(val_pos) else np.nan,
                        "metric": "NMAE",
                    }
                )
                rows.append(
                    {
                        "variable": var,
                        "band_index": j,
                        "hemisphere": "south",
                        "model": lab,
                        "value": float(np.nanmean(neg_arr[j])) if neg_arr[j].size else np.nan,
                        "metric": "NMAE",
                    }
                )
        # Save summary only if we have at least two distinct models with values
        if rows:
            df = pd.DataFrame(rows)
            if df["model"].nunique() >= 2:
                out_csv = dst / base.replace(".npz", "_summary.csv")
                df.to_csv(out_csv, index=False)
                c.success(f"Saved {out_csv.relative_to(out_root)}")

    # Global Profiles
    global_basenames = _find_vertical_profile_global_files(models)
    if global_basenames:
        print_file_list(
            f"Found {len(global_basenames)} common global vertical profile files", global_basenames
        )
        for base in global_basenames:
            payloads = []
            for m in models:
                try:
                    payloads.append(load_npz(m / "vertical_profiles" / base))
                except Exception:
                    payloads.append({})

            # Check validity
            valid_models = [
                p for p in payloads if p.get("nmae") is not None and p.get("level") is not None
            ]
            if len(valid_models) < 2:
                continue

            level_values = payloads[0]["level"]

            fig, ax = plt.subplots(figsize=(8, 10), dpi=160)
            for idx, (lab, pay) in enumerate(zip(labels, payloads, strict=False)):
                arr = np.asarray(pay.get("nmae"))
                if arr is None:
                    continue
                ax.plot(arr, level_values, label=lab, color=color_palette[idx], linewidth=2)

            ax.set_xlabel("NMAE [%]")
            ax.set_ylabel("Level")
            ax.invert_yaxis()
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend()

            # Variable name extraction
            var = base[:-4]
            if "vertical_profiles_nmae_" in var:
                with contextlib.suppress(IndexError):
                    var = var.split("vertical_profiles_nmae_")[1].split("_multi_global_profile")[0]

            date_suffix = extract_date_from_filename(base)
            ax.set_title(
                f"Global Vertical Profile — {format_variable_name(var)}{date_suffix}", fontsize=14
            )

            out_png = dst / base.replace(".npz", "_compare.png")
            plt.savefig(out_png, bbox_inches="tight", dpi=200)
            c.success(f"Saved {out_png.relative_to(out_root)}")
            plt.close(fig)
