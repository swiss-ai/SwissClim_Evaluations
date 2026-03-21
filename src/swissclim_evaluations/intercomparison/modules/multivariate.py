from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from swissclim_evaluations import console as c
from swissclim_evaluations.helpers import format_variable_name
from swissclim_evaluations.intercomparison.core import (
    common_files,
    ensure_dir,
    print_file_list,
    report_checklist,
    report_missing,
    scan_model_sets,
)
from swissclim_evaluations.plots.bivariate_histograms import plot_bivariate_histogram


def _load_hist_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        return {k: payload[k] for k in payload.files}


def _scalar_str(value: np.ndarray | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0])
        return None
    return str(value)


def _scalar_float(value: np.ndarray | float | None) -> float | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    val = float(arr.reshape(-1)[0])
    if np.isnan(val):
        return None
    return val


def _infer_var_pair(fname: str, payload: dict[str, np.ndarray]) -> tuple[str, str]:
    var_x = _scalar_str(payload.get("var_x"))
    var_y = _scalar_str(payload.get("var_y"))
    if var_x and var_y:
        return var_x, var_y

    stem = fname.replace("bivariate_hist_", "").replace(".npz", "")
    stem = re.sub(r"_ens[a-zA-Z0-9]+$", "", stem)

    known_pairs = [
        ("temperature", "specific_humidity"),
        ("specific_humidity", "temperature"),
        ("geopotential_height", "wind_speed"),
        ("wind_speed", "geopotential_height"),
        ("geopotential_height_gradient", "wind_speed"),
        ("wind_speed", "geopotential_height_gradient"),
    ]
    for a, b in known_pairs:
        token = f"{a}_{b}"
        if stem == token:
            return a, b

    return "", ""


def intercompare_multivariate(models: list[Path], labels: list[str], out_root: Path) -> None:
    """Compare multivariate bivariate-histogram artifacts across models."""
    pattern = "multivariate/bivariate_hist_*.npz"

    per_model, _, uni = scan_model_sets(models, pattern)
    report_missing("multivariate", models, labels, per_model, uni)

    common = common_files(models, pattern)
    report_checklist("multivariate", {"Bivariate histograms": len(common)})

    if not common:
        c.warn("No common multivariate NPZ files found. Skipping multivariate intercomparison.")
        return

    print_file_list(f"Found {len(common)} common multivariate files", common)
    dst = ensure_dir(out_root / "multivariate")

    for fname in common:
        first_payload = _load_hist_payload(models[0] / "multivariate" / fname)
        if not {"hist_target", "bins_x", "bins_y"}.issubset(first_payload):
            c.warn(f"[multivariate] Skipping {fname}: missing required target/bin arrays")
            continue

        var_x, var_y = _infer_var_pair(fname, first_payload)
        level_hpa = _scalar_float(first_payload.get("level_hpa"))

        ref_hist_target = np.asarray(first_payload["hist_target"])
        ref_bins_x = np.asarray(first_payload["bins_x"])
        ref_bins_y = np.asarray(first_payload["bins_y"])

        model_entries: list[dict[str, np.ndarray | str]] = []

        for label, model_dir in zip(labels, models, strict=False):
            payload = _load_hist_payload(model_dir / "multivariate" / fname)
            required = {"hist", "hist_target", "bins_x", "bins_y"}
            if not required.issubset(payload):
                c.warn(
                    f"[multivariate] Missing required arrays in "
                    f"{model_dir / 'multivariate' / fname}"
                )
                continue

            hist = np.asarray(payload["hist"])
            hist_target = np.asarray(payload["hist_target"])
            bins_x = np.asarray(payload["bins_x"])
            bins_y = np.asarray(payload["bins_y"])

            if hist.shape != hist_target.shape:
                c.warn(
                    f"[multivariate] Shape mismatch for {fname} in {label}: "
                    f"model={hist.shape}, target={hist_target.shape}; skipped"
                )
                continue

            model_entries.append(
                {
                    "label": label,
                    "hist": hist,
                    "hist_target": hist_target,
                    "bins_x": bins_x,
                    "bins_y": bins_y,
                }
            )

        if not model_entries:
            c.warn(f"[multivariate] No valid model histograms for {fname}; skipped")
            continue

        # Use common axis limits across all model panels for this pair.
        # This guarantees directly comparable visual placement between subplots.
        all_x_min = min(float(np.asarray(entry["bins_x"]).min()) for entry in model_entries)
        all_x_max = max(float(np.asarray(entry["bins_x"]).max()) for entry in model_entries)
        all_y_min = min(float(np.asarray(entry["bins_y"]).min()) for entry in model_entries)
        all_y_max = max(float(np.asarray(entry["bins_y"]).max()) for entry in model_entries)

        x_range = all_x_max - all_x_min
        y_range = all_y_max - all_y_min
        x_center = (all_x_max + all_x_min) / 2.0
        y_center = (all_y_max + all_y_min) / 2.0
        shared_xlim = (x_center - 0.625 * x_range, x_center + 0.625 * x_range)
        shared_ylim = (y_center - 0.625 * y_range, y_center + 0.625 * y_range)

        # Warn when models were produced on different histogram grids.
        off_grid_labels = [
            str(entry["label"])
            for entry in model_entries
            if not (
                np.array_equal(np.asarray(entry["bins_x"]), ref_bins_x)
                and np.array_equal(np.asarray(entry["bins_y"]), ref_bins_y)
            )
        ]
        if off_grid_labels:
            c.warn(
                "[multivariate] Inconsistent histogram bin grids detected in "
                f"{fname}. Using per-model bins/target to avoid contour misalignment. "
                f"Affected: {', '.join(off_grid_labels)}"
            )

        n_cols = len(model_entries)
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6), squeeze=False)

        for idx, entry in enumerate(model_entries):
            ax = axes[0, idx]
            hist = np.asarray(entry["hist"])
            hist_target = np.asarray(entry["hist_target"])
            bins_x = np.asarray(entry["bins_x"])
            bins_y = np.asarray(entry["bins_y"])
            label = str(entry["label"])
            plot_bivariate_histogram(
                hist_1=hist,
                hist_2=hist_target,
                bins_x=bins_x,
                bins_y=bins_y,
                label_1=label,
                label_2="Target",
                var_x=var_x,
                var_y=var_y,
                level_hpa=level_hpa,
                ax=ax,
                xlabel=format_variable_name(var_x) if var_x else None,
                ylabel=format_variable_name(var_y) if var_y else None,
                xlim=shared_xlim,
                ylim=shared_ylim,
            )
            ax.set_title(label)

        if var_x and var_y:
            title = f"{format_variable_name(var_x)} vs {format_variable_name(var_y)}"
            if level_hpa is not None:
                title += f" ({level_hpa:g} hPa)"
            fig.suptitle(title)
        else:
            fig.suptitle(fname.replace("bivariate_hist_", "").replace(".npz", ""))
        fig.tight_layout()

        stem = fname.replace("bivariate_hist_", "").replace(".npz", "")
        out_png = dst / f"bivariate_{stem}_compare.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

        out_npz = dst / f"bivariate_hist_{stem}_compare.npz"
        valid_labels = [str(entry["label"]) for entry in model_entries]
        uniform_grid = all(
            np.array_equal(np.asarray(entry["bins_x"]), ref_bins_x)
            and np.array_equal(np.asarray(entry["bins_y"]), ref_bins_y)
            for entry in model_entries
        )
        uniform_target = all(
            np.array_equal(np.asarray(entry["hist_target"]), ref_hist_target)
            for entry in model_entries
        )

        if uniform_grid and uniform_target:
            np.savez(
                out_npz,
                bins_x=ref_bins_x,
                bins_y=ref_bins_y,
                hist_target=ref_hist_target,
                var_x=var_x,
                var_y=var_y,
                level_hpa=np.nan if level_hpa is None else level_hpa,
                model_labels=np.array(valid_labels, dtype=object),
                hist_models=np.stack(
                    [np.asarray(entry["hist"]) for entry in model_entries], axis=0
                ),
            )
        else:
            np.savez(
                out_npz,
                var_x=var_x,
                var_y=var_y,
                level_hpa=np.nan if level_hpa is None else level_hpa,
                model_labels=np.array(valid_labels, dtype=object),
                hist_models=np.array(
                    [np.asarray(entry["hist"]) for entry in model_entries], dtype=object
                ),
                hist_targets=np.array(
                    [np.asarray(entry["hist_target"]) for entry in model_entries], dtype=object
                ),
                bins_x_list=np.array(
                    [np.asarray(entry["bins_x"]) for entry in model_entries], dtype=object
                ),
                bins_y_list=np.array(
                    [np.asarray(entry["bins_y"]) for entry in model_entries], dtype=object
                ),
                bins_x_ref=ref_bins_x,
                bins_y_ref=ref_bins_y,
                hist_target_ref=ref_hist_target,
            )

        c.success(f"[multivariate] Saved compare plot: {out_png}")
        c.success(f"[multivariate] Saved combined artifact: {out_npz}")
