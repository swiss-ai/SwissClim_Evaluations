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

        hist_target = np.asarray(first_payload["hist_target"])
        bins_x = np.asarray(first_payload["bins_x"])
        bins_y = np.asarray(first_payload["bins_y"])

        model_hists: list[np.ndarray] = []
        valid_labels: list[str] = []

        for label, model_dir in zip(labels, models, strict=False):
            payload = _load_hist_payload(model_dir / "multivariate" / fname)
            if "hist" not in payload:
                c.warn(f"[multivariate] Missing 'hist' in {model_dir / 'multivariate' / fname}")
                continue

            hist = np.asarray(payload["hist"])
            if hist.shape != hist_target.shape:
                c.warn(
                    f"[multivariate] Shape mismatch for {fname} in {label}: "
                    f"model={hist.shape}, target={hist_target.shape}; skipped"
                )
                continue

            model_hists.append(hist)
            valid_labels.append(label)

        if not model_hists:
            c.warn(f"[multivariate] No valid model histograms for {fname}; skipped")
            continue

        n_cols = len(model_hists)
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6), squeeze=False)

        for idx, (hist, label) in enumerate(zip(model_hists, valid_labels, strict=False)):
            ax = axes[0, idx]
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
        np.savez(
            out_npz,
            bins_x=bins_x,
            bins_y=bins_y,
            hist_target=hist_target,
            var_x=var_x,
            var_y=var_y,
            level_hpa=np.nan if level_hpa is None else level_hpa,
            model_labels=np.array(valid_labels, dtype=object),
            hist_models=np.stack(model_hists, axis=0),
        )

        c.success(f"[multivariate] Saved compare plot: {out_png}")
        c.success(f"[multivariate] Saved combined artifact: {out_npz}")
