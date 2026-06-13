from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from swissclim_evaluations import console as c
from swissclim_evaluations.helpers import format_variable_name, get_variable_units
from swissclim_evaluations.intercomparison.core import (
    common_files,
    ensure_dir,
    print_file_list,
    report_checklist,
    report_missing,
    scan_model_sets,
)
from swissclim_evaluations.plots.bivariate_histograms import (
    compute_font_scale,
    plot_bivariate_histogram,
)


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


def _extract_units(payload: dict[str, np.ndarray], axis: str) -> str | None:
    if axis not in {"x", "y"}:
        return None
    for key in (
        f"units_{axis}",
        f"unit_{axis}",
        f"{axis}_units",
        f"{axis}_unit",
        f"var_{axis}_units",
        f"var_{axis}_unit",
    ):
        units = _scalar_str(payload.get(key))
        if units:
            return units
    return None


def _format_axis_label(var_name: str, units: str | None = None) -> str:
    label = format_variable_name(var_name)
    resolved_units = units or get_variable_units(None, var_name, latex=True)
    return f"{label} [{resolved_units}]" if resolved_units else label


def _infer_var_pair(fname: str, payload: dict[str, np.ndarray]) -> tuple[str, str]:
    var_x = _scalar_str(payload.get("var_x"))
    var_y = _scalar_str(payload.get("var_y"))
    if var_x and var_y:
        return var_x, var_y

    stem = fname.replace("bivariate_", "").replace(".npz", "")
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


def _align_entries_to_common_grid(model_entries: list[dict]) -> tuple | None:
    """Zero-pad every model's histogram onto a common bin grid.

    Each model's edges share the same target-derived bin width and lattice but
    may extend different distances outward to show their own prediction tail
    (see ``plots.bivariate_histograms._extended_edges``). They are therefore all
    sub-grids of one lattice, so each ``hist``/``hist_target`` is zero-padded into
    a common grid spanning the union extent. ``model_entries`` is mutated in place
    (hist, hist_target, bins_x, bins_y replaced) and the common ``(bins_x,
    bins_y)`` is returned. Returns ``None`` if the grids are not lattice-compatible
    (different bin widths or off-lattice offsets), so the caller can fall back.
    """

    def _axis(axis: str):
        edges = [np.asarray(e[f"bins_{axis}"], dtype=float) for e in model_entries]
        if any(ed.size < 2 for ed in edges):
            return None
        dx = float(edges[0][1] - edges[0][0])
        tol = 1e-9 * max(abs(dx), 1.0)
        if dx <= 0 or any(abs(float(ed[1] - ed[0]) - dx) > tol for ed in edges):
            return None
        lo = min(float(ed[0]) for ed in edges)
        hi = max(float(ed[-1]) for ed in edges)
        n = int(round((hi - lo) / dx))
        offsets = []
        for ed in edges:
            off = (float(ed[0]) - lo) / dx
            if abs(off - round(off)) > 1e-6:
                return None
            offsets.append(int(round(off)))
        return lo + dx * np.arange(n + 1), offsets

    ax_x = _axis("x")
    ax_y = _axis("y")
    if ax_x is None or ax_y is None:
        return None
    common_x, offs_x = ax_x
    common_y, offs_y = ax_y
    nx, ny = common_x.size - 1, common_y.size - 1
    for entry, ox, oy in zip(model_entries, offs_x, offs_y, strict=False):
        for key in ("hist", "hist_target"):
            h = np.asarray(entry[key])
            padded = np.zeros((nx, ny), dtype=h.dtype)
            padded[ox : ox + h.shape[0], oy : oy + h.shape[1]] = h
            entry[key] = padded
        entry["bins_x"] = common_x
        entry["bins_y"] = common_y
    return common_x, common_y


def intercompare_multivariate(models: list[Path], labels: list[str], out_root: Path) -> None:
    """Compare multivariate bivariate-histogram artifacts across models."""
    pattern = "multivariate/bivariate_*.npz"

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
        if not var_x or not var_y:
            c.warn(
                f"[multivariate] Could not infer var_x/var_y from {fname}; "
                "axis labels will be empty. Add a recognised pair to known_pairs or "
                "ensure the NPZ was saved with var_x/var_y keys."
            )
        level_hpa = _scalar_float(first_payload.get("level_hpa"))
        coriolis_parameter = _scalar_float(first_payload.get("coriolis_parameter")) or 1.0e-4
        unit_x = _extract_units(first_payload, "x")
        unit_y = _extract_units(first_payload, "y")
        xlabel = _format_axis_label(var_x, unit_x) if var_x else ""
        ylabel = _format_axis_label(var_y, unit_y) if var_y else ""

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

        # Baseline guard: panel 0 and the shared truth contour come from
        # model_entries[0], which must be the first configured model. If that
        # model's NPZ was missing it would have been skipped, silently promoting
        # another model into the baseline slot.
        if labels and str(model_entries[0]["label"]) != str(labels[0]):
            c.warn(
                f"[multivariate] Baseline '{labels[0]}' is missing for {fname}; "
                f"panel 0 and the shared target now come from "
                f"'{model_entries[0]['label']}'."
            )

        # Models extend their grids by different amounts to show their own
        # prediction tail; pad them all onto a common lattice-aligned grid so
        # panels align and one shared truth histogram is valid everywhere.
        common = _align_entries_to_common_grid(model_entries)
        if common is None:
            c.warn(
                f"[multivariate] Histogram grids for {fname} are not lattice-"
                "compatible (different bin widths); panels may be misaligned. "
                "Regenerate the affected models."
            )
            ref_bins_x = np.asarray(model_entries[0]["bins_x"])
            ref_bins_y = np.asarray(model_entries[0]["bins_y"])
        else:
            ref_bins_x, ref_bins_y = common
        ref_hist_target = np.asarray(model_entries[0]["hist_target"])

        # Common axis limits from the shared grid (25% zoom-out), so panels are
        # directly comparable.
        all_x_min, all_x_max = float(ref_bins_x.min()), float(ref_bins_x.max())
        all_y_min, all_y_max = float(ref_bins_y.min()), float(ref_bins_y.max())
        x_range = all_x_max - all_x_min
        y_range = all_y_max - all_y_min
        x_center = (all_x_max + all_x_min) / 2.0
        y_center = (all_y_max + all_y_min) / 2.0
        shared_xlim = (x_center - 0.625 * x_range, x_center + 0.625 * x_range)
        shared_ylim = (y_center - 0.625 * y_range, y_center + 0.625 * y_range)

        n_models = len(model_entries)
        max_cols = 3
        n_cols = min(max_cols, n_models)
        n_rows = int(np.ceil(n_models / n_cols))
        n_panels = n_cols * n_rows
        fig_w, fig_h = 6 * n_cols, 7 * n_rows
        font_scale = compute_font_scale(fig_w, fig_h)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fig_w, fig_h),
            dpi=150,
            constrained_layout=True,
            squeeze=False,
        )
        # Reserve space at the top so the suptitle has breathing room below it.
        fig.get_layout_engine().set(rect=(0, 0, 1, 0.97))
        axs_flat = axes.flatten()

        # Use ONE shared truth histogram (from the first model) for every
        # panel. Otherwise each panel renders the truth contours from its own
        # ``hist_target``, which can differ in mass and shape if models were
        # evaluated against slightly different time-matching, masking, or
        # member subsetting. Using the first model's target guarantees the
        # truth contour lines are byte-identical across panels.
        ref_target = np.asarray(model_entries[0]["hist_target"])
        shared_fill_cs = None
        shared_target_cs = None

        for idx, entry in enumerate(model_entries):
            ax = axs_flat[idx]
            hist = np.asarray(entry["hist"])
            bins_x = np.asarray(entry["bins_x"])
            bins_y = np.asarray(entry["bins_y"])
            label = str(entry["label"])
            col = idx % n_cols
            last_row_start = (n_rows - 1) * n_cols
            is_bottom = idx >= last_row_start
            _result = plot_bivariate_histogram(
                hist_1=hist,
                hist_2=ref_target,
                bins_x=bins_x,
                bins_y=bins_y,
                label_1="Prediction",
                label_2="Target",
                var_x=var_x,
                var_y=var_y,
                level_hpa=level_hpa,
                ax=ax,
                xlabel=xlabel if (xlabel and is_bottom) else None,
                ylabel=ylabel if ylabel else None,
                return_contour_sets=True,
                xlim=shared_xlim,
                ylim=shared_ylim,
                show_colorbar=False,
                show_legend=(idx == 0),
                coriolis_parameter=coriolis_parameter,
                font_scale=font_scale,
            )
            ax.set_title(label, fontsize=int(round(12 * font_scale)))
            if not is_bottom:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)
            if col != 0:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)
            # Capture the truth contour set once, so the shared colorbar can
            # add greyscale level marks via ``cbar.add_lines`` below. This
            # restores the contour-line annotations that the per-panel
            # colorbars carry in single-eval mode (see plots/bivariate
            # _histograms.py: ``cbar.add_lines(cs1)``). Without it the shared
            # intercomp colorbar is a bare gradient with no level cues.
            if shared_target_cs is None and isinstance(_result, tuple):
                shared_fill_cs = _result[1]
                shared_target_cs = _result[2]

        # Hide surplus axes in the last row.
        for idx in range(n_models, n_panels):
            axs_flat[idx].set_visible(False)

        # ── Shared horizontal colorbar at the bottom ──────────────────────────
        # Build the colorbar from the panels' filled-contour artist, exactly
        # like single-eval mode. Every panel shares the same target-derived
        # density levels/norm, so this artist is a correct shared reference and
        # its density scale keeps the grey target isolines added via
        # ``add_lines`` aligned with the colorbar ticks. A hand-rolled
        # ScalarMappable on raw histogram counts would put the colorbar on a
        # different (count) scale and shift the markers off the isolines.
        if shared_fill_cs is not None:
            cbar = fig.colorbar(
                shared_fill_cs,
                ax=axs_flat[:n_models].tolist(),
                orientation="horizontal",
                location="bottom",
                pad=0.04,
                fraction=0.08,
                shrink=1.0,
                format="%.2e",
            )
            cbar.set_label("Density (log scale)", fontsize=int(round(11 * font_scale)))
            cbar.ax.tick_params(labelsize=int(round(9 * font_scale)))
            if shared_target_cs is not None:
                cbar.add_lines(shared_target_cs)

        if var_x and var_y:
            title = f"{format_variable_name(var_x)} vs {format_variable_name(var_y)}"
            if level_hpa is not None:
                title += f" ({level_hpa:g} hPa)"
            fig.suptitle(title, fontsize=int(round(14 * font_scale)))
        else:
            fig.suptitle(
                fname.replace("bivariate_", "").replace(".npz", ""),
                fontsize=int(round(14 * font_scale)),
            )

        stem = fname.replace("bivariate_", "").replace(".npz", "")
        out_png = dst / f"bivariate_{stem}_compare.png"
        out_pdf = dst / f"bivariate_{stem}_compare.pdf"
        fig.savefig(out_png, dpi=150)
        fig.savefig(out_pdf)
        plt.close(fig)

        out_npz = dst / f"bivariate_{stem}_compare.npz"
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
        c.success(f"[multivariate] Saved compare plot: {out_pdf}")
        c.success(f"[multivariate] Saved combined artifact: {out_npz}")
