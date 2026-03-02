from __future__ import annotations

import contextlib
import re as _re
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from swissclim_evaluations.helpers import (
    SPATIAL_METRIC_SPECS,
    extract_date_from_filename,
    format_level_token,
    format_variable_name,
    get_colormap_for_variable,
)
from swissclim_evaluations.intercomparison.core import (
    c,
    common_files,
    ensure_dir,
    load_npz,
    print_file_list,
    report_checklist,
    report_missing,
    scan_model_sets,
)


def _parse_map_filename(name: str) -> str:
    """Return base key without extension."""
    return name[:-4] if name.endswith(".npz") else name


def _is_3d(arr: np.ndarray) -> bool:
    return isinstance(arr, np.ndarray) and arr.ndim == 3


def _filter_combined_level_files(file_list: list[str]) -> list[str]:
    """Remove combined 'X_to_Y' level-range NPZ files when per-level files exist.

    When ``plots/maps.py`` saves multi-lead 3D variables it produces one NPZ
    per pressure level (e.g. ``map_temperature_500_…npz``).  Older runs may
    have also written a combined NPZ like ``map_temperature_500_to_850_…npz``.
    If both the combined *and* at least one per-level file are present we
    drop the combined entry to avoid creating redundant intercomparison
    plots that lack lead-time information.
    """
    names = set(file_list)
    to_remove: set[str] = set()
    for f in file_list:
        m = _re.search(r"_(\d+)_to_(\d+)_", f)
        if not m:
            continue
        lo, hi = m.group(1), m.group(2)
        prefix = f[: m.start()]
        suffix = f[m.end() - 1 :]  # keep the trailing '_'
        lo_file = f"{prefix}_{lo}{suffix}"
        hi_file = f"{prefix}_{hi}{suffix}"
        if lo_file in names or hi_file in names:
            to_remove.add(f)
    if to_remove:
        c.print(
            f"[maps] Skipping {len(to_remove)} combined-level NPZ file(s) "
            "in favour of per-level equivalents"
        )
    return [f for f in file_list if f not in to_remove]


def _plot_one_row(
    fig: plt.Figure,
    axes: list,
    lons: np.ndarray,
    lats: np.ndarray,
    target_slice: np.ndarray,
    pred_slices: list[np.ndarray],
    labels: list[str],
    vmin: float,
    vmax: float,
    units: str | None,
    title_target: str,
    var_name: str | None = None,
) -> None:
    """Fill one row of map panels: Target + each model, plus a shared colorbar."""
    cmap = get_colormap_for_variable(str(var_name)) if var_name else "viridis"
    im0 = axes[0].pcolormesh(
        lons,
        lats,
        target_slice,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
    )
    axes[0].coastlines(linewidth=0.5)
    axes[0].set_title(title_target)
    for ax, lab, pred_slice in zip(axes[1:], labels, pred_slices, strict=False):
        ax.pcolormesh(
            lons,
            lats,
            pred_slice,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
        )
        ax.coastlines(linewidth=0.5)
        ax.set_title(lab)
    cbar = fig.colorbar(
        im0,
        ax=axes if isinstance(axes, (list | np.ndarray)) else [axes],
        orientation="horizontal",
        fraction=0.05,
        pad=0.08,
        aspect=35,
    )
    with contextlib.suppress(Exception):
        cbar.set_label(str(units) if units else "Value")


def intercompare_maps(
    models: list[Path],
    labels: list[str],
    out_root: Path,
    max_panels: int = 4,
) -> None:
    src_rel = Path("maps")
    dst = ensure_dir(out_root / "maps")

    per_model, _, uni = scan_model_sets(models, "maps/map_*.npz")
    report_missing("maps", models, labels, per_model, uni)

    all_maps = common_files(models, "maps/map_*.npz")
    results: dict = {"Maps": min(len(all_maps), max_panels)}
    ignored = max(0, len(all_maps) - max_panels)
    if ignored:
        results["Maps (Ignored)"] = ignored
    report_checklist("maps", results)

    common = common_files(models, str(src_rel / "map_*.npz"))
    if not common:
        c.warn("No common map files found. Skipping plots.")
        return
    # Drop combined level-range NPZs when per-level equivalents exist
    common = _filter_combined_level_files(common)
    print_file_list(f"Found {len(common)} common map files", common)

    for base in common[:max_panels]:
        key = _parse_map_filename(base)
        payloads = [load_npz(m / src_rel / f"{key}.npz") for m in models]

        target = payloads[0].get("target")
        predictions = [p.get("prediction") for p in payloads]

        if target is None or any(x is None for x in predictions):
            continue

        lats = payloads[0].get("latitude")
        lons = payloads[0].get("longitude")
        if lats is None or lons is None:
            continue

        units = payloads[0].get("units")
        date_suffix = extract_date_from_filename(key)
        ncols = 1 + len(models)

        var_name = payloads[0].get("variable")
        if not var_name and key.startswith("map_"):
            rest = key[4:]
            for token in ("_init", "_lead", "_ens", "_level"):
                if token in rest:
                    rest = rest.split(token, 1)[0]
                    break
            var_name = rest

        # ── Case 1: lead_time is the leading dimension (2D var stacked by lead) ──
        lead_times: np.ndarray = payloads[0].get("lead_time")
        is_lead_time_dim = (
            lead_times is not None
            and lead_times.size > 1
            and _is_3d(target)
            and target.shape[0] == lead_times.size
        )

        if is_lead_time_dim:
            n_leads = lead_times.size
            try:
                all_data = [target] + [p for p in predictions if p is not None]
                vmin = float(np.nanmin([np.nanmin(x) for x in all_data]))
                vmax = float(np.nanmax([np.nanmax(x) for x in all_data]))
            except ValueError:
                c.warn(f"maps: all-NaN data for {key}; skipping")
                continue

            fig, axes = plt.subplots(
                n_leads,
                ncols,
                figsize=(6 * ncols, 4 * n_leads),
                dpi=160,
                subplot_kw={"projection": ccrs.PlateCarree()},
                constrained_layout=True,
            )
            if n_leads == 1:
                axes = axes.reshape(1, ncols)
            elif ncols == 1:
                axes = axes.reshape(n_leads, 1)

            im0 = None
            for i in range(n_leads):
                lt = lead_times[i]
                if isinstance(lt, np.timedelta64):
                    lead_str = f"+{lt.astype('timedelta64[h]').astype(int)}h"
                else:
                    lead_str = f"+{int(lt)}h" if isinstance(lt, (int | float)) else str(lt)

                target_slice = target[i]
                pred_slices = [m[i] for m in predictions if m is not None]

                ax_tgt = axes[i, 0]
                _cmap = get_colormap_for_variable(str(var_name)) if var_name else "viridis"
                im0 = ax_tgt.pcolormesh(
                    lons,
                    lats,
                    target_slice,
                    cmap=_cmap,
                    vmin=vmin,
                    vmax=vmax,
                    transform=ccrs.PlateCarree(),
                )
                ax_tgt.coastlines(linewidth=0.5)
                if i == 0:
                    ax_tgt.set_title(f"{format_variable_name(str(var_name))} — Target{date_suffix}")
                else:
                    ax_tgt.set_title("")
                # Row label on left side
                ax_tgt.text(
                    -0.08,
                    0.5,
                    f"({lead_str})",
                    transform=ax_tgt.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=9,
                    fontweight="bold",
                )
                for j, (lab, pred_slice) in enumerate(zip(labels, pred_slices, strict=False)):
                    ax = axes[i, j + 1]
                    ax.pcolormesh(
                        lons,
                        lats,
                        pred_slice,
                        cmap=_cmap,
                        vmin=vmin,
                        vmax=vmax,
                        transform=ccrs.PlateCarree(),
                    )
                    ax.coastlines(linewidth=0.5)
                    if i == 0:
                        ax.set_title(lab)
                    else:
                        ax.set_title("")

            if im0 is not None:
                cbar = fig.colorbar(
                    im0,
                    ax=axes,
                    orientation="horizontal",
                    fraction=0.05,
                    pad=0.05,
                    aspect=35,
                )
                with contextlib.suppress(Exception):
                    cbar.set_label(str(units) if units else "Value")

            out_png = dst / (key + "_compare.png")
            plt.savefig(out_png, bbox_inches="tight", dpi=200)
            c.success(f"Saved {out_png.relative_to(out_root)}")
            plt.close(fig)

        # ── Case 2: 3D or 2D — one figure per level, same layout as 2D ─────────
        else:
            n_levels = target.shape[0] if _is_3d(target) else 1
            if any(
                (_is_3d(target) and (not isinstance(m, np.ndarray) or m.ndim != 3))
                or (not _is_3d(target) and (not isinstance(m, np.ndarray) or m.ndim != 2))
                for m in predictions
            ):
                c.warn(f"maps: shape mismatch for {key}; skipping")
                continue

            level_vals = payloads[0].get("level")

            for lvl in range(n_levels):
                target_slice = target[lvl] if n_levels > 1 else target
                pred_slices = [
                    m[lvl] if n_levels > 1 else m for m in predictions if isinstance(m, np.ndarray)
                ]
                try:
                    vmin = float(
                        np.nanmin([np.nanmin(target_slice)] + [np.nanmin(x) for x in pred_slices])
                    )
                    vmax = float(
                        np.nanmax([np.nanmax(target_slice)] + [np.nanmax(x) for x in pred_slices])
                    )
                except ValueError:
                    c.warn(f"maps: all-NaN data for {key} level {lvl}; skipping")
                    continue

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

                title_target = (
                    f"{format_variable_name(str(var_name))} — Target{date_suffix}"
                    if var_name
                    else f"Target{date_suffix}"
                )
                if n_levels > 1:
                    if isinstance(level_vals, np.ndarray) and len(level_vals) == n_levels:
                        title_target += f" (level {format_level_token(level_vals[lvl])})"
                    else:
                        title_target += f" (level {format_level_token(lvl)})"

                _plot_one_row(
                    fig,
                    axes,
                    lons,
                    lats,
                    target_slice,
                    pred_slices,
                    labels,
                    vmin,
                    vmax,
                    units,
                    title_target,
                    var_name=var_name,
                )

                suffix = ""
                if n_levels > 1:
                    if isinstance(level_vals, np.ndarray) and len(level_vals) == n_levels:
                        suffix = f"_level{format_level_token(level_vals[lvl])}"
                    else:
                        suffix = f"_level{format_level_token(lvl)}"

                out_png = dst / (key + suffix + "_compare.png")
                plt.savefig(out_png, bbox_inches="tight", dpi=200)
                c.success(f"Saved {out_png.relative_to(out_root)}")
                plt.close(fig)

    # ── Spatial metric maps intercomparison (MAE / RMSE / Bias) ────────────
    det_rel = Path("deterministic")
    for _metric_name, _sspec in SPATIAL_METRIC_SPECS.items():
        metric_key = _sspec["key"]
        spec = _sspec
        glob_pat = f"deterministic/det_{metric_key}_map_*.npz"
        metric_common = common_files(models, glob_pat)
        # Fallback: also check legacy maps/ folder for mae
        if not metric_common and metric_key == "mae":
            metric_common = common_files(models, "maps/mae_map_*.npz")
            if metric_common:
                det_rel = Path("maps")
        if not metric_common:
            continue
        display_name = metric_key.upper()
        c.print(f"[maps] Found {len(metric_common)} common" f" {display_name} map files")
        for base in metric_common[:max_panels]:
            key = _parse_map_filename(base)
            payloads = [load_npz(m / det_rel / f"{key}.npz") for m in models]

            metric_arrays = [p.get(metric_key) for p in payloads]
            if any(x is None for x in metric_arrays):
                # Fallback: compute from target/prediction if present
                metric_arrays = []
                for p in payloads:
                    t = p.get("target")
                    pr = p.get("prediction")
                    if t is not None and pr is not None:
                        metric_arrays.append(
                            spec.get(
                                "fn",
                                lambda pr, t: np.abs(pr - t),
                            )(pr, t)
                        )
                    else:
                        metric_arrays.append(None)

            if any(x is None for x in metric_arrays):
                continue

            lats = payloads[0].get("latitude")
            lons = payloads[0].get("longitude")
            if lats is None or lons is None:
                continue

            var_name = payloads[0].get("variable")
            units = payloads[0].get("units")

            try:
                if spec["diverging"]:
                    abs_max = float(
                        np.nanmax([np.nanmax(np.abs(x)) for x in metric_arrays if x is not None])
                    )
                    vmin, vmax = -abs_max, abs_max
                elif spec["vmin_zero"]:
                    vmin = 0.0
                    vmax = float(np.nanmax([np.nanmax(x) for x in metric_arrays if x is not None]))
                else:
                    vmin = float(np.nanmin([np.nanmin(x) for x in metric_arrays if x is not None]))
                    vmax = float(np.nanmax([np.nanmax(x) for x in metric_arrays if x is not None]))
            except ValueError:
                continue

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

            im0 = None
            for _j, (ax, lab, arr) in enumerate(zip(axes, labels, metric_arrays, strict=False)):
                if arr is None:
                    continue
                plot_arr = arr.squeeze() if arr.ndim > 2 else arr
                im0 = ax.pcolormesh(
                    lons,
                    lats,
                    plot_arr,
                    cmap=spec["cmap"],
                    vmin=vmin,
                    vmax=vmax,
                    transform=ccrs.PlateCarree(),
                )
                ax.coastlines(linewidth=0.5)
                ax.set_title(lab)

            if im0 is not None:
                cbar = fig.colorbar(
                    im0,
                    ax=axes,
                    orientation="horizontal",
                    fraction=0.05,
                    pad=0.05,
                    aspect=35,
                )
                cb_label = display_name
                if units:
                    cb_label += f" [{units}]"
                with contextlib.suppress(Exception):
                    cbar.set_label(cb_label)

                title = display_name
                if var_name:
                    title += f" — {format_variable_name(str(var_name))}"
                fig.suptitle(title, fontsize=11, y=1.02)

            out_png = dst / (key + "_compare.png")
            plt.savefig(out_png, bbox_inches="tight", dpi=200)
            c.success(f"Saved {out_png.relative_to(out_root)}")
            plt.close(fig)

            # ── Per-lead gridded error map (rows=leads, cols=models) ──────
            per_lead_key = f"{metric_key}_per_lead"
            per_lead_arrays = [p.get(per_lead_key) for p in payloads]
            lead_labels_arr = payloads[0].get("lead_labels")
            if (
                all(x is not None for x in per_lead_arrays)
                and lead_labels_arr is not None
                and per_lead_arrays[0].ndim == 3
                and per_lead_arrays[0].shape[0] > 1
            ):
                n_leads = per_lead_arrays[0].shape[0]
                ncols_pl = len(models)
                try:
                    if spec["diverging"]:
                        abs_max_pl = float(
                            np.nanmax(
                                [np.nanmax(np.abs(x)) for x in per_lead_arrays if x is not None]
                            )
                        )
                        vmin_pl, vmax_pl = -abs_max_pl, abs_max_pl
                    elif spec["vmin_zero"]:
                        vmin_pl = 0.0
                        vmax_pl = float(
                            np.nanmax([np.nanmax(x) for x in per_lead_arrays if x is not None])
                        )
                    else:
                        vmin_pl = float(
                            np.nanmin([np.nanmin(x) for x in per_lead_arrays if x is not None])
                        )
                        vmax_pl = float(
                            np.nanmax([np.nanmax(x) for x in per_lead_arrays if x is not None])
                        )
                except ValueError:
                    vmin_pl, vmax_pl = None, None

                if vmin_pl is not None:
                    fig_pl, axes_pl = plt.subplots(
                        n_leads,
                        ncols_pl,
                        figsize=(5 * ncols_pl, 3.5 * n_leads),
                        dpi=160,
                        subplot_kw={"projection": ccrs.PlateCarree()},
                        constrained_layout=True,
                    )
                    axes_pl = np.atleast_2d(axes_pl)
                    im_pl = None
                    for li in range(n_leads):
                        lt_raw = str(lead_labels_arr[li]) if li < len(lead_labels_arr) else ""
                        # Normalise to "Target (+Xh)" notation
                        lt_label = (
                            f"Target ({lt_raw})"
                            if lt_raw and not lt_raw.startswith("Target")
                            else lt_raw
                        )
                        for mi, (lab, arr_pl) in enumerate(
                            zip(labels, per_lead_arrays, strict=False)
                        ):
                            ax = axes_pl[li, mi]
                            im_pl = ax.pcolormesh(
                                lons,
                                lats,
                                arr_pl[li],
                                cmap=spec["cmap"],
                                vmin=vmin_pl,
                                vmax=vmax_pl,
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
                    if im_pl is not None:
                        cbar_pl = fig_pl.colorbar(
                            im_pl,
                            ax=axes_pl.ravel().tolist(),
                            orientation="horizontal",
                            fraction=0.03,
                            pad=0.04,
                            aspect=40,
                        )
                        cb_lbl = display_name
                        if units:
                            cb_lbl += f" [{units}]"
                        with contextlib.suppress(Exception):
                            cbar_pl.set_label(cb_lbl)
                        title_pl = f"{display_name} per Lead"
                        if var_name:
                            title_pl += f" — {format_variable_name(str(var_name))}"
                        fig_pl.suptitle(title_pl, fontsize=11, y=1.02)
                    out_pl = dst / (key + "_per_lead_compare.png")
                    plt.savefig(out_pl, bbox_inches="tight", dpi=200)
                    c.success(f"Saved {out_pl.relative_to(out_root)}")
                    plt.close(fig_pl)

        # Reset det_rel for the next metric
        det_rel = Path("deterministic")
