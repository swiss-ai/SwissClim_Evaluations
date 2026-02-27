from __future__ import annotations

import contextlib
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from swissclim_evaluations.helpers import (
    extract_date_from_filename,
    format_level_token,
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
    scan_model_sets,
)


def _parse_map_filename(name: str) -> str:
    """Return base key without extension."""
    return name[:-4] if name.endswith(".npz") else name


def _is_3d(arr: np.ndarray) -> bool:
    return isinstance(arr, np.ndarray) and arr.ndim == 3


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
) -> None:
    """Fill one row of map panels: Target + each model, plus a shared colorbar."""
    im0 = axes[0].pcolormesh(
        lons,
        lats,
        target_slice,
        cmap="viridis",
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
            cmap="viridis",
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
                im0 = ax_tgt.pcolormesh(
                    lons,
                    lats,
                    target_slice,
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                    transform=ccrs.PlateCarree(),
                )
                ax_tgt.coastlines(linewidth=0.5)
                ax_tgt.set_title(
                    f"{format_variable_name(str(var_name))} — Target{date_suffix}"
                    if i == 0
                    else f"Target ({lead_str})"
                )
                for j, (lab, pred_slice) in enumerate(zip(labels, pred_slices, strict=False)):
                    ax = axes[i, j + 1]
                    ax.pcolormesh(
                        lons,
                        lats,
                        pred_slice,
                        cmap="viridis",
                        vmin=vmin,
                        vmax=vmax,
                        transform=ccrs.PlateCarree(),
                    )
                    ax.coastlines(linewidth=0.5)
                    if i == 0:
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
