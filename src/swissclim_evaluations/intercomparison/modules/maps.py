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
    """Return base key without extension.

    New schema already omits placeholder tokens; we simply strip extension.
    """
    return name[:-4] if name.endswith(".npz") else name


def intercompare_maps(
    models: list[Path],
    labels: list[str],
    out_root: Path,
    max_panels: int = 4,
) -> None:
    src_rel = Path("maps")
    dst = ensure_dir(out_root / "maps")
    # Availability report
    per_model, _, uni = scan_model_sets(models, "maps/map_*.npz")
    report_missing("maps", models, labels, per_model, uni)

    results = {}
    maps = common_files(models, "maps/map_*.npz")

    # Maps are 1-to-1, but limited by max_panels
    processed_count = min(len(maps), max_panels)
    ignored_count = max(0, len(maps) - max_panels)

    results["Maps"] = processed_count
    if ignored_count > 0:
        results["Maps (Ignored)"] = ignored_count

    report_checklist("maps", results)

    # New schema: map_<var>[ _<level>][ _init...][ _lead...]_ens*.npz
    common = common_files(models, str(src_rel / "map_*.npz"))
    if not common:
        c.warn("No common map files found. Skipping plots.")
        return
    print_file_list(f"Found {len(common)} common map files", common)
    # Limit to first N common map artifacts to avoid huge outputs
    for base in common[:max_panels]:
        key = _parse_map_filename(base)
        payloads = [load_npz(m / src_rel / f"{key}.npz") for m in models]
        # Extract DS from first payload
        target = payloads[0].get("target")

        predictions = []
        for p in payloads:
            val = p.get("prediction")
            predictions.append(val)

        if any(x is None for x in predictions) or target is None:
            continue
        lats = payloads[0].get("latitude")
        lons = payloads[0].get("longitude")
        var_name = payloads[0].get("variable")
        if not var_name and key.startswith("map_"):
            # Fallback: try to extract variable from filename key
            # key format: map_<var>_init... or map_<var>_ens...
            # Remove 'map_' prefix
            rest = key[4:]
            # Split by known tokens
            for token in ("_init", "_lead", "_ens", "_level"):
                if token in rest:
                    rest = rest.split(token, 1)[0]
                    break
            var_name = rest

        units = payloads[0].get("units")
        if lats is None or lons is None:
            continue

        def _is_3d(arr: np.ndarray) -> bool:
            return isinstance(arr, np.ndarray) and arr.ndim == 3

        # Check for lead_time dimension
        lead_times: np.ndarray = payloads[0].get("lead_time")
        is_lead_time_dim = False
        if (
            lead_times is not None
            and lead_times.size > 1
            and _is_3d(target)
            and target.shape[0] == lead_times.size
        ):
            is_lead_time_dim = True

        if is_lead_time_dim:
            # Plot all lead times in one figure (rows)
            n_leads = lead_times.size
            ncols = 1 + len(models)
            nrows = n_leads

            # Compute global vmin/vmax
            try:
                all_data = [target] + [p for p in predictions if p is not None]
                vmin = float(np.nanmin([np.nanmin(x) for x in all_data]))
                vmax = float(np.nanmax([np.nanmax(x) for x in all_data]))
            except ValueError:
                c.warn(f"maps: all-NaN data for {key}; skipping")
                continue

            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(6 * ncols, 4 * nrows),
                dpi=160,
                subplot_kw={"projection": ccrs.PlateCarree()},
                constrained_layout=True,
            )
            # Ensure axes is 2D array (nrows, ncols)
            if nrows == 1 and ncols == 1:
                axes = np.array([[axes]])
            elif nrows == 1:
                axes = axes.reshape(1, ncols)
            elif ncols == 1:
                axes = axes.reshape(nrows, 1)

            im0 = None
            date_suffix = extract_date_from_filename(key)

            for i in range(n_leads):
                target_slice = target[i]
                pred_slices = [m[i] for m in predictions if m is not None]

                # Format lead time
                lt = lead_times[i]
                if isinstance(lt, np.timedelta64):
                    lt_h = lt.astype("timedelta64[h]").astype(int)
                    lead_str = f"+{lt_h}h"
                else:
                    lead_str = f"+{int(lt)}h" if isinstance(lt, (int | float)) else str(lt)

                # Plot Target
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

                if i == 0:
                    title = f"{format_variable_name(str(var_name))} — Target{date_suffix}"
                else:
                    title = f"Target ({lead_str})"
                ax_tgt.set_title(title)

                # Plot Models
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

        else:
            n_levels = target.shape[0] if _is_3d(target) else 1
            if any(
                (_is_3d(target) and (not isinstance(m, np.ndarray) or m.ndim != 3))
                or ((not _is_3d(target)) and (not isinstance(m, np.ndarray) or m.ndim != 2))
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
                ncols = 1 + len(models)
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
                # Extract date info from filename if possible
                date_suffix = extract_date_from_filename(key)

                title_base = "Target"
                if var_name:
                    title_base = (
                        f"{format_variable_name(str(var_name))} — {title_base}{date_suffix}"
                    )
                if n_levels > 1:
                    if isinstance(level_vals, np.ndarray) and len(level_vals) == n_levels:
                        level_token = format_level_token(level_vals[lvl])
                        title_base += f" (level {level_token})"
                    else:
                        title_token = format_level_token(lvl)
                        title_base += f" (level {title_token})"
                axes[0].set_title(title_base)
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
                    # Prediction columns should not have lead time in title
                    ax.set_title(lab if n_levels == 1 else f"{lab}")
                # Use a constrained-layout-friendly colorbar spanning all axes
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
                # No tight_layout here; constrained_layout handles spacing
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
