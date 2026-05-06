from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..helpers import format_level_token, format_variable_name, save_figure

_NORM_LABELS: dict[str, str] = {
    "global_std": "/ σ(target)",
    "one_step_std": "/ σ(Δ target)",
    "per_variable_max": "/ max (per variable)",
}


def plot_error_heatmap(
    long_df: pd.DataFrame,
    metric: str,
    out_path: Path,
    title: str = "",
    normalization: str = "global_std",
    var_stats: dict[str, float] | None = None,
    dpi: int = 96,
) -> None:
    """Plot a variables x lead-time heatmap of normalized errors.

    Rows are variables (with level suffix when a ``level`` column is present),
    columns are lead times in hours.

    The colorscale is driven by ``normalization``:

    - ``"global_std"`` (default): each cell = metric / σ(target).  Requires
      ``var_stats`` mapping variable name → target global std; falls back to
      ``per_variable_max`` for variables missing from ``var_stats``.
    - ``"one_step_std"``: each cell = metric / σ(Δ target, one lead step).
      Same fallback as above.
    - ``"per_variable_max"``: each row normalized to [0, 1] by its own
      maximum absolute value.

    Cell annotations show original metric values (in physical units) when the
    grid has at most 400 cells.
    """
    if metric not in long_df.columns:
        return

    df = long_df.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")

    # Build row labels: "VAR" or "VAR @ LEVEL"
    if "level" in df.columns and df["level"].notna().any():

        def _row_label(r: pd.Series) -> str:
            name = format_variable_name(str(r["variable"]))
            if pd.notna(r["level"]):
                return f"{name} @ {format_level_token(r['level'])}"
            return name

        df["_row"] = df.apply(_row_label, axis=1)
    else:
        df["_row"] = df["variable"].apply(lambda v: format_variable_name(str(v)))

    # Pivot to variables x lead_times
    matrix = df.pivot_table(index="_row", columns="lead_time_hours", values=metric, aggfunc="mean")
    matrix = matrix.sort_index(axis=1)

    if matrix.empty:
        return

    # Map each display row back to its raw variable name for stat lookup
    row_to_var: dict[str, str] = df.groupby("_row")["variable"].first().to_dict()

    # Build normalization denominator (Series indexed by _row label)
    if normalization in ("global_std", "one_step_std") and var_stats:
        denom = pd.Series(
            {row: var_stats.get(str(var), np.nan) for row, var in row_to_var.items()},
            dtype=float,
        )
        # Fall back to per-row max for rows whose variable is not in var_stats
        missing_mask = denom.isna() | (denom == 0)
        if missing_mask.any():
            row_max_fallback = matrix.abs().max(axis=1).replace(0, np.nan)
            denom = denom.where(~missing_mask, row_max_fallback)
        norm_matrix = matrix.div(denom, axis=0).fillna(0.0)
        vmax_val = None  # let data drive the upper bound (shows actual error / std ratio)
        cb_label = f"{metric} {_NORM_LABELS[normalization]}"
    else:
        # per_variable_max: clip colorscale to [0, 1]
        row_max = matrix.abs().max(axis=1).replace(0, np.nan)
        norm_matrix = matrix.div(row_max, axis=0).fillna(0.0)
        vmax_val = 1.0
        cb_label = f"{metric} {_NORM_LABELS['per_variable_max']}"

    n_vars, n_times = matrix.shape

    # Adaptive figure sizing
    cell_w = max(0.5, min(1.4, 12.0 / max(n_times, 1)))
    cell_h = max(0.35, min(0.8, 8.0 / max(n_vars, 1)))
    fig_w = max(8.0, n_times * cell_w + 3.0)
    fig_h = max(3.0, n_vars * cell_h + 2.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi * 2)

    im = ax.imshow(
        norm_matrix.values,
        cmap="Reds",
        vmin=0.0,
        vmax=vmax_val,
        aspect="auto",
        interpolation="none",
    )

    ax.set_xticks(np.arange(n_times))
    ax.set_yticks(np.arange(n_vars))

    x_rotation = 45 if n_times > 12 else 0
    ax.set_xticklabels(
        [str(int(h)) for h in matrix.columns],
        rotation=x_rotation,
        ha="right" if x_rotation else "center",
    )
    ax.set_yticklabels(matrix.index.tolist())
    ax.set_xlabel("Lead Time [h]")

    # Cell annotations (original values) -- suppress when grid is too dense
    font_size = max(5, min(9, int(120 / max(n_vars * n_times, 1) ** 0.5)))
    if n_vars * n_times <= 400:
        _finite = norm_matrix.values[np.isfinite(norm_matrix.values)]
        norm_max = float(_finite.max()) if _finite.size else 1.0
        for i in range(n_vars):
            for j in range(n_times):
                val = matrix.iloc[i, j]
                if pd.notna(val):
                    nv = norm_matrix.iloc[i, j]
                    text_color = "white" if (norm_max > 0 and nv / norm_max > 0.6) else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.3g}",
                        ha="center",
                        va="center",
                        fontsize=font_size,
                        color=text_color,
                    )

    plt.colorbar(im, ax=ax, label=cb_label, fraction=0.03, pad=0.02)
    ax.set_title(title or f"{metric} \u2014 Variables \u00d7 Lead Time", loc="left", fontsize=10)

    plt.tight_layout()
    save_figure(fig, out_path, module="deterministic")
    plt.close(fig)
