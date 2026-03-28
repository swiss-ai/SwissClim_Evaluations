import itertools
import re
from pathlib import Path
from typing import Any

import dask.array as dsa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import Colormap, LinearSegmentedColormap

from . import console as c


def get_pit_evolution_colormap() -> Colormap:
    """
    Returns a diverging colormap for PIT evolution plots:
    - COLOR_DIAGNOSTIC (blue) for under-population (density < 1)
    - white for ideal calibration (density = 1)
    - COLOR_MODEL_PREDICTION (vermillion) for over-population (density > 1)
    """
    return LinearSegmentedColormap.from_list(
        "pit_evolution",
        [COLOR_DIAGNOSTIC, "white", COLOR_MODEL_PREDICTION],
        N=256,
    )


# Consistent colors for single-model evaluation plots
COLOR_GROUND_TRUTH = "black"
COLOR_MODEL_PREDICTION = "#D55E00"  # Vermilion (colorblind-friendly)
COLOR_DIAGNOSTIC = "#4C78A8"  # Neutral blue for diagnostic metrics (e.g. PIT histogram)


def get_colormap_for_variable(variable_name: str) -> str | Colormap:
    """
    Returns an appropriate colormap for a given physical variable.

    Args:
        variable_name (str): Name of the variable (case-insensitive).
            Can contain underscores or spaces.
            Matching is performed using substring search after converting to lowercase.

    Returns:
        str or Colormap: A matplotlib-compatible colormap name (string) or Colormap object.
            Returns "viridis" as default if no match is found.

    Matching logic:
        - Substring matching via `in` is used to determine the appropriate colormap.
        - For example, if "temperature" is in the variable name, "magma" is returned.

    Examples:
        >>> get_colormap_for_variable("temperature")
        'magma'
        >>> get_colormap_for_variable("U_Component_Of_Wind")
        'RdBu_r'
        >>> get_colormap_for_variable("precipitation")
        'Blues'
        >>> get_colormap_for_variable("unknown_variable")
        'viridis'
    """
    variable_name = variable_name.lower()

    # Note: The order of checks matters!
    # Some variables might match multiple categories (e.g. "integrated_vapor_transport"
    # contains "vapor" which is also in the precipitation list).
    # We check for diverging variables first to catch things like fluxes/transport
    # before checking for general water/moisture terms.

    # Diverging variables (wind components, vertical velocity, anomalies, fluxes)
    diverging_vars = [
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity",
        "divergence",
        "vorticity",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_surface_latent_heat_flux",
        "mean_surface_sensible_heat_flux",
        "mean_vertically_integrated_moisture_divergence",
        "integrated_vapor_transport",  # Often vector magnitude but sometimes treated as flux
    ]

    if any(v in variable_name for v in diverging_vars):
        return "RdBu_r"

    # Precipitation / Water / Moisture (sequential - Blues)
    if any(
        x in variable_name
        for x in [
            "precipitation",
            "total_column_water",
            "total_column_water_vapour",
            "total_column_vapor",
            "volumetric_soil_water",
            "snow_depth",
            "sea_ice_cover",
            "lake_cover",
        ]
    ):
        return "Blues"

    # Temperature / Heat / Radiation (sequential - Magma/Inferno/Hot)
    if any(
        x in variable_name
        for x in [
            "temperature",
            "2m_temperature",
            "2m_dewpoint_temperature",
            "sea_surface_temperature",
            "radiation_flux",  # short/long wave fluxes
        ]
    ):
        return "magma"

    # Humidity / Vegetation (sequential - Greens)
    if any(
        x in variable_name
        for x in [
            "humidity",
            "specific_humidity",
            "relative_humidity",
            "leaf_area_index",
            "vegetation_cover",
        ]
    ):
        return "Greens"

    # Pressure / Geopotential / Orography (sequential - Viridis)
    if any(
        x in variable_name
        for x in [
            "geopotential",
            "pressure",
            "mean_sea_level_pressure",
            "surface_pressure",
            "orography",
            "land_sea_mask",
            "soil_type",
            "boundary_layer_height",
            "lapse_rate",
        ]
    ):
        return "viridis"

    # Cloud cover (sequential - Greys)
    if "cloud_cover" in variable_name:
        return "Greys_r"

    # Wind speed / Energy (sequential - YlOrRd)
    if any(
        x in variable_name
        for x in [
            "wind_speed",
            "10m_wind_speed",
            "eddy_kinetic_energy",
            "potential_vorticity",
        ]
    ):
        return "YlOrRd"

    # Default
    return "viridis"


def aggregate_member_dfs(dfs):
    """Aggregate a list of per-member pandas DataFrames by arithmetic mean.

    Assumes all DataFrames share identical index and column sets. Non-numeric
    columns (object, string) are dropped before averaging. Returns a new
    DataFrame with the same index ordering and averaged numeric columns.
    If list empty returns an empty DataFrame.
    """
    if not dfs:
        import pandas as _pd  # local import to avoid heavy dependency upfront

        return _pd.DataFrame()
    import pandas as _pd

    # Intersect numeric columns across all frames
    numeric_cols = None
    for df in dfs:
        cols = [c for c, dt in df.dtypes.items() if _pd.api.types.is_numeric_dtype(dt)]
        cols_set = set(cols)
        numeric_cols = cols_set if numeric_cols is None else numeric_cols & cols_set
    if not numeric_cols:
        return dfs[0].copy()
    # Ensure identical index; if not, align by outer join then average (filling with NaN)
    base_index = dfs[0].index
    if any(not df.index.equals(base_index) for df in dfs[1:]):
        dfs = [df.reindex(base_index) for df in dfs]
    arr_stack = np.stack([df[list(numeric_cols)].to_numpy(dtype=float) for df in dfs], axis=0)
    # Guard against all-NaN columns to avoid empty-slice warnings; nanmean already ignores NaNs,
    # but if an entire column is NaN across all members, keep it as NaN without warning.
    with np.errstate(all="ignore"):
        mean_arr = np.nanmean(arr_stack, axis=0)
    out = dfs[0][[]].copy()
    for i, col in enumerate(list(numeric_cols)):
        out[col] = mean_arr[:, i]
    return out[sorted(numeric_cols)]


def format_init_time_range(ts: np.ndarray) -> tuple[str, str]:
    """Format init_time array (datetime64) to YYYY-MM-DDTHH strings (hour precision)."""
    if ts.size == 0:
        return ("", "")
    start = np.datetime64(ts.min()).astype("datetime64[h]")
    end = np.datetime64(ts.max()).astype("datetime64[h]")

    def _fmt(x):
        return np.datetime_as_string(x, unit="h").replace(":", "")

    return _fmt(start), _fmt(end)


def _fmt_lead(ts: np.ndarray) -> tuple[int, int]:
    """Return min/max lead hours as integers from timedelta64 array."""
    hrs = (ts / np.timedelta64(1, "h")).astype(int)
    return int(hrs.min()), int(hrs.max())


def time_range_suffix(ds: xr.Dataset) -> str:
    """Build suffix string encoding init_time and lead_time ranges.

    Patterns required by tests:
    - Both dims: 'init_time_<start>_to_<end>__lead_time_<h0>_to_<h1>'
    - Only one dim present → single segment without separator.
    Datetime formatted as YYYY-MM-DDTHH. Lead times in hours (ints).
    """
    segments: list[str] = []
    if "init_time" in ds.coords:
        try:
            init_vals = np.asarray(ds["init_time"].values)
            if init_vals.size:
                s, e = format_init_time_range(init_vals)
                if s and e:
                    segments.append(f"init_time_{s}_to_{e}")
        except Exception:
            pass
    if "lead_time" in ds.coords:
        lead_vals = np.asarray(ds["lead_time"].values)
        if lead_vals.size:
            h0, h1 = _fmt_lead(lead_vals)
            segments.append(f"lead_time_{h0}_to_{h1}")
    return "__".join(segments) if segments else ""


def build_output_filename(
    metric: str,
    variable: str | list[str] | None = None,
    level: str | int | None = None,
    qualifier: str | None = None,
    init_time_range: tuple[str, str] | None = None,
    lead_time_range: tuple[str, str] | None = None,
    ensemble: str | int | None = None,
    ext: str = "csv",
) -> str:
    """Build standardized output filename.

    Always includes ensemble token (ensmean by default).
    Args:
        metric: Short identifier.
        variable: Variable name; list/None omitted.
        level: Pressure level value.
        qualifier: Extra discriminator (averaged, combined, plot, spectrum, etc.).
        init_time_range: (start,end) timestamps YYYY-MM-DDTHH → init<start>-<end>.
        lead_time_range: (start,end) lead hours → lead<start>-<end>.
        ensemble: Index, 'mean', or None.
        ext: File extension without dot.
    Returns: Filename string.
    """
    parts: list[str] = [metric]
    # Variable: omit if aggregate (None or list)
    if isinstance(variable, list) or variable in (None, ""):
        pass
    else:
        parts.append(str(variable))
    if level is not None:
        parts.append(str(level))
    if qualifier:
        parts.append(str(qualifier))
    # Time ranges (order: init then lead)
    if init_time_range:
        parts.append(f"init{init_time_range[0]}-{init_time_range[1]}")
    if lead_time_range:
        parts.append(f"lead{lead_time_range[0]}-{lead_time_range[1]}")
    # Ensemble token always last before extension
    if ensemble is None:
        # Default: treat deterministic/no-explicit-ensemble as mean for naming consistency
        parts.append("ensmean")
    else:
        ens_lower = str(ensemble).lower()
        # Accept already fully-qualified tokens from resolver
        if ens_lower in {"ensmean", "enspooled", "ensprob"}:
            parts.append(ens_lower)
        elif ens_lower == "mean":
            parts.append("ensmean")
        elif ens_lower.startswith("ens"):
            parts.append(ens_lower)
        else:
            parts.append(f"ens{ensemble}")
    return "_".join(parts) + f".{ext}"


def format_level_token(level: str | int | float | None) -> str:
    """Format level for filenames/tokens (e.g. '500', 'sfc')."""
    if level is None:
        return "sfc"
    s = str(level).lower().strip()
    if s in ("sfc", "surface", "0", "0.0", "-1", "-1.0"):
        return "sfc"
    return str(level).replace(".", "p")


"""Helper utilities for chunking over init and lead times."""


def time_chunks(init_times, lead_times, init_time_chunk_size=None, lead_time_chunk_size=None):
    # Accept non-contiguous init_times; slice by chunk size without assuming uniform spacing
    init_times = init_times.astype("datetime64[ns]")
    total_init = len(init_times)
    step_i = init_time_chunk_size or total_init
    init_time_chunks = [init_times[i : i + step_i] for i in range(0, total_init, step_i)]

    if isinstance(lead_times, slice):
        lead_time_chunks = [lead_times]
    else:
        lead_times = lead_times.astype("timedelta64[ns]")
        total_lead = len(lead_times)
        step_l = lead_time_chunk_size or total_lead
        lead_time_chunks = [lead_times[i : i + step_l] for i in range(0, total_lead, step_l)]

    return itertools.product(init_time_chunks, lead_time_chunks)


# ---------------- Ensemble handling utilities (centralized) -----------------

_DEFAULT_ENSEMBLE_MODES: dict[str, str] = {
    # metrics
    "deterministic": "mean",
    "ets": "mean",
    "probabilistic": "prob",
    "multivariate": "mean",
    # plots / diagnostics
    "energy_spectra": "mean",
    "vertical_profiles": "mean",
    "histograms": "pooled",
    "wd_kde": "pooled",
    "maps": "members",
}

_VALID_MODES = {"mean", "pooled", "prob", "members"}

# Explicit per-module allowed modes (logical + implemented semantics).
_ALLOWED_PER_MODULE: dict[str, set[str]] = {
    "maps": {"mean", "members"},
    "vertical_profiles": {"mean", "pooled", "members"},
    "probabilistic": {"prob"},
    "histograms": {"mean", "pooled", "members"},
    "wd_kde": {"mean", "pooled", "members"},
    "energy_spectra": {"mean", "pooled", "members"},
    "deterministic": {"mean", "pooled", "members"},
    "ets": {"mean", "pooled", "members"},
    "multivariate": {"mean", "pooled", "members"},
}


def resolve_ensemble_mode(
    module: str,
    requested: str | None,
    ds_target,
    ds_prediction,
) -> str:
    """Determine effective ensemble handling mode for a module.

    Modes:
    - mean: reduce ensemble → single field (ensmean)
    - pooled: treat all members' samples jointly (enspooled)
    - prob: keep ensemble dimension intrinsically (ensprob)
    - members: iterate per member producing separate per-member artifacts (ens<idx>)
    """
    base = (requested or _DEFAULT_ENSEMBLE_MODES.get(module, "mean")).lower()
    if base not in _VALID_MODES:
        base = _DEFAULT_ENSEMBLE_MODES.get(module, "mean")
    if module == "probabilistic":
        # Force prob; other modes invalid here.
        return "prob"
    return base


def ensemble_mode_to_token(mode: str | None, member_index: int | None = None) -> str | None:
    """Map resolved ensemble mode to filename token.

    Returns token WITHOUT leading underscore; build_output_filename will append as part list.
    For members mode we expect caller to invoke once per member with member_index.
    """
    if mode == "mean":
        return "ensmean"  # builder normalises to ensmean
    if mode == "pooled":
        return "enspooled"
    if mode == "prob":
        return "ensprob"
    if mode == "members":
        if member_index is None:
            raise ValueError("member_index required when mode='members'")
        return f"ens{member_index}"
    return None


def format_ensemble_log(module: str, mode: str, ens_size: int, selection: str | None = None) -> str:
    """Generate a standardized ensemble handling log line.

    Args:
        module: module name (maps, histograms, wd_kde, energy_spectra, vertical_profiles,
            deterministic, ets, probabilistic)
        mode: resolved ensemble mode (none|mean|pooled|prob|members)
        ens_size: number of ensemble members (if present)
        selection: optional description of subset (e.g. "selected 3 of 8")
    Returns:
        Human-readable single line.
    """
    base = f"Ensemble (size={ens_size})" if ens_size >= 0 else "Ensemble"
    if selection:
        base += f" {selection}"
    mode_desc = {
        "mean": "mode=mean token=ensmean",
        "pooled": "mode=pooled token=enspooled",
        "prob": "mode=prob token=ensprob",
        "members": "mode=members tokens=ens0..ensN",
    }.get(mode, f"mode={mode}")
    return f"{base} → {module}: {mode_desc}."


def validate_and_normalize_ensemble_config(
    ensemble_cfg: dict | None,
    has_ensemble: bool,
) -> tuple[dict, list[str]]:
    """Validate and normalize user-specified per-module ensemble modes.

    Behaviour:
    * Accept typo 'member' → normalize to 'members'.
    * Lower-case values; unknown modes replaced by module default with warning.
    * If ensemble dim absent, any non-'mean' mode downgraded to 'mean' with warning.

    Returns (normalized_config, warnings_list).
    """
    if ensemble_cfg is None:
        return {}, []
    warnings: list[str] = []
    normalized: dict = {}
    for module, value in ensemble_cfg.items():
        if value is None:
            normalized[module] = None
            continue
        val_raw = str(value)
        val = val_raw.strip().lower()
        if val == "member":  # common typo
            warnings.append(f"ensemble.{module}='member' corrected to 'members' (per-member mode).")
            val = "members"
        if val not in _VALID_MODES:
            default = _DEFAULT_ENSEMBLE_MODES.get(module, "mean")
            warnings.append(
                f"ensemble.{module}='{val_raw}' is invalid; falling back to default '{default}'."
            )
            val = default
        # Enforce per-module allowed set.
        allowed = _ALLOWED_PER_MODULE.get(module)
        if allowed is not None and val not in allowed:
            # If user provided 'pooled' for maps or vertical_profiles, provide actionable message.
            raise ValueError(
                "Unsupported ensemble mode for module: "
                f"ensemble.{module}='{val_raw}' not in allowed set {sorted(allowed)}."
            )
        normalized[module] = val
    return normalized, warnings


def display_outputs(
    output_dir, pattern_img="*.png", pattern_csv="*.csv", limit=None, exclude_pattern=None
):
    """Display all images and tables in the given directory matching the patterns.

    Args:
        output_dir: Path to the directory containing outputs.
        pattern_img: Glob pattern for images (default: "*.png"). Set to None or "" to skip.
        pattern_csv: Glob pattern for CSV tables (default: "*.csv"). Set to None or "" to skip.
        limit: Maximum number of images to display (default: None, show all).
        exclude_pattern: Substring to exclude from filenames (default: None).
    """
    from itertools import islice
    from pathlib import Path

    from IPython.display import HTML, Image, display

    def natural_key(file_path):
        """Key for natural sorting (numbers sorted numerically)."""
        parts = file_path.stem.split("_")
        key = []
        for part in parts:
            try:
                val = int(part)
                key.append((0, val))
            except ValueError:
                key.append((1, part))
        return key

    path = Path(output_dir)
    if not path.exists():
        c.print(f"Directory not found: {path}")
        return

    # Images
    if pattern_img:
        images = sorted(path.glob(pattern_img), key=natural_key)
        if exclude_pattern:
            if isinstance(exclude_pattern, str):
                exclude_patterns = [exclude_pattern]
            else:
                exclude_patterns = exclude_pattern
            images = [img for img in images if not any(pat in img.name for pat in exclude_patterns)]

        if images:
            to_show = images if limit is None else list(islice(images, 0, limit))
            count_str = f"({len(to_show)}/{len(images)})" if limit is not None else ""
            c.print(f"--- Images in {path.name} {count_str} ---")
            for img in to_show:
                c.print(f"Displaying: {img.name}")
                display(Image(filename=str(img)))
        else:
            c.print(f"No images found for pattern '{pattern_img}' in {path.name}")

    # Tables
    if pattern_csv:
        tables = sorted(path.glob(pattern_csv), key=natural_key)
        if exclude_pattern:
            if isinstance(exclude_pattern, str):
                exclude_patterns = [exclude_pattern]
            else:
                exclude_patterns = exclude_pattern
            tables = [tbl for tbl in tables if not any(pat in tbl.name for pat in exclude_patterns)]

        if tables:
            c.print(f"--- Tables in {path.name} ---")
            for tbl in tables:
                c.print(f"Table: {tbl.name}")
                try:
                    df = pd.read_csv(tbl)
                    with pd.option_context("display.max_rows", None):
                        display(
                            HTML(
                                f"<div style='max-height: 350px; overflow: "
                                f"auto;'>{df.to_html()}</div>"
                            )
                        )
                except Exception as e:
                    c.print(f"Could not read {tbl.name}: {e}")
        else:
            c.print(f"No tables found for pattern '{pattern_csv}' in {path.name}")


def extract_date_from_filename(filename: str) -> str:
    """Extract date suffix from filename if it contains a single init time.

    Looks for pattern 'init<YYYY-MM-DDTHHstart>-<YYYY-MM-DDTHHend>'. If start == end, returns
    ' (<start>)'. Also checks for 'lead<start>-<end>' and if start == end, appends ' +<start>h'.
    Otherwise returns empty string.
    """
    suffix = ""
    match_init = re.search(r"init(\d{4}-?\d{2}-?\d{2}T\d{2})-(\d{4}-?\d{2}-?\d{2}T\d{2})", filename)
    if match_init:
        start, end = match_init.groups()
        if start == end:
            # Normalize to YYYY-MM-DDTHH
            if len(start) == 10 and "T" not in start:
                start = start[:8] + "T" + start[8:]
            if len(start) == 11 and "T" in start and "-" not in start:
                start = f"{start[:4]}-{start[4:6]}-{start[6:8]}{start[8:]}"
            suffix = f" ({start}"

    if suffix:
        match_lead = re.search(r"lead(\d+)h?-(\d+)h?", filename)
        if match_lead:
            start_lead, end_lead = match_lead.groups()
            if start_lead == end_lead:
                suffix += f" +{start_lead}h"
        suffix += ")"
        return suffix
    return ""


def extract_date_from_dataset(ds: Any) -> str:
    """Extract date suffix from dataset if it contains a single init time.

    Checks 'init_time' coordinate. If size is 1, formats as ' (YYYY-MM-DDTHH)'.
    Also checks 'lead_time' coordinate. If size is 1, appends ' +Xh'.
    Otherwise returns empty string.
    """
    if not hasattr(ds, "coords") or "init_time" not in ds.coords:
        return ""

    try:
        its = ds.coords["init_time"]
        if its.size == 1:
            # Use values directly to avoid .item() converting datetime64 to int (ns)
            # Handle both scalar (0-d) and 1-d arrays
            ts_val = its.values if its.ndim == 0 else its.values.flatten()[0]
            ts = np.datetime64(ts_val).astype("datetime64[h]")
            suffix = f" ({np.datetime_as_string(ts, unit='h').replace(':', '')}"

            if "lead_time" in ds.coords:
                lts = ds.coords["lead_time"]
                if lts.size == 1:
                    lt_val = lts.values if lts.ndim == 0 else lts.values.flatten()[0]
                    if np.issubdtype(type(lt_val), np.timedelta64):
                        h = int(lt_val / np.timedelta64(1, "h"))
                        suffix += f" +{h}h"
                    else:
                        # Fallback if lead_time is not timedelta (e.g. int hours)
                        suffix += f" +{int(lt_val)}h"

            suffix += ")"
            return suffix
    except Exception:
        # If extraction or formatting fails, return empty string as fallback.
        pass
    return ""


def format_variable_name(var_name: str) -> str:
    """Format variable name for plot titles (e.g. '2m_temperature' -> '2m Temperature')."""
    formatted = " ".join(word.capitalize() for word in var_name.replace("_", " ").split())
    # Remove trailing 2d/3d indicators
    lower = formatted.lower()
    if lower.endswith(" 2d") or lower.endswith(" 3d"):
        formatted = formatted[:-3]
    return formatted


def format_level_label(level: str | int | float | None) -> str:
    """Format level label for plot titles.

    Returns empty string for surface levels ('sfc', 'surface', 0),
    otherwise returns ' (Level {level})'.
    """
    if level is None:
        return ""

    lvl_str = str(level).lower().strip()
    if lvl_str in ("sfc", "surface", "0", "0.0", "-1", "-1.0"):
        return ""

    return f" (Level {level})"


# ── Spatial metric map specifications (MAE / RMSE / Bias) ────────────────────
# Single source-of-truth used by the deterministic orchestrator (generation)
# and the intercomparison maps module (comparison).  Keys are title-case to
# match the deterministic ``include`` config list; the ``key`` field gives the
# lower-case token used in filenames and NPZ arrays.
SPATIAL_METRIC_SPECS: dict[str, dict] = {
    "MAE": {
        "key": "mae",
        "fn": lambda pred, tgt: np.abs(pred - tgt),
        "cmap": "RdBu_r",
        "vmin_zero": True,
        "diverging": False,
    },
    "RMSE": {
        "key": "rmse",
        "fn": lambda pred, tgt: np.sqrt((pred - tgt) ** 2),
        "cmap": "RdBu_r",
        "vmin_zero": True,
        "diverging": False,
    },
    "Bias": {
        "key": "bias",
        "fn": lambda pred, tgt: pred - tgt,
        "cmap": "RdBu_r",
        "vmin_zero": False,
        "diverging": True,
    },
}


# Common variable units fallback mapping
VARIABLE_UNITS = {
    "2m_temperature": "K",
    "temperature": "K",
    "10m_u_component_of_wind": "m s**-1",
    "10m_v_component_of_wind": "m s**-1",
    "u_component_of_wind": "m s**-1",
    "v_component_of_wind": "m s**-1",
    # Derived wind variables
    "wind_speed": "m s**-1",
    "10m_wind_speed": "m s**-1",
    # Derived geopotential height and its gradient
    "geopotential_height": "m",
    "geopotential_height_gradient": "m m**-1",
    "geopotential": "m**2 s**-2",
    "specific_humidity": "kg kg**-1",
    "mean_sea_level_pressure": "Pa",
    "total_precipitation": "m",
}


def _to_latex_units(unit_str: str) -> str:
    """Convert CF-style unit strings to LaTeX ratio notation.

    Examples: 'm s**-1' -> '$\\mathrm{m/s}$', 'kg kg**-1' -> '$\\mathrm{kg/kg}$',
    's**-1' -> '$\\mathrm{1/s}$', 'm**2 s**-2' -> '$\\mathrm{m^{2}/s^{2}}$'.
    """
    if not unit_str:
        return unit_str
    if "$" in unit_str or "\\" in unit_str:
        return unit_str  # already LaTeX
    # Convert ** exponent notation to ^{N}
    s = re.sub(r"\*\*(-?\d+)", lambda m: f"^{{{m.group(1)}}}", unit_str)
    # Split tokens and separate into numerator / denominator by exponent sign
    _neg = re.compile(r"^(.+?)\^\{(-\d+)\}$")
    numer: list[str] = []
    denom: list[str] = []
    for tok in s.split():
        m = _neg.match(tok)
        if m:
            base, abs_exp = m.group(1), -int(m.group(2))
            denom.append(base if abs_exp == 1 else f"{base}^{{{abs_exp}}}")
        else:
            numer.append(tok)
    numer_str = r"\,".join(numer) if numer else "1"
    combined = numer_str + "/" + r"\,".join(denom) if denom else numer_str
    return rf"$\mathrm{{{combined}}}$"


def get_variable_units(
    ds: xr.Dataset | xr.DataArray | None, var_name: str, latex: bool = False
) -> str:
    """Get units for a variable, falling back to a default mapping if missing.

    Args:
        ds: Dataset or DataArray to read units from, or ``None``.
        var_name: Variable name used for the fallback lookup.
        latex: If ``True``, return LaTeX-formatted units suitable for plot
            labels.  If ``False`` (default), return plain CF-style strings
            suitable for metadata, CSV headers, and downstream unit parsing.
    """
    if ds is not None:
        if isinstance(ds, xr.DataArray):
            if "units" in ds.attrs:
                raw = str(ds.attrs["units"])
                return _to_latex_units(raw) if latex else raw
        elif var_name in ds and "units" in ds[var_name].attrs:
            raw = str(ds[var_name].attrs["units"])
            return _to_latex_units(raw) if latex else raw
    raw = VARIABLE_UNITS.get(var_name, "")
    return _to_latex_units(raw) if latex else raw


def subsample_values(
    da: xr.DataArray, k: int | None, seed: int, lazy: bool = False
) -> np.ndarray | dsa.Array | None:
    """Dimension-aware uniform subsample across all dims.

    Uses per-dimension index sampling so very large arrays don't need to be fully
    materialized. Always pairs subsamples when given the same seed.

    Args:
        da: Input DataArray
        k: Number of samples to take (approximate)
        seed: Random seed
        lazy: If True, return a dask array without computing.
            If False, compute and return numpy array with finite values only.
    """
    size = int(getattr(da, "size", 0) or 0)
    if size == 0:
        return None if lazy else np.array([], dtype=float)

    # If k is None or >= size, take all valid values
    if k is None or size <= k:
        if lazy:
            return da.data.flatten()
        else:
            arr = np.asarray(da.compute().values).ravel()
            return arr[np.isfinite(arr)]

    # Subsampling logic
    dims = list(da.dims)
    nd = max(1, len(dims))
    frac = (k / float(size)) ** (1.0 / nd)
    rng = np.random.default_rng(seed)

    indexers: dict[str, Any] = {}
    for d in dims:
        n = int(da.sizes.get(str(d), 1))
        take = max(1, int(np.ceil(frac * n)))
        take = min(take, n)
        idx = rng.choice(n, size=take, replace=False)
        idx.sort()
        indexers[str(d)] = idx  # numpy array

    # Lazy selection
    sub = da.isel(indexers)

    if lazy:
        return sub.data.flatten()
    else:
        arr = np.asarray(sub.compute().values).ravel()
        return arr[np.isfinite(arr)]


def save_figure(fig: plt.Figure, path: Path, dpi: int = 200, module: str | None = None) -> None:
    """Save figure to path, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    prefix = f"[{module}] " if module else ""
    c.print(f"{prefix}Saved {path}")
    plt.close(fig)


def save_data(path: Path, module: str | None = None, **kwargs: Any) -> None:
    """Save data to NPZ file, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **kwargs)
    prefix = f"[{module}] " if module else ""
    c.print(f"{prefix}Saved {path}")


def save_dataframe(
    df: pd.DataFrame,
    path: Path,
    index: bool = True,
    index_label: str | None = None,
    module: str | None = None,
    **kwargs: Any,
) -> None:
    """Save pandas DataFrame to CSV file, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, index_label=index_label, na_rep="NaN", **kwargs)
    prefix = f"[{module}] " if module else ""
    c.print(f"{prefix}Saved {path}")


def save_metric_by_lead_tables(
    long_df: pd.DataFrame,
    section_output: Path,
    metric: str,
    init_time_range: tuple[str, str] | None,
    lead_time_range: tuple[str, str] | None,
    ensemble: str | int | None,
    module: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Save long and wide by-lead metric tables with standardized naming.

    Expects long_df with columns:
    - required: lead_time_hours, variable
    - optional: level
    - one or more metric value columns
    """
    if long_df.empty:
        return long_df, pd.DataFrame()

    if "lead_time_hours" not in long_df.columns or "variable" not in long_df.columns:
        raise ValueError("long_df must contain 'lead_time_hours' and 'variable' columns")

    id_cols = ["lead_time_hours", "variable"]
    if "level" in long_df.columns:
        id_cols.append("level")

    cols = id_cols + [col for col in long_df.columns if col not in id_cols]
    long_df = long_df[cols].copy()
    long_df = long_df.sort_values(id_cols).reset_index(drop=True)

    out_long = section_output / build_output_filename(
        metric=metric,
        variable=None,
        level=None,
        qualifier="by_lead_long",
        init_time_range=init_time_range,
        lead_time_range=lead_time_range,
        ensemble=ensemble,
        ext="csv",
    )
    save_dataframe(long_df, out_long, index=False, module=module)

    melted = long_df.melt(id_vars=id_cols, var_name="metric", value_name="value")
    if "level" in melted.columns:
        level_token = (
            pd.to_numeric(melted["level"], errors="coerce")
            .astype("Int64")
            .astype(str)
            .replace("<NA>", "")
        )
        level_suffix = np.where(level_token != "", "_" + level_token, "")
        melted["key"] = (
            melted["variable"].astype(str) + level_suffix + "_" + melted["metric"].astype(str)
        )
    else:
        melted["key"] = melted["variable"].astype(str) + "_" + melted["metric"].astype(str)

    wide_df = melted.pivot(index="lead_time_hours", columns="key", values="value").reset_index()
    wide_df.columns.name = None

    out_wide = section_output / build_output_filename(
        metric=metric,
        variable=None,
        level=None,
        qualifier="by_lead_wide",
        init_time_range=init_time_range,
        lead_time_range=lead_time_range,
        ensemble=ensemble,
        ext="csv",
    )
    save_dataframe(wide_df, out_wide, index=False, module=module)

    return long_df, wide_df


def unwrap_longitude_for_plot(da: xr.DataArray, lon_name: str = "longitude") -> xr.DataArray:
    """Unwrap wrapped longitudes for plotting (e.g. 335..360 U 0..45 -> -25..45).

    Detects if the domain crosses the prime meridian (values < 90 and > 270 present)
    and shifts the western segment (values > 180) to negative degrees, then re-sorts.
    """
    if lon_name not in da.coords:
        return da

    lons = np.asarray(da[lon_name].values)
    if lons.size == 0:
        return da

    lmin, lmax = float(np.nanmin(lons)), float(np.nanmax(lons))

    # Heuristic: large span plus presence of values on both sides of Greenwich
    if (lmax - lmin) > 180 and np.any(lons < 90) and np.any(lons > 270):
        new = lons.copy()
        new[new > 180] -= 360  # shift western segment to negative degrees
        order = np.argsort(new)
        da = da.isel({lon_name: order}).assign_coords({lon_name: (lon_name, new[order])})

    return da
