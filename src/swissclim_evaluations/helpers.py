import contextlib
import itertools
from typing import Any

import numpy as np
import xarray as xr


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


def _fmt_init(ts: np.ndarray) -> tuple[str, str]:
    """Format init_time array (datetime64) to YYYYMMDDHH strings (hour precision)."""
    if ts.size == 0:
        return ("", "")
    start = np.datetime64(ts.min()).astype("datetime64[h]")
    end = np.datetime64(ts.max()).astype("datetime64[h]")

    def _fmt(x):
        return np.datetime_as_string(x, unit="h").replace("-", "").replace(":", "").replace("T", "")

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
    Datetime formatted as YYYYMMDDHH (no separators). Lead times in hours (ints).
    """
    segments: list[str] = []
    if "init_time" in ds.coords:
        try:
            init_vals = np.asarray(ds["init_time"].values)
            if init_vals.size:
                s, e = _fmt_init(init_vals)
                if s and e:
                    segments.append(f"init_time_{s}_to_{e}")
        except Exception:
            pass
    if "lead_time" in ds.coords:
        try:
            lead_vals = np.asarray(ds["lead_time"].values)
            if lead_vals.size:
                h0, h1 = _fmt_lead(lead_vals)
                segments.append(f"lead_time_{h0}_to_{h1}")
        except Exception:
            pass
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
        init_time_range: (start,end) timestamps YYYYMMDDHH → init<start>-<end>.
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


def format_level_token(level: Any) -> str:
    """Return a filesystem-safe label for a single level value."""
    value = level.item() if hasattr(level, "item") else level
    try:
        as_int = int(value)
        if float(as_int) == float(value):
            return str(as_int)
    except Exception:
        pass
    return str(value).replace(".", "_")


"""Helper utilities for chunking over init and lead times."""


def time_chunks(init_times, lead_times, init_time_chunk_size=None, lead_time_chunk_size=None):
    # Accept non-contiguous init_times; slice by chunk size without assuming uniform spacing
    with contextlib.suppress(Exception):
        init_times = init_times.astype("datetime64[ns]")
    total_init = len(init_times)
    step_i = init_time_chunk_size or total_init
    init_time_chunks = [init_times[i : i + step_i] for i in range(0, total_init, step_i)]

    if isinstance(lead_times, slice):
        lead_time_chunks = [lead_times]
    else:
        with contextlib.suppress(Exception):
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
    # plots / diagnostics
    "energy_spectra": "mean",
    "vertical_profiles": "mean",
    "histograms": "pooled",
    "wd_kde": "pooled",
    "maps": "members",
}

_VALID_MODES = {"none", "mean", "pooled", "prob", "members"}

# Explicit per-module allowed modes (logical + implemented semantics).
_ALLOWED_PER_MODULE: dict[str, set[str]] = {
    "maps": {"none", "mean", "members"},
    "vertical_profiles": {"none", "mean", "pooled", "members"},
    "probabilistic": {"prob"},
    "histograms": {"none", "mean", "pooled", "members"},
    "wd_kde": {"none", "mean", "pooled", "members"},
    "energy_spectra": {"none", "mean", "pooled", "members"},
    "deterministic": {"none", "mean", "pooled", "members"},
    "ets": {"none", "mean", "pooled", "members"},
}


def resolve_ensemble_mode(
    module: str,
    requested: str | None,
    ds_target,
    ds_prediction,
) -> str:
    """Determine effective ensemble handling mode for a module.

    Modes:
      - none: no ensemble behaviour (or ensemble dim absent)
      - mean: reduce ensemble → single field (ensmean)
      - pooled: treat all members' samples jointly (enspooled)
      - prob: keep ensemble dimension intrinsically (ensprob)
      - members: iterate per member producing separate per-member artifacts (ens<idx>)
    """
    has_ens = "ensemble" in getattr(ds_prediction, "dims", {})
    base = (requested or _DEFAULT_ENSEMBLE_MODES.get(module, "none")).lower()
    if base not in _VALID_MODES:
        base = _DEFAULT_ENSEMBLE_MODES.get(module, "none")
    if not has_ens:
        # If no ensemble dim, collapse to none (probabilistic handled upstream).
        return "none"
    if module == "probabilistic":
        # Force prob; other modes invalid here.
        return "prob"
    return base


def ensemble_mode_to_token(mode: str, member_index: int | None = None) -> str | None:
    """Map resolved ensemble mode to filename token.

    Returns token WITHOUT leading underscore; build_output_filename will append as part list.
    For members mode we expect caller to invoke once per member with member_index.
    """
    if mode == "none":
        return None  # builder will inject default 'ensmean'
    if mode == "mean":
        return "mean"  # builder normalises to ensmean
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
        "none": "no ensemble behaviour",
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
            * If ensemble dimension present and user sets 'none' (except probabilistic), replace
                with module default (mean/pooled/members/prob) and warn.
      * If ensemble dim absent, any non-'none' mode downgraded to 'none' with warning.

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
            default = _DEFAULT_ENSEMBLE_MODES.get(module, "none")
            warnings.append(
                f"ensemble.{module}='{val_raw}' is invalid; falling back to default '{default}'."
            )
            val = default
        # Enforce per-module allowed set (before handling 'none' degradations).
        allowed = _ALLOWED_PER_MODULE.get(module)
        if allowed is not None and val not in allowed:
            # If user provided 'pooled' for maps or vertical_profiles, provide actionable message.
            raise ValueError(
                "Unsupported ensemble mode for module: "
                f"ensemble.{module}='{val_raw}' not in allowed set {sorted(allowed)}."
            )
        if has_ensemble:
            if val == "none" and module != "probabilistic":
                default = _DEFAULT_ENSEMBLE_MODES.get(module, "mean")
                warnings.append(
                    "[ensemble-fallback] ensemble."
                    f"{module}='none' while ensemble dimension exists → using "
                    f"'{default}' instead."
                )
                val = default
        else:
            if val != "none":
                warnings.append(
                    f"ensemble.{module}='{val}' ignored (no ensemble dimension) → using 'none'."
                )
                val = "none"
        normalized[module] = val
    return normalized, warnings
