import itertools

import numpy as np
import xarray as xr


def _fmt_init(ts: np.ndarray) -> tuple[str, str]:
    """Format init_time array (datetime64) to YYYYMMDDHH strings (hour precision)."""
    if ts.size == 0:
        return ("", "")
    start = np.datetime64(ts.min()).astype("datetime64[h]")
    end = np.datetime64(ts.max()).astype("datetime64[h]")

    def _fmt(x):
        return (
            np.datetime_as_string(x, unit="h")
            .replace("-", "")
            .replace(":", "")
            .replace("T", "")
        )

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
    """
    Build standardized output filename for metrics, plots, arrays.
    Always includes ensemble token (ensnone if not present).
    Args:
        metric: Short identifier for artifact family.
        variable: Variable name. If list or None -> omitted.
        level: Pressure level value; omitted if None.
        qualifier: Optional extra discriminator (averaged, combined, plot, spectrum, etc.).
        init_time_range: (start, end) hour timestamps as YYYYMMDDHH; if provided encoded as init<start>-<end>.
        lead_time_range: (start, end) lead hours (already zero-padded); encoded as lead<start>-<end>.
        ensemble: Index, 'mean', or None (becomes ensnone).
        ext: File extension without leading dot.
    Returns:
        Filename string.
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
        parts.append("ensnone")
    else:
        ens_lower = str(ensemble).lower()
        if ens_lower in ("mean", "ensmean"):
            parts.append("ensmean")
        elif ens_lower.startswith("ens"):
            parts.append(ens_lower)
        else:
            parts.append(f"ens{ensemble}")
    return "_".join(parts) + f".{ext}"


"""Helper utilities for chunking over init and lead times."""


def time_chunks(
    init_times, lead_times, init_time_chunk_size=None, lead_time_chunk_size=None
):
    # Accept non-contiguous init_times; just slice by chunk size without assuming uniform spacing
    try:
        init_times = init_times.astype("datetime64[ns]")
    except Exception:
        pass
    total_init = len(init_times)
    step_i = init_time_chunk_size or total_init
    init_time_chunks = [
        init_times[i : i + step_i] for i in range(0, total_init, step_i)
    ]

    if isinstance(lead_times, slice):
        lead_time_chunks = [lead_times]
    else:
        try:
            lead_times = lead_times.astype("timedelta64[ns]")
        except Exception:
            pass
        total_lead = len(lead_times)
        step_l = lead_time_chunk_size or total_lead
        lead_time_chunks = [
            lead_times[i : i + step_l] for i in range(0, total_lead, step_l)
        ]

    return itertools.product(init_time_chunks, lead_time_chunks)
