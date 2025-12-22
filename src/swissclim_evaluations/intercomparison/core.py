from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from swissclim_evaluations.helpers import format_variable_name

# Rich-style console utilities for consistent terminal output
try:  # pragma: no cover (console printing)
    from swissclim_evaluations import console as c
except ImportError:
    try:
        import console as c
    except ImportError:

        class _DummyConsole:
            def __getattr__(self, _name):
                def _noop(*args, **kwargs):
                    # Fallback to basic print when console is not available
                    if args:
                        print(*args, flush=True)

                return _noop

        c = _DummyConsole()

# Global flag for quiet mode
quiet = False


def as_paths(items: Iterable[str]) -> list[Path]:
    return [Path(x).resolve() for x in items]


def model_label(p: Path, explicit: str | None = None) -> str:
    return explicit if explicit else p.name


def scan_model_sets(models: list[Path], rel_glob: str) -> tuple[list[set[str]], set[str], set[str]]:
    """Return per-model sets, intersection and union for a relative glob pattern.

    rel_glob: e.g. "energy_spectra/*_spectrum*.npz", "maps/map_*.npz".
    """
    dir_part = rel_glob.split("/")[0]
    pat_part = "/".join(rel_glob.split("/")[1:]) if "/" in rel_glob else rel_glob
    per_model: list[set[str]] = []
    for m in models:
        base = (m / dir_part) if dir_part and dir_part != rel_glob else m
        files = list(base.glob(pat_part))
        per_model.append({f.name for f in files if f.is_file()})
    inter = set.intersection(*per_model) if per_model else set()
    uni = set().union(*per_model) if per_model else set()
    return per_model, inter, uni


def load_npz(path: Path) -> dict:
    """Load an NPZ file and return its content as a dict."""
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def report_missing(
    module: str,
    models: list[Path],
    labels: list[str],
    per_model: list[set[str]],
    union: set[str],
) -> None:
    """Pretty-print which basenames were missing per model for a module scan."""
    if not union:
        c.warn(f"[{module}] No files found in any model.")
        return

    intersection = set.intersection(*per_model) if per_model else set()

    rows: list[str] = []
    rows.append(f"Total unique files: {len(union)}")
    rows.append(f"Common files (in all models): {len(intersection)}")
    rows.append("")  # Spacer

    for lab, files in zip(labels, per_model, strict=False):
        missing = sorted(union - files)
        rows.append(f"• {lab}: present={len(files)} missing={len(missing)}")
        if missing:
            preview = ", ".join(missing[:8])
            if len(missing) > 8:
                preview += ", …"
            rows.append(f"  ↳ missing: {preview}")

    rows.append("")
    rows.append("(Missing counts are relative to the union of files found across all models)")
    rows.append("(Counts refer to files, not necessarily unique atmospheric variables)")

    c.panel(
        "\n".join(rows),
        title=f"Input Availability — {module}",
        style="yellow",
    )


def report_checklist(module: str, results: dict[str, int]) -> None:
    """Print a checklist panel for the module with counts."""
    lines = []
    for label, count in results.items():
        if count > 0:
            if "(Ignored)" in label:
                lines.append(f"❌ {label} ({count})")
            else:
                lines.append(f"✅ {label} ({count})")
        else:
            clean_label = label.replace(" (Ignored)", "")
            lines.append(f"❌ {clean_label} (Missing)")

    lines.append("")
    lines.append("(Counts refer to files, not necessarily unique atmospheric variables)")

    c.panel(
        "\n".join(lines),
        title=f"Output Checklist — {module}",
        style="blue",
    )


def print_file_list(title: str, files: list[str]) -> None:
    """Print a list of files with a header message, using bullets on new lines."""
    if not files:
        return
    # Format: "Header:\n  • item1\n  • item2"
    if not quiet:
        c.print(f"[bold cyan]{title}:[/bold cyan]")
        for f in files:
            c.print(f"  - {f}")


def clean_var_from_filename(filename: str, prefix: str = "", format: bool = True) -> str:
    """Clean variable name from filename for plot titles."""
    stem = filename[:-4] if filename.endswith(".npz") else filename
    if prefix and stem.startswith(prefix):
        stem = stem[len(prefix) :]

    # Remove common suffixes/tokens
    for token in ["_global", "_latbands", "_combined", "_grid", "_data", "_surface"]:
        stem = stem.replace(token, "")

    # Remove ensemble token
    if "_ens" in stem:
        stem = stem.rsplit("_ens", 1)[0]

    # Remove init/lead time patterns
    # initYYYY-MM-DDTHH-YYYY-MM-DDTHH
    stem = re.sub(r"_init\d{4}-?\d{2}-?\d{2}T\d{2}-\d{4}-?\d{2}-?\d{2}T\d{2}", "", stem)
    # leadXXXh-YYYh
    stem = re.sub(r"_lead\d+h-\d+h", "", stem)

    if format:
        return format_variable_name(stem)
    return stem


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def common_files(models: list[Path], rel_glob: str) -> list[str]:
    """Find filenames (basenames) that exist in ALL model folders for a given relative glob.

    Returns a sorted list of basenames present in all model folders that match the pattern.
    """
    sets: list[set[str]] = []
    for m in models:
        files = (
            list((m / rel_glob.split("/")[0]).glob("/".join(rel_glob.split("/")[1:])))
            if "/" in rel_glob
            else list(m.glob(rel_glob))
        )
        sets.append({f.name for f in files if f.is_file()})
    if not sets:
        return []
    common = set.intersection(*sets) if len(sets) > 1 else sets[0]
    return sorted(common)
