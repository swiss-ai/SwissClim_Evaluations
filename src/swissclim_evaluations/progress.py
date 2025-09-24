from __future__ import annotations

"""Progress utilities for SwissClim Evaluations.

Features:
- One progress bar per logical module (e.g. deterministic, ets, maps, probabilistic)
- Rich based when available, with graceful plain fallback when Rich is absent or disabled
- Simple iterator wrapper: for item in iter_progress(seq, module="deterministic", total=len(seq)):
- Dask progress callback helper: dask.compute(..., scheduler='threads') with transient task progress

Design goals:
- Zero-cost when disabled (env SWISSCLIM_PROGRESS=0 or Rich unavailable)
- Non-intrusive integration: minimal code changes (wrap loops)

Environment variables:
- SWISSCLIM_PROGRESS: "1" (default auto) to enable if Rich present; "0" to force disable.
"""

import os
import time
from typing import Dict, Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")

try:  # Rich optional
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        TaskID,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    _HAS_RICH = True
except Exception:  # pragma: no cover
    _HAS_RICH = False

_ENABLE = os.environ.get("SWISSCLIM_PROGRESS", "1") != "0" and _HAS_RICH

_console: Console | None = None
_progress: Progress | None = None
_tasks: Dict[str, TaskID] = {}


def _ensure_progress() -> None:
    global _progress, _console
    if not _ENABLE:
        return
    if _progress is None:
        _console = Console(highlight=False, soft_wrap=True)
        _progress = Progress(
            TextColumn("[bold]{task.fields[module]}[/]"),
            BarColumn(bar_width=None),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=False,
            console=_console,
            refresh_per_second=4,
        )
        _progress.start()


def shutdown() -> None:
    """Stop progress display (called at end of CLI run)."""
    global _progress
    if _progress is not None:
        try:
            _progress.stop()
        except Exception:
            pass
        _progress = None
        _tasks.clear()


def iter_progress(
    seq: Iterable[T], module: str, total: Optional[int] = None
) -> Iterator[T]:
    """Wrap an iterable with a module progress bar.

    Parameters
    ----------
    seq : iterable
        Sequence to iterate.
    module : str
        Logical module name (deterministic, ets, maps, probabilistic, etc.)
    total : int, optional
        Total length; if None and seq has __len__, it is inferred.
    """
    if not _ENABLE:
        yield from seq
        return
    _ensure_progress()
    global _progress
    assert _progress is not None  # for type checkers
    if total is None:
        try:
            total = len(seq)  # type: ignore[arg-type]
        except Exception:
            total = None
    task_id: TaskID
    if module not in _tasks:
        task_id = _progress.add_task("work", total=total, module=module)
        _tasks[module] = task_id
    else:
        task_id = _tasks[module]
        if total is not None:
            _progress.update(task_id, total=total)
    for item in seq:
        yield item
        try:
            _progress.advance(task_id, 1)
        except Exception:
            pass
    # leave bar; not marking finished allows subsequent loops to re-use (optional)


def dask_progress(futures, module: str, description: str = "dask.compute"):
    """Track progress for a collection of dask delayed objects / futures.

    Usage:
        result = dask_progress(dask.persist(obj), module="deterministic")
        or
        result = dask_progress(dask.compute(a, b), module="probabilistic")

    This is a very lightweight heuristic: if futures supports len() we set total, then
    iterate over them if they are already computed (noop) or return result directly.
    For full task graph progress users can still set DASK_PROGRESS=1 and rely on
    Dask's own diagnostics.
    """
    if not _ENABLE:
        return futures
    _ensure_progress()
    total = None
    try:
        total = len(futures)  # type: ignore[arg-type]
    except Exception:
        pass
    # register a transient task
    global _progress
    assert _progress is not None
    tid = _progress.add_task(description, total=total, module=module)
    # If iterable, consume to advance
    if total is not None and hasattr(futures, "__iter__"):
        out = []
        for f in futures:
            out.append(f)
            try:
                _progress.advance(tid, 1)
            except Exception:
                pass
        _progress.update(tid, completed=total)
        return out
    # Fallback: simulate brief spinner
    start = time.time()
    while not _progress.finished and (time.time() - start) < 0.1:
        time.sleep(0.01)
    return futures


__all__ = [
    "iter_progress",
    "dask_progress",
    "shutdown",
]
