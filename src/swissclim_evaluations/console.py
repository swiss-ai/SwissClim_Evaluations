"""Pretty console helpers for SwissClim Evaluations.

Uses Rich for modern, colorized output with emojis and clean layouts.
Falls back to plain prints if Rich isn't available.
"""

from __future__ import annotations

import os
import re
from typing import Any

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text

    _HAS_RICH = True
except Exception:  # pragma: no cover - fallback when Rich not installed
    _HAS_RICH = False


class _PlainConsole:
    def print(self, *objects: Any, **kwargs: Any) -> None:  # noqa: D401
        # Plain print fallback
        print(*objects, flush=True)


_COLOR_PREF = os.environ.get("SWISSCLIM_COLOR", "always").lower()
# always|auto|never
USE_RICH = bool(_HAS_RICH and _COLOR_PREF != "never")
if USE_RICH:
    force = _COLOR_PREF == "always"
    console = Console(
        highlight=False,
        force_terminal=True if force else None,
        color_system="truecolor",
        soft_wrap=True,
    )
else:
    console = _PlainConsole()


_MARKUP_RE = re.compile(r"\[/?[^\]]+\]")


def _strip_markup(text: str) -> str:
    """Remove Rich markup tags from a string for plain-text logs."""
    try:
        return _MARKUP_RE.sub("", text)
    except Exception:
        return text


def header(title: str) -> None:
    if USE_RICH:
        console.print(Rule(Text.from_markup(f"[bold cyan]🚀 {title}[/]")))
    else:
        console.print(f"===== {title} =====")


def section(title: str) -> None:
    if USE_RICH:
        console.print(Rule(Text.from_markup(f"[bold]{title}[/]")))
    else:
        console.print(f"--- {title} ---")


def info(msg: str) -> None:
    prefix = "ℹ️  "
    if USE_RICH:
        console.print(f"[cyan]{prefix}[bold]{msg}[/]")
    else:
        console.print(prefix + msg)


def success(msg: str) -> None:
    prefix = "✅ "
    if USE_RICH:
        console.print(f"[green]{prefix}[bold]{msg}[/]")
    else:
        console.print(prefix + msg)


def warn(msg: str) -> None:
    prefix = "⚠️  "
    if USE_RICH:
        console.print(f"[yellow]{prefix}[bold]{msg}[/]")
    else:
        console.print(prefix + msg)


def error(msg: str) -> None:
    prefix = "❌ "
    if USE_RICH:
        console.print(f"[red]{prefix}[bold]{msg}[/]")
    else:
        console.print(prefix + msg)


def panel(content: str, title: str | None = None, style: str = "") -> None:
    if USE_RICH:
        console.print(
            Panel.fit(
                content,
                title=title,
                border_style=style or "cyan",
                box=box.ROUNDED,
            )
        )
    else:
        # Plain fallback
        if title:
            console.print(f"[{title}] {_strip_markup(content)}")
        else:
            console.print(_strip_markup(content))


def dims_table(ds) -> None:
    """Render a tiny table of dataset dimensions and sizes."""
    if not USE_RICH:
        console.print("dims: " + ", ".join(f"{k}={v}" for k, v in ds.dims.items()))
        return
    tbl = Table(box=box.SIMPLE_HEAVY)
    tbl.add_column("Dim", style="bold")
    tbl.add_column("Size", justify="right")
    for k, v in ds.dims.items():
        tbl.add_row(str(k), str(v))
    console.print(tbl)


def module_status(name: str, status: str, detail: str = "") -> None:
    """Pretty module status line.

    status: one of "run", "skip", "info".
    """
    if status == "run":
        icon = "▶"
        color = "green"
    elif status == "skip":
        icon = "⏭"
        color = "yellow"
    else:
        icon = "ℹ"
        color = "cyan"
    if USE_RICH:
        msg = f"[bold]{icon} {name}[/]"
        suffix = f" [dim]{detail}[/]" if detail else ""
        console.print(f"[{color}]{msg}[/]{suffix}")
    else:
        suffix = f" ({detail})" if detail else ""
        console.print(f"{icon} {name}{suffix}")


def ensemble_panel(message: str, level: str = "info") -> None:
    """Show ensemble handling message with appropriate style."""
    style = {"info": "cyan", "warn": "yellow", "ok": "green"}.get(level, "cyan")
    title = {
        "info": "Ensemble",
        "warn": "Ensemble Warning",
        "ok": "Ensemble",
    }.get(level, "Ensemble")
    panel(message, title=title, style=style)


def timings_summary(entries: list[tuple[str, float]], total: float) -> None:
    """Print a summary of per-module timings and total.

    entries: list of (module_name, seconds)
    total: total runtime in seconds
    """
    if not entries:
        return
    # Keep display order as provided
    if USE_RICH:
        tbl = Table(title="Module durations", box=box.SIMPLE_HEAVY)
        tbl.add_column("Module", style="bold")
        tbl.add_column("Time (s)", justify="right")
        tbl.add_column("Percent", justify="right")
        for name, secs in entries:
            pct = (secs / total * 100.0) if total > 0 else 0.0
            tbl.add_row(str(name), f"{secs:,.2f}", f"{pct:,.1f}%")
        console.print(tbl)
        console.print(f"Total: [bold]{total:,.2f}[/] s")
    else:
        console.print("Module durations:")
        # Compute simple padding for alignment
        name_w = max(len(str(n)) for n, _ in entries)
        for name, secs in entries:
            pct = (secs / total * 100.0) if total > 0 else 0.0
            console.print(f"  - {str(name).ljust(name_w)}  {secs:7.2f}s  ({pct:5.1f}%)")
        console.print(f"Total: {total:,.2f}s")
