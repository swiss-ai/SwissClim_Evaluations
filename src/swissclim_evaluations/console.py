"""Pretty console helpers for SwissClim Evaluations.

Uses Rich for modern, colorized output with emojis and clean layouts.
Falls back to plain prints if Rich isn't available.
"""

from __future__ import annotations

import os
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
        print(*objects)


_COLOR_PREF = os.environ.get("SWISSCLIM_COLOR", "always").lower()
# always|auto|never
if _HAS_RICH and _COLOR_PREF != "never":
    force = _COLOR_PREF == "always"
    console = Console(
        highlight=False,
        force_terminal=True if force else None,
        color_system="truecolor",
        soft_wrap=True,
    )
else:
    console = _PlainConsole()


def header(title: str) -> None:
    if _HAS_RICH:
        console.print(Rule(Text.from_markup(f"[bold cyan]🚀 {title}[/]")))
    else:
        console.print(f"===== {title} =====")


def section(title: str) -> None:
    if _HAS_RICH:
        console.print(Rule(Text.from_markup(f"[bold]{title}[/]")))
    else:
        console.print(f"--- {title} ---")


def info(msg: str) -> None:
    prefix = "ℹ️  "
    if _HAS_RICH:
        console.print(f"[cyan]{prefix}[bold]{msg}[/]")
    else:
        console.print(prefix + msg)


def success(msg: str) -> None:
    prefix = "✅ "
    if _HAS_RICH:
        console.print(f"[green]{prefix}[bold]{msg}[/]")
    else:
        console.print(prefix + msg)


def warn(msg: str) -> None:
    prefix = "⚠️  "
    if _HAS_RICH:
        console.print(f"[yellow]{prefix}[bold]{msg}[/]")
    else:
        console.print(prefix + msg)


def error(msg: str) -> None:
    prefix = "❌ "
    if _HAS_RICH:
        console.print(f"[red]{prefix}[bold]{msg}[/]")
    else:
        console.print(prefix + msg)


def panel(content: str, title: str | None = None, style: str = "") -> None:
    if _HAS_RICH:
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
            console.print(f"[{title}] {content}")
        else:
            console.print(content)


def dims_table(ds) -> None:
    """Render a tiny table of dataset dimensions and sizes."""
    if not _HAS_RICH:
        console.print(
            "dims: " + ", ".join(f"{k}={v}" for k, v in ds.dims.items())
        )
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
        msg = f"[bold]▶ {name}[/]"
        color = "green"
    elif status == "skip":
        msg = f"[bold]⏭ {name}[/]"
        color = "yellow"
    else:
        msg = f"[bold]ℹ {name}[/]"
        color = "cyan"
    if _HAS_RICH:
        suffix = f" [dim]{detail}[/]" if detail else ""
        console.print(f"[{color}]{msg}[/]{suffix}")
    else:
        console.print(f"{msg} {detail}")


def ensemble_panel(message: str, level: str = "info") -> None:
    """Show ensemble handling message with appropriate style."""
    style = {"info": "cyan", "warn": "yellow", "ok": "green"}.get(level, "cyan")
    title = {
        "info": "Ensemble",
        "warn": "Ensemble Warning",
        "ok": "Ensemble",
    }.get(level, "Ensemble")
    panel(message, title=title, style=style)