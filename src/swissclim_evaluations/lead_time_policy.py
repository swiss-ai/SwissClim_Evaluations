"""Lead time selection policy (simplified).

Simplified multi-lead evaluation logic used across all plotting & metrics.

Modes
-----
first   : Keep only the first (t+0) lead (single-lead evaluation).
full    : Keep all leads (subject to optional max_hour cap).
subset  : Keep an explicit list of hours (subset_hours).
stride  : Keep every Nth hour (stride_hours, optionally capped by max_hour).

Core Fields
-----------
subset_hours     : Explicit hours to retain (subset mode).
stride_hours     : Interval for stride selection (stride mode).
max_hour         : Inclusive upper bound (applies to all modes except 'first').
chunk_size       : Reserved for future adaptive chunking optimisations.
store_full_fields: Hint allowing modules to persist full per-lead arrays.

Helper Functions
----------------
parse_lead_time_policy      : Parse dict config → LeadTimePolicy.
apply_lead_time_selection   : Apply selection logic to an xarray Dataset.

"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import xarray as xr


@dataclass
class LeadTimePolicy:
    mode: str = "first"  # first | full | subset | stride
    subset_hours: list[int] | None = None
    stride_hours: int | None = None
    max_hour: int | None = None  # hard cap (inclusive) on lead hours retained
    chunk_size: int = 8
    panel_selection: str = "first"  # first | evenly_spaced | specific
    panel_specific_hours: list[int] | None = None
    max_panels: int = 4
    store_full_fields: bool = False

    def as_dict(self) -> dict[str, Any]:  # for serialization
        d = asdict(self)
        return d

    def select_panel_hours(self, available_hours: list[int]) -> list[int]:
        # Deprecated; now always returns full list (panel concept removed).
        return list(available_hours)

    @property
    def preserve_all_leads(self) -> bool:
        return self.mode != "first"


def parse_lead_time_policy(cfg: dict[str, Any] | None) -> LeadTimePolicy:
    # Default to 'first' (single-lead) if configuration is omitted, preserving
    # classic behavior.
    if cfg is None:
        return LeadTimePolicy(mode="first")
    mode = str(cfg.get("mode", "first")).lower()
    # Accept both flat 'subset_hours' and nested 'subset: { hours: [...] }'
    subset_val = cfg.get("subset")
    if isinstance(subset_val, dict):
        subset_hours_nested = subset_val.get("hours")
    elif isinstance(subset_val, list):
        subset_hours_nested = subset_val
    else:
        subset_hours_nested = None
    subset_hours = cfg.get("subset_hours", subset_hours_nested)

    # Accept both flat 'stride_hours' and nested 'stride: { hours: N }'
    stride_val = cfg.get("stride")
    if isinstance(stride_val, dict):
        stride_hours_nested = stride_val.get("hours")
    elif isinstance(stride_val, int):
        stride_hours_nested = stride_val
    else:
        stride_hours_nested = None
    stride_hours = cfg.get("stride_hours", stride_hours_nested)
    max_hour = cfg.get("max_hour")
    # 'bins' mode has been removed; ignore any provided 'bins' config if present.
    if mode == "bins":
        mode = "full"
    # Panel selection can be configured either flat or under a nested 'panel' block
    panel_cfg = cfg.get("panel") or {}
    panel_selection = str(cfg.get("panel_selection", panel_cfg.get("strategy", "first"))).lower()
    # prefer 'panel_specific_hours' flat, else nested; allow alias 'hours'
    panel_specific_hours = cfg.get(
        "panel_specific_hours", panel_cfg.get("panel_specific_hours", panel_cfg.get("hours"))
    )
    # prefer 'max_panels' flat, else nested 'count'
    max_panels = int(cfg.get("max_panels", panel_cfg.get("count", 4)))
    chunk_size = int(cfg.get("chunk_size", 8))
    store_full_fields = bool(cfg.get("store_full_fields", False))

    policy = LeadTimePolicy(
        mode=mode,
        subset_hours=subset_hours,
        stride_hours=stride_hours,
        max_hour=max_hour,
        chunk_size=chunk_size,
        panel_selection=panel_selection,
        panel_specific_hours=panel_specific_hours,
        max_panels=max_panels,
        store_full_fields=store_full_fields,
    )
    # Deprecated validation removed: panel_selection no longer enforced.
    return policy


def _lead_hours(ds: xr.Dataset) -> list[int]:
    if "lead_time" not in ds.dims:
        return []
    raw = ds["lead_time"].values  # timedelta64[ns]
    hours = (raw // np.timedelta64(1, "h")).astype(int)
    return list(map(int, hours.tolist()))


def apply_lead_time_selection(ds: xr.Dataset, policy: LeadTimePolicy) -> xr.Dataset:
    if "lead_time" not in ds.dims:
        return ds
    if policy.mode == "first":
        if ds.lead_time.size > 1:
            return ds.isel(lead_time=0, drop=False)
        return ds
    hours = _lead_hours(ds)
    if not hours:
        return ds
    if policy.mode == "full":
        # Apply max_hour cap if provided
        if policy.max_hour is not None:
            hours = _lead_hours(ds)
            idx = [i for i, h in enumerate(hours) if h <= int(policy.max_hour)]
            if idx:
                ds = ds.isel(lead_time=idx)
        return ds
    if policy.mode == "subset" and policy.subset_hours:
        targets = {int(h) for h in policy.subset_hours}
        if policy.max_hour is not None:
            targets = {h for h in targets if h <= int(policy.max_hour)}
        if not targets:
            raise ValueError(
                "LeadTimePolicy subset_hours produced empty selection after max_hour filtering."
            )
        idx = [i for i, h in enumerate(hours) if h in targets]
        if not idx:
            raise ValueError("LeadTimePolicy subset_hours produced empty selection.")
        return ds.isel(lead_time=idx)
    if policy.mode == "stride" and policy.stride_hours:
        stride = int(policy.stride_hours)
        if stride <= 0:
            return ds
        # choose indices where hour % stride == 0
        idx = [i for i, h in enumerate(hours) if h % stride == 0]
        if policy.max_hour is not None:
            idx = [i for i in idx if hours[i] <= int(policy.max_hour)]
        # STRICT: Do not silently fall back; if nothing matches, fail early.
        if not idx:
            msg = (
                "LeadTimePolicy stride="
                f"{stride} produced empty selection for hours {hours} (after max_hour filter)."
            )
            raise ValueError(msg)
        return ds.isel(lead_time=idx)
    return ds
