"""Lead time selection & panel policy.

This module defines the data structures and utilities governing multi-lead
evaluation. The central :class:`LeadTimePolicy` encapsulates how lead_time
coordinates are reduced or grouped prior to downstream metric computation
and plotting.

Modes
-----
first   : Keep only the first (t+0) lead (single-lead evaluation mode /
          backward compatible default).
full    : Keep all leads (subject to optional max_hour cap).
subset  : Keep an explicit list of hours (subset_hours).
stride  : Keep every Nth hour (stride_hours, optionally capped by max_hour).
bins    : Keep all leads but allow later aggregation into user-defined bins.

Key Fields
----------
subset_hours     : List[int] of absolute forecast hours to retain (subset).
stride_hours     : Interval in hours for stride selection (stride mode).
max_hour         : Inclusive upper bound on hours considered (applied across
                   modes except 'first').
bins             : List[LeadTimeBin] definitions used only in bins mode.
panel_selection  : Strategy for choosing a subset of hours for expensive
                   multi-panel plots (first | evenly_spaced | specific).
panel_specific_hours : Optional explicit list for 'specific' panel strategy.
max_panels       : How many panel hours to return.
chunk_size       : Reserved for future adaptive chunking of lead dimension.
store_full_fields: Hint allowing modules to persist full per-lead arrays.

Helper Functions
----------------
parse_lead_time_policy : Parse dict config → LeadTimePolicy.
apply_lead_time_selection : Apply selection logic to an xarray Dataset.
aggregate_bins          : Aggregate selected leads into named bins (mean).

The selection logic is conservative: if filtering would remove all leads it
raises a ValueError (subset) or falls back to the first lead (stride).

"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import xarray as xr


@dataclass
class LeadTimeBin:
    start: int  # inclusive hours
    end: int  # exclusive hours
    label: str

    def contains(self, h: int) -> bool:
        return self.start <= h < self.end


@dataclass
class LeadTimePolicy:
    mode: str = "first"  # first | full | subset | stride | bins
    subset_hours: list[int] | None = None
    stride_hours: int | None = None
    bins: list[LeadTimeBin] | None = None
    max_hour: int | None = None  # hard cap (inclusive) on lead hours retained
    chunk_size: int = 8
    panel_selection: str = "first"  # first | evenly_spaced | specific
    panel_specific_hours: list[int] | None = None
    max_panels: int = 4
    store_full_fields: bool = False

    def as_dict(self) -> dict[str, Any]:  # for serialization
        d = asdict(self)
        if self.bins is not None:
            d["bins"] = [asdict(b) for b in self.bins]
        return d

    def select_panel_hours(self, available_hours: list[int]) -> list[int]:
        if not available_hours:
            return []
        if self.panel_selection == "first":
            return available_hours[: self.max_panels]
        if self.panel_selection == "evenly_spaced":
            if self.max_panels >= len(available_hours):
                return available_hours
            # linspace over indices
            idx = np.linspace(0, len(available_hours) - 1, self.max_panels)
            return sorted({available_hours[int(round(i))] for i in idx})
        if self.panel_selection == "specific" and self.panel_specific_hours:
            chosen = []
            avail_set = set(available_hours)
            for h in self.panel_specific_hours:
                # pick nearest available hour
                if h in avail_set:
                    chosen.append(h)
                else:
                    # nearest by absolute diff
                    nearest = min(available_hours, key=lambda x: abs(x - h))
                    if nearest not in chosen:
                        chosen.append(nearest)
            return chosen[: self.max_panels]
        return available_hours[: self.max_panels]


def parse_lead_time_policy(cfg: dict[str, Any] | None) -> LeadTimePolicy:
    if cfg is None:
        return LeadTimePolicy()  # defaults to first
    mode = str(cfg.get("mode", "first")).lower()
    subset_hours = cfg.get("subset_hours")
    stride_hours = cfg.get("stride_hours")
    max_hour = cfg.get("max_hour")
    bins_cfg = cfg.get("bins") or []
    bins: list[LeadTimeBin] | None = None
    if mode == "bins" and bins_cfg:
        bins = []
        for b in bins_cfg:
            try:
                bins.append(
                    LeadTimeBin(
                        int(b["start"]),
                        int(b["end"]),
                        str(b.get("label", f"{b['start']}-{b['end']}h")),
                    )
                )
            except Exception:
                continue
        if not bins:
            mode = "full"  # fallback
    panel_selection = str(cfg.get("panel_selection", "first")).lower()
    panel_specific_hours = cfg.get("panel_specific_hours")
    max_panels = int(cfg.get("max_panels", 4))
    chunk_size = int(cfg.get("chunk_size", 8))
    store_full_fields = bool(cfg.get("store_full_fields", False))

    return LeadTimePolicy(
        mode=mode,
        subset_hours=subset_hours,
        stride_hours=stride_hours,
        bins=bins,
        max_hour=max_hour,
        chunk_size=chunk_size,
        panel_selection=panel_selection,
        panel_specific_hours=panel_specific_hours,
        max_panels=max_panels,
        store_full_fields=store_full_fields,
    )


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
        if not idx:
            idx = [0]
        return ds.isel(lead_time=idx)
    if policy.mode == "bins" and policy.bins:
        # For bins we may still apply max_hour truncation if provided, then keep
        # all for bin aggregation
        if policy.max_hour is not None:
            hours = _lead_hours(ds)
            idx = [i for i, h in enumerate(hours) if h <= int(policy.max_hour)]
            if idx and len(idx) < ds.lead_time.size:
                ds = ds.isel(lead_time=idx)
        return ds
    return ds


def aggregate_bins(ds: xr.Dataset, policy: LeadTimePolicy) -> list[tuple[str, xr.Dataset]]:
    if policy.mode != "bins" or not policy.bins or "lead_time" not in ds.dims:
        return []
    hours = _lead_hours(ds)
    out: list[tuple[str, xr.Dataset]] = []
    for b in policy.bins:
        mask_idx = [i for i, h in enumerate(hours) if b.contains(h)]
        if not mask_idx:
            continue
        sel = ds.isel(lead_time=mask_idx)
        # Aggregate mean over lead_time for representation
        agg = sel.mean(dim="lead_time", keep_attrs=True)
        # Add metadata coordinate for traceability
        agg = agg.assign_coords({"lead_time_bin": ("lead_time_bin", [b.label])}).expand_dims(
            "lead_time_bin"
        )
        out.append((b.label, agg))
    return out
