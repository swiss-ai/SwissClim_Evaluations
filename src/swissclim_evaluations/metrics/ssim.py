from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from skimage.metrics import structural_similarity as ssim

from .. import console as c
from ..dask_utils import compute_jobs
from ..helpers import (
    build_output_filename,
    ensemble_mode_to_token,
    resolve_ensemble_mode,
)


def calculate_ssim(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
    gaussian_weights: bool = True,
    use_sample_covariance: bool = True,
    preserve_dims: list[str] | None = None,
) -> pd.DataFrame:
    """Calculate SSIM for each variable.

    SSIM is applied per 2-D spatial slice (lat/lon) and the resulting per-slice
    scores are averaged over all dimensions that are *not* in ``preserve_dims``.

    Args:
        ds_target: Target/reference dataset.
        ds_prediction: Prediction dataset.
        sigma: Standard deviation for the Gaussian kernel.
        K1: Algorithm constant (luminance).
        K2: Algorithm constant (contrast).
        gaussian_weights: If ``True``, use Gaussian weighting.
        use_sample_covariance: If ``True``, normalize covariance with N-1.
        preserve_dims: Dimensions to *keep* in the output instead of averaging
            over them.  For example ``["level"]`` returns one row per
            (variable, level), ``["lead_time"]`` one row per (variable,
            lead_time).  ``None`` (default) collapses all non-spatial dims to
            a single scalar per variable.

    Returns:
        * When ``preserve_dims`` is ``None``: DataFrame indexed by variable
          with a single ``SSIM`` column.
        * Otherwise: long-form DataFrame with ``variable``, one column per
          preserved dimension, and an ``SSIM`` column.
    """
    variables = list(ds_target.data_vars)

    # ── Pass 1: collect lazy range jobs ──────────────────────────────────────
    range_jobs: list[dict] = []

    for var in variables:
        if var not in ds_prediction:
            continue

        da_target = ds_target[var]
        spatial_dims = [d for d in da_target.dims if d in {"latitude", "longitude", "lat", "lon"}]
        if len(spatial_dims) != 2:
            continue

        range_jobs.append(
            {
                "t_min": da_target.min(skipna=True),
                "t_max": da_target.max(skipna=True),
                "p_min": ds_prediction[var].min(skipna=True),
                "p_max": ds_prediction[var].max(skipna=True),
                "var": var,
            }
        )

    if not range_jobs:
        return pd.DataFrame()

    compute_jobs(
        range_jobs,
        key_map={
            "t_min": "t_min_res",
            "t_max": "t_max_res",
            "p_min": "p_min_res",
            "p_max": "p_max_res",
        },
        desc="SSIM: computing data ranges",
    )

    # ── Pass 2: build lazy SSIM jobs ──────────────────────────────────────────
    ssim_jobs: list[dict] = []

    for job in range_jobs:
        var = job["var"]
        data_range = max(float(job["t_max_res"]), float(job["p_max_res"])) - min(
            float(job["t_min_res"]), float(job["p_min_res"])
        )
        if data_range == 0:
            data_range = 1.0

        da_target = ds_target[var]
        da_prediction = ds_prediction[var]
        spatial_dims = [d for d in da_target.dims if d in {"latitude", "longitude", "lat", "lon"}]

        def _ssim_wrapper(t, p, data_range=data_range):
            return ssim(
                t,
                p,
                data_range=data_range,
                gaussian_weights=gaussian_weights,
                sigma=sigma,
                use_sample_covariance=use_sample_covariance,
                K1=K1,
                K2=K2,
            )

        # ssim_da has all non-spatial dims (time, level, lead_time, …)
        ssim_da = xr.apply_ufunc(
            _ssim_wrapper,
            da_target,
            da_prediction,
            input_core_dims=[spatial_dims, spatial_dims],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        # Determine which dims to keep vs reduce
        # Preserve dims are those both requested AND present in ssim_da, in
        # the order they appear in ssim_da so array shape stays predictable.
        _preserve = [d for d in ssim_da.dims if d in (preserve_dims or [])]
        _reduce = [d for d in ssim_da.dims if d not in _preserve]

        # Eagerly store coordinate values for preserved dims — these are tiny
        # index arrays; they won't survive dask.compute() on the DataArray.
        _coords: dict[str, np.ndarray] = {}
        for d in _preserve:
            if d in ssim_da.coords:
                _coords[d] = np.asarray(ssim_da[d].values)

        result_lazy = ssim_da.mean(dim=_reduce, skipna=True) if _reduce else ssim_da

        ssim_jobs.append(
            {
                "ssim_mean": result_lazy,
                "var": var,
                "_preserve": _preserve,
                "_coords": _coords,
            }
        )

    compute_jobs(
        ssim_jobs,
        key_map={"ssim_mean": "ssim_res"},
        desc="SSIM: computing metrics",
    )

    # ── Pass 3: build output DataFrame ───────────────────────────────────────
    rows: list[dict] = []
    for job in ssim_jobs:
        var = job["var"]
        result = job["ssim_res"]
        _preserve = job["_preserve"]
        _coords = job["_coords"]

        arr = np.asarray(result)

        if not _preserve or arr.ndim == 0:
            # Scalar (no preserved dims, or all had size 1)
            row: dict[str, Any] = {"variable": var, "SSIM": float(arr)}
            for d in _preserve:
                vals = _coords.get(d, np.array([]))
                row[d] = vals[0] if len(vals) else np.nan
            rows.append(row)
        elif arr.ndim == 1:
            d0 = _preserve[0]
            coord0 = _coords.get(d0, np.arange(arr.shape[0]))
            for i, val in enumerate(arr):
                rows.append({"variable": var, d0: coord0[i], "SSIM": float(val)})
        else:
            # 2-D (e.g. lead_time × level)
            d0, d1 = _preserve[0], _preserve[1]
            c0 = _coords.get(d0, np.arange(arr.shape[0]))
            c1 = _coords.get(d1, np.arange(arr.shape[1]))
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    rows.append({"variable": var, d0: c0[i], d1: c1[j], "SSIM": float(arr[i, j])})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if not preserve_dims:
        df = df.set_index("variable")
    return df


# ── Output helpers ────────────────────────────────────────────────────────────


def _lead_time_to_hours(val: Any) -> int:
    """Convert a lead_time coordinate value to integer hours."""
    try:
        return int(val / np.timedelta64(1, "h"))
    except Exception:
        try:
            return int(val)
        except Exception:
            return val


def run(
    ds_target: xr.Dataset,
    ds_prediction: xr.Dataset,
    out_root: Path,
    metrics_cfg: dict[str, Any] | None,
    ensemble_mode: str | None = None,
) -> None:
    """Compute and write SSIM metrics CSVs.

    Produces up to three CSV files per ensemble token:

    * ``ssim/ssim_ssim_<ens>.csv`` — overall mean SSIM per variable.
    * ``ssim/ssim_ssim_per_level_<ens>.csv`` — per-pressure-level SSIM for 3-D
      variables (written when ``report_per_level: true``, the default).
    * ``ssim/ssim_ssim_by_lead_<ens>.csv`` — per-lead-time SSIM for datasets
      with a ``lead_time`` dimension (written when ``report_per_lead: true``,
      the default).

    All three outputs are derived from a single lazy SSIM array per variable so
    the expensive per-slice computation runs exactly once per variable regardless
    of how many output granularities are requested.

    Args:
        ds_target: Target/reference dataset.
        ds_prediction: Prediction dataset.
        out_root: Root output directory. A ``ssim/`` sub-folder is created.
        metrics_cfg: Full ``metrics`` config dict (reads ``metrics_cfg["ssim"]``
            for ``sigma``, ``k1``, ``k2``, ``gaussian_weights``,
            ``use_sample_covariance``, ``report_per_level``,
            ``report_per_lead``).
        ensemble_mode: Resolved ensemble handling mode.
    """
    cfg: dict[str, Any] = (metrics_cfg or {}).get("ssim", {}) or {}

    sigma = float(cfg.get("sigma", 1.5))
    K1 = float(cfg.get("k1", cfg.get("K1", 0.01)))
    K2 = float(cfg.get("k2", cfg.get("K2", 0.03)))
    gaussian_weights = bool(cfg.get("gaussian_weights", True))
    use_sample_covariance = bool(cfg.get("use_sample_covariance", True))
    report_per_level = bool(cfg.get("report_per_level", True))
    report_per_lead = bool(cfg.get("report_per_lead", True))

    mode = resolve_ensemble_mode("ssim", ensemble_mode, ds_target, ds_prediction)
    out_dir = out_root / "ssim"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save_overall(df: pd.DataFrame, ens_token: str | None) -> None:
        if df.empty:
            return
        avg = df["SSIM"].mean()
        df_out = pd.concat([df, pd.DataFrame({"SSIM": [avg]}, index=["AVERAGE_SSIM"])])
        path = out_dir / build_output_filename(
            metric="ssim", qualifier="ssim", ensemble=ens_token, ext="csv"
        )
        df_out.to_csv(path, index_label="variable")
        c.info(f"[ssim] saved {path.name}")

    def _save_per_level(df: pd.DataFrame, ens_token: str | None) -> None:
        if df.empty or "level" not in df.columns:
            return
        df = df.copy()
        df["level"] = df["level"].astype(int)
        path = out_dir / build_output_filename(
            metric="ssim", qualifier="ssim_per_level", ensemble=ens_token, ext="csv"
        )
        df.to_csv(path, index=False)
        c.info(f"[ssim] saved {path.name}")

    def _save_by_lead(df: pd.DataFrame, ens_token: str | None) -> None:
        if df.empty or "lead_time" not in df.columns:
            return
        df = df.copy()
        df["lead_time_hours"] = df["lead_time"].apply(_lead_time_to_hours)
        df = df.drop(columns=["lead_time"])
        path = out_dir / build_output_filename(
            metric="ssim", qualifier="ssim_by_lead", ensemble=ens_token, ext="csv"
        )
        df.to_csv(path, index=False)
        c.info(f"[ssim] saved {path.name}")

    def _run_one(ds_t: xr.Dataset, ds_p: xr.Dataset, ens_token: str | None) -> None:
        """Compute all output granularities in two dask passes.

        Pass 1 – data ranges (one scalar per variable).
        Pass 2 – all reductions (overall, per-level, per-lead) derived from a
        single lazy SSIM DataArray per variable so the expensive per-spatial-
        slice computation runs exactly once.  All lazy arrays are submitted to
        dask.compute() together so dask can deduplicate the shared task graph.
        """
        variables = list(ds_t.data_vars)

        # ── Pass 1: data ranges ───────────────────────────────────────────────
        range_jobs: list[dict] = []
        for var in variables:
            if var not in ds_p:
                continue
            da_t = ds_t[var]
            spatial_dims = [d for d in da_t.dims if d in {"latitude", "longitude", "lat", "lon"}]
            if len(spatial_dims) != 2:
                continue
            range_jobs.append(
                {
                    "t_min": da_t.min(skipna=True),
                    "t_max": da_t.max(skipna=True),
                    "p_min": ds_p[var].min(skipna=True),
                    "p_max": ds_p[var].max(skipna=True),
                    "var": var,
                }
            )

        if not range_jobs:
            return

        compute_jobs(
            range_jobs,
            key_map={
                "t_min": "t_min_res",
                "t_max": "t_max_res",
                "p_min": "p_min_res",
                "p_max": "p_max_res",
            },
            desc="SSIM: data ranges",
        )

        # ── Pass 2: one ssim_da per variable → all lazy reductions ───────────
        # Every entry always has "overall".  "per_level" / "per_lead" are added
        # only when the corresponding dimension is present, so compute_jobs will
        # only schedule those arrays for variables that actually need them.
        batch: list[dict] = []
        for job in range_jobs:
            var = job["var"]
            data_range = max(float(job["t_max_res"]), float(job["p_max_res"])) - min(
                float(job["t_min_res"]), float(job["p_min_res"])
            )
            if data_range == 0:
                data_range = 1.0

            da_t = ds_t[var]
            da_p = ds_p[var]
            spatial_dims = [d for d in da_t.dims if d in {"latitude", "longitude", "lat", "lon"}]

            def _wrapper(t, p, dr=data_range):
                return ssim(
                    t,
                    p,
                    data_range=dr,
                    gaussian_weights=gaussian_weights,
                    sigma=sigma,
                    use_sample_covariance=use_sample_covariance,
                    K1=K1,
                    K2=K2,
                )

            # One lazy SSIM array for this variable; all reductions share this graph.
            ssim_da = xr.apply_ufunc(
                _wrapper,
                da_t,
                da_p,
                input_core_dims=[spatial_dims, spatial_dims],
                output_core_dims=[[]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )

            # Eagerly capture coordinates — they won't survive dask.compute().
            _coords: dict[str, np.ndarray] = {
                d: np.asarray(ssim_da[d].values) for d in ssim_da.dims if d in ssim_da.coords
            }

            entry: dict[str, Any] = {"var": var, "_coords": _coords}

            # Overall: reduce all non-spatial dims to a scalar.
            entry["overall"] = ssim_da.mean(skipna=True)

            # Per-level: reduce everything except level (same ssim_da).
            if report_per_level and "level" in ssim_da.dims:
                reduce = [d for d in ssim_da.dims if d != "level"]
                entry["per_level"] = ssim_da.mean(dim=reduce, skipna=True) if reduce else ssim_da

            # Per-lead: reduce everything except lead_time (same ssim_da).
            if report_per_lead and "lead_time" in ssim_da.dims:
                reduce = [d for d in ssim_da.dims if d != "lead_time"]
                entry["per_lead"] = ssim_da.mean(dim=reduce, skipna=True) if reduce else ssim_da

            batch.append(entry)

        # One dask.compute() for all reductions; dask deduplicates the shared
        # ssim_da task graph so each per-slice SSIM value is computed once.
        compute_jobs(
            batch,
            key_map={
                "overall": "overall_res",
                "per_level": "per_level_res",
                "per_lead": "per_lead_res",
            },
            desc="SSIM: computing metrics",
        )

        # ── Pass 3: build DataFrames and save ─────────────────────────────────
        rows_overall: list[dict] = []
        rows_level: list[dict] = []
        rows_lead: list[dict] = []

        for entry in batch:
            var = entry["var"]
            _coords = entry["_coords"]

            rows_overall.append({"variable": var, "SSIM": float(np.asarray(entry["overall_res"]))})

            if "per_level_res" in entry:
                arr = np.asarray(entry["per_level_res"])
                level_vals = _coords.get("level", np.arange(arr.size))
                for i, val in enumerate(arr.ravel()):
                    rows_level.append({"variable": var, "level": level_vals[i], "SSIM": float(val)})

            if "per_lead_res" in entry:
                arr = np.asarray(entry["per_lead_res"])
                lead_vals = _coords.get("lead_time", np.arange(arr.size))
                for i, val in enumerate(arr.ravel()):
                    rows_lead.append(
                        {"variable": var, "lead_time": lead_vals[i], "SSIM": float(val)}
                    )

        if rows_overall:
            df_overall = pd.DataFrame(rows_overall).set_index("variable")
            _save_overall(df_overall, ens_token)
        if rows_level:
            _save_per_level(pd.DataFrame(rows_level), ens_token)
        if rows_lead:
            _save_by_lead(pd.DataFrame(rows_lead), ens_token)

    # ── Ensemble dispatch ─────────────────────────────────────────────────────
    if "ensemble" in ds_prediction.dims and mode == "mean":
        ds_prediction = ds_prediction.mean(dim="ensemble")
        if "ensemble" in ds_target.dims:
            ds_target = ds_target.mean(dim="ensemble")

    if mode == "members" and "ensemble" in ds_prediction.dims:
        n_members = int(ds_prediction.sizes["ensemble"])
        for m in range(n_members):
            ds_p_mem = ds_prediction.isel(ensemble=m, drop=True)
            if "ensemble" in ds_target.dims:
                ens_idx = 0 if ds_target.sizes.get("ensemble", 0) == 1 else m
                ds_t_mem = ds_target.isel(ensemble=ens_idx, drop=True)
            else:
                ds_t_mem = ds_target
            _run_one(ds_t_mem, ds_p_mem, ensemble_mode_to_token(mode, member_index=m))

    elif mode == "pooled" and "ensemble" in ds_prediction.dims:
        non_ens_dims = [d for d in ds_prediction.dims if d != "ensemble"]
        if "init_time" in non_ens_dims:
            sample_dim = "init_time"
        elif "lead_time" in non_ens_dims:
            sample_dim = "lead_time"
        elif "time" in non_ens_dims:
            sample_dim = "time"
        elif non_ens_dims:
            sample_dim = non_ens_dims[0]
        else:
            raise ValueError(
                "Pooled ensemble mode requires at least one non-ensemble dimension to stack "
                f"over, but only found: {tuple(ds_prediction.dims)!r}"
            )
        ds_p_stacked = ds_prediction.stack(pooled_sample=("ensemble", sample_dim))
        ds_t_stacked = (
            ds_target.stack(pooled_sample=("ensemble", sample_dim))
            if "ensemble" in ds_target.dims
            else ds_target
        )
        _run_one(ds_t_stacked, ds_p_stacked, ensemble_mode_to_token(mode))

    else:
        _run_one(ds_target, ds_prediction, ensemble_mode_to_token(mode))
