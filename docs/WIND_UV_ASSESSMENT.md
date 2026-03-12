# Wind U/V vs. Speed/Direction: Evaluation Impact Assessment

Evaluating U and V wind components separately meaningfully reduces interpretability for wind-specific
science conclusions.

---

## Impact by Module

### Where it hurts most

| Module | Impact | Reason |
|---|---|---|
| **WD KDE / Wasserstein** | 🔴 High | Marginal distributions of U and V are not meaningful for wind. The Wasserstein distance between two U distributions says almost nothing about whether the model captures e.g. the prevalence of southwesterly flow. Wind direction is a circular variable — its marginal decomposition into U/V is essentially meaningless statistically. |
| **Deterministic metrics (MAE/RMSE)** | 🟡 Medium | MAE on U and V separately is not rotation-invariant. A model that rotates the wind vector by 10° has zero MAE on speed but large MAE on both U and V. Misleading for convective or channelled flow regimes. |
| **Energy Spectra** | 🟡 Medium | Kinetic energy spectra should use `0.5*(U²+V²)`. Computing per-component spectra double-counts correlated variance and misses the cross-spectrum `U·V`. LSD numbers remain useful comparatively but are not physically interpretable as kinetic energy spectra. |
| **ETS (Extreme Threshold Statistics)** | 🟡 Medium | Thresholding `U > 10 m/s` is not the same as `wind_speed > 10 m/s`. A strong southerly (`V=15, U=0`) is missed entirely. |
| **Probabilistic CRPS** | 🟠 Low-Medium | CRPS per component is mathematically valid, but the energy score (multivariate CRPS) over the 2D wind vector would be more meaningful for ensemble calibration. |

### Where it barely matters

| Module | Impact | Reason |
|---|---|---|
| **Deterministic metrics (Pearson R, bias)** | 🟢 Low | Spatial correlation and bias on U and V are individually meaningful — a model with biased zonal flow shows up clearly in U bias. |
| **Vertical profiles** | 🟢 Low | NMAE profiles of U and V are still a valid diagnostic for jet structure, even evaluated separately. |
| **Maps** | 🟢 Low | Spatial bias maps of U and V are perfectly interpretable. |

---

## Implementation status and next steps

### 1. Derive wind speed
A `derived_variables` config block computes `sqrt(U²+V²)` lazily before
any module runs using the `add_derived_variables()` function in `data.py`:

```yaml
derived_variables:
  10m_wind_speed:
    kind: wind_speed
    u: 10m_u_component_of_wind
    v: 10m_v_component_of_wind
  wind_speed:
    kind: wind_speed
    u: u_component_of_wind
    v: v_component_of_wind
```

Both U and V must be included in `variables_2d` / `variables_3d` (inner-join
guard: skips with a warning if either component is missing from either dataset).

### 2. Energy spectra kinetic energy proxy
`_compute_spectra_pair` in `plots/energy_spectra.py` checks `_KE_PAIRS` and
automatically computes `KE = 0.5*(spec_U + spec_V)` whenever `wind_speed` or
`10m_wind_speed` is requested — no config change needed.

### 3. WD KDE circular statistics ⏳ Not yet implemented
Wind direction requires circular statistics for meaningful evaluation.
Two separate things are missing:

- `wind_direction` is **not available as a derived variable** — it was
  deliberately excluded from `_DERIVED_RECIPES` in `data.py` because the
  standard scalar metrics (MAE, RMSE, CRPS, histograms, Wasserstein, ETS)
  are all invalid for a circular variable without wrappers. Adding it would
  produce silently wrong numbers.
- The WD KDE module would need a dedicated circular-statistics rewrite
  (`atan2(V, U)` + von Mises / circular mean / circular Wasserstein) before
  wind direction evaluation can be added safely.
