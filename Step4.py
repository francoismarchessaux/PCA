"""
Step 4 — Hedge performance framework (PnL tracking) for Family A and Family B

What this step does
-------------------
Given:
  - X_changes: curve changes (dates x tenors), typically in bp
  - Sensitivities: either
        (A) sens_by_tenor: DataFrame (dates x tenors) of DV01-like sensitivities
            - units: PnL per 1bp move (e.g., USD per bp) OR any consistent unit
        OR
        (B) a single, time-invariant vector sens_vec (tenors,) if you only have one snapshot

And given hedge engines:
  - Family A mappings: dict[asof] -> ImpliedPCAMapping  (from Step 3 Family A)
  - Family B fits:     dict[asof] -> dict["PC1"/"PC2"/"PC3"] -> FactorProxyFit (from Step 3 Family B)

We compute, for each day t (usually t is "next day" after asof calibration):
  - Full PnL:         pnl_full[t] = sum_i sens[t,i] * X_changes[t,i]
  - Family A hedge PnL:
        We first compute optimal anchor hedge weights w_A (one number per anchor tenor) that best
        replicates the full PnL of that day using only anchor moves.
        In matrix form for one day:
            pnl_full = sens^T * x_full
            pnl_A    = w_A^T * x_anchor
        The best (least squares) w_A across the calibration window is:
            w_A = argmin_w || (X_anchor w) - pnl_full ||^2  (+ ridge)
        This produces a stable, desk-like "hedge ratio" on each anchor.
        Then we apply it out-of-sample day-by-day.

  - Family B hedge PnL:
        We build instrument moves Z (out/spread/fly) for the selected instruments at asof,
        then similarly fit hedge weights w_B on the calibration window to replicate pnl_full.

Key point
---------
Step 3 (A/B) gives you a way to *represent curve moves* with fewer instruments.
Step 4 translates that into *actual hedge ratios to track book PnL* and evaluates performance.

This matches your project definition:
compare Real PnL (full sensitivities x moves) vs PCA-hedge PnL (reduced instruments x moves).

Outputs
-------
- Daily PnL series for:
    pnl_full, pnl_hedge_A, pnl_hedge_B, residual_A, residual_B
- Summary stats:
    RMSE, MAE, correlation, tail metrics (95/99%), turnover (if you change hedges daily)

Dependencies
------------
numpy, pandas, sklearn (optional ridge)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Literal

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge


# ----------------------------
# Types: reuse from Step 3
# ----------------------------
# from step3_familyA_impliedpca import ImpliedPCAMapping, run_daily_implied_pca_family_A
# from step3_familyB_stability import InstrumentSpec, instruments_to_matrix, build_instrument_library, FactorProxyFit

# To keep this file self-contained, we only require the minimal fields.
@dataclass(frozen=True)
class ImpliedPCAMappingLite:
    asof: pd.Timestamp
    anchors: List[str]
    B: pd.DataFrame  # anchors x tenors


@dataclass(frozen=True)
class FactorProxyFitLite:
    asof: pd.Timestamp
    factor_name: str
    selected: List[str]
    beta: pd.Series
    intercept: float


# ----------------------------
# Core computations
# ----------------------------

def align_sensitivities(
    X_changes: pd.DataFrame,
    sens_by_tenor: Optional[pd.DataFrame] = None,
    sens_vec: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Produce sensitivities DataFrame aligned to X_changes index and columns.
    """
    tenors = list(X_changes.columns)
    idx = X_changes.index

    if sens_by_tenor is not None:
        S = sens_by_tenor.copy()
        if not isinstance(S.index, pd.DatetimeIndex):
            raise ValueError("sens_by_tenor must have a DatetimeIndex.")
        # align index: forward-fill sensitivities if provided less frequently (optional)
        S = S.reindex(idx).ffill()
        # align columns
        missing = [c for c in tenors if c not in S.columns]
        if missing:
            raise ValueError(f"sens_by_tenor missing columns: {missing}")
        return S.loc[:, tenors].astype(float)

    if sens_vec is not None:
        if isinstance(sens_vec, pd.DataFrame):
            raise ValueError("sens_vec should be a Series or 1D array, not DataFrame.")
        sv = pd.Series(sens_vec).copy()
        # ensure index is tenors
        if sv.index.dtype == object and all(t in sv.index for t in tenors):
            sv = sv.reindex(tenors)
        else:
            # assume same order as tenors
            if len(sv) != len(tenors):
                raise ValueError("sens_vec length does not match number of tenors.")
            sv.index = tenors
        S = pd.DataFrame(np.tile(sv.to_numpy(), (len(idx), 1)), index=idx, columns=tenors)
        return S.astype(float)

    raise ValueError("Provide either sens_by_tenor or sens_vec.")


def compute_full_pnl(
    X_changes: pd.DataFrame,
    sens: pd.DataFrame,
) -> pd.Series:
    """
    Full PnL: pnl[t] = sum_i sens[t,i] * X_changes[t,i]
    """
    Xc = X_changes[sens.columns].astype(float)
    Sc = sens.astype(float)
    pnl = (Xc * Sc).sum(axis=1)
    pnl.name = "pnl_full"
    return pnl


def fit_hedge_weights(
    X_inst: pd.DataFrame,
    pnl_full: pd.Series,
    *,
    ridge_alpha: float = 0.0,
) -> Tuple[pd.Series, float]:
    """
    Fit hedge weights w to replicate pnl_full on a calibration window:
      minimize ||X_inst w - pnl||^2 + alpha||w||^2

    Returns weights and intercept (we'll allow intercept; desks sometimes force 0).
    """
    df = pd.concat([X_inst, pnl_full.rename("pnl")], axis=1).dropna(axis=0, how="any")
    if df.shape[0] < 50:
        raise ValueError(f"Too few rows to fit hedge weights: {df.shape[0]}")
    Z = df[X_inst.columns].to_numpy(dtype=float)
    y = df["pnl"].to_numpy(dtype=float)

    if ridge_alpha > 0:
        model = Ridge(alpha=ridge_alpha, fit_intercept=True)
        model.fit(Z, y)
        w = pd.Series(model.coef_.reshape(-1), index=X_inst.columns)
        b = float(model.intercept_)
        return w, b

    # OLS with intercept
    Xd = np.column_stack([np.ones(len(Z)), Z])
    coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    b = float(coef[0])
    w = pd.Series(coef[1:].reshape(-1), index=X_inst.columns)
    return w, b


def apply_hedge_weights(
    X_inst: pd.DataFrame,
    w: pd.Series,
    intercept: float = 0.0,
) -> pd.Series:
    """
    Apply hedge weights to generate hedge PnL series:
      pnl_hedge[t] = intercept + sum_j w_j * X_inst[t, j]
    """
    cols = list(w.index)
    Xi = X_inst.reindex(columns=cols)
    pnl = intercept + (Xi * w).sum(axis=1)
    return pnl


# ----------------------------
# Family A: build anchor instrument moves
# ----------------------------

def build_anchor_moves(
    X_changes: pd.DataFrame,
    mapping: ImpliedPCAMappingLite,
) -> pd.DataFrame:
    """
    Anchor moves are just the curve changes at the anchor tenors (dates x |S|).
    """
    anchors = mapping.anchors
    missing = [a for a in anchors if a not in X_changes.columns]
    if missing:
        raise ValueError(f"Anchors missing from X_changes: {missing}")
    return X_changes.loc[:, anchors].copy()


# ----------------------------
# Family B: instrument library moves
# ----------------------------

def parse_instrument_name_to_spec(name: str) -> Tuple[List[str], List[float]]:
    """
    Parse instrument names produced in Step 3 Family B:
      OUT_<T>
      SPR_<short>_<long>  => r(long) - r(short)
      FLY_<s>_<m>_<l>     => r(s) - 2 r(m) + r(l)
    Returns legs, weights.
    """
    if name.startswith("OUT_"):
        t = name.replace("OUT_", "", 1)
        return [t], [1.0]
    if name.startswith("SPR_"):
        _, s, l = name.split("_", 2)
        return [s, l], [-1.0, 1.0]
    if name.startswith("FLY_"):
        _, s, m, l = name.split("_", 3)
        return [s, m, l], [1.0, -2.0, 1.0]
    raise ValueError(f"Unknown instrument name format: {name}")


def build_selected_instrument_moves(
    X_changes: pd.DataFrame,
    selected_instruments: List[str],
) -> pd.DataFrame:
    """
    Build Z_selected (dates x selected_instruments) directly from X_changes.
    """
    Z = pd.DataFrame(index=X_changes.index)
    for inst in selected_instruments:
        legs, weights = parse_instrument_name_to_spec(inst)
        missing = [t for t in legs if t not in X_changes.columns]
        if missing:
            raise ValueError(f"Instrument {inst} missing legs in X_changes: {missing}")
        vals = np.zeros(len(X_changes), dtype=float)
        for t, w in zip(legs, weights):
            vals += w * X_changes[t].to_numpy(dtype=float)
        Z[inst] = vals
    return Z


# ----------------------------
# Evaluation / stats
# ----------------------------

@dataclass
class HedgeStats:
    rmse: float
    mae: float
    corr: float
    p95_abs: float
    p99_abs: float


def compute_stats(residual: pd.Series) -> HedgeStats:
    r = residual.dropna().to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean(r * r))) if len(r) else np.nan
    mae = float(np.mean(np.abs(r))) if len(r) else np.nan
    corr = np.nan  # filled by caller when comparing series
    p95 = float(np.quantile(np.abs(r), 0.95)) if len(r) else np.nan
    p99 = float(np.quantile(np.abs(r), 0.99)) if len(r) else np.nan
    return HedgeStats(rmse=rmse, mae=mae, corr=corr, p95_abs=p95, p99_abs=p99)


def corr_safe(a: pd.Series, b: pd.Series) -> float:
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if df.shape[0] < 20:
        return np.nan
    return float(df["a"].corr(df["b"]))


# ----------------------------
# Main backtest runner
# ----------------------------

@dataclass
class BacktestConfig:
    # Calibration: hedges are fit on the same rolling window used for PCA/mapping
    lookback: Union[int, str] = "2Y"
    min_obs: int = 252

    # Hedge fit
    ridge_alpha: float = 1e-6
    fit_intercept: bool = True  # currently always fits intercept in ridge/OLS

    # Apply: choose whether hedge fitted at asof is applied on:
    #   - next day only ("one_step_ahead"), or
    #   - all days until next calibration ("hold_until_next") (not implemented here)
    apply_mode: Literal["one_step_ahead"] = "one_step_ahead"


def run_backtest_family_A(
    X_changes: pd.DataFrame,
    sens: pd.DataFrame,
    mappings_A: Dict[pd.Timestamp, ImpliedPCAMappingLite],
    cfg: BacktestConfig,
) -> Dict[str, pd.Series]:
    """
    Family A backtest:
      - For each asof, fit w_A on window to replicate pnl_full using anchor moves
      - Apply to next day pnl

    Returns dict of series: pnl_full, pnl_A, resid_A
    """
    X = X_changes.sort_index()
    S = sens.reindex(X.index).ffill()
    pnl_full = compute_full_pnl(X, S)

    # Determine asof dates we can use (need next day)
    asofs = sorted([d for d in mappings_A.keys() if d in X.index])
    asofs = [d for d in asofs if X.index.get_loc(d) < len(X.index) - 1]

    pnl_A = pd.Series(index=X.index, dtype=float, name="pnl_A")

    # Rolling slicing helper
    if isinstance(cfg.lookback, int):
        def window_for_asof(asof: pd.Timestamp) -> pd.DataFrame:
            i = X.index.get_loc(asof)
            start_i = max(0, i - cfg.lookback + 1)
            return X.iloc[start_i:i+1]
    else:
        offset = pd.tseries.frequencies.to_offset(cfg.lookback)
        def window_for_asof(asof: pd.Timestamp) -> pd.DataFrame:
            start = asof - offset
            return X.loc[(X.index > start) & (X.index <= asof)]

    for asof in asofs:
        mapping = mappings_A[asof]
        Xw = window_for_asof(asof).dropna(axis=0, how="any")
        if Xw.shape[0] < cfg.min_obs:
            continue

        # Calibration target and regressors on window
        pnlw = pnl_full.loc[Xw.index]
        X_anchor_w = build_anchor_moves(Xw, mapping)

        # Fit hedge weights
        w, b = fit_hedge_weights(X_anchor_w, pnlw, ridge_alpha=cfg.ridge_alpha)

        # Apply one-step-ahead
        t_next = X.index[X.index.get_loc(asof) + 1]
        X_anchor_next = build_anchor_moves(X.loc[[t_next]], mapping)
        pnl_A.loc[t_next] = float(b + (X_anchor_next.iloc[0] * w).sum())

    resid_A = pnl_full - pnl_A
    resid_A.name = "resid_A"
    return {"pnl_full": pnl_full, "pnl_A": pnl_A, "resid_A": resid_A}


def run_backtest_family_B(
    X_changes: pd.DataFrame,
    sens: pd.DataFrame,
    fits_B: Dict[pd.Timestamp, Dict[str, FactorProxyFitLite]],
    cfg: BacktestConfig,
) -> Dict[str, pd.Series]:
    """
    Family B backtest:
      - For each asof, build the selected instruments across PCs (union set)
      - Fit hedge weights w_B on window to replicate pnl_full using those instrument moves
      - Apply to next day
    """
    X = X_changes.sort_index()
    S = sens.reindex(X.index).ffill()
    pnl_full = compute_full_pnl(X, S)

    asofs = sorted([d for d in fits_B.keys() if d in X.index])
    asofs = [d for d in asofs if X.index.get_loc(d) < len(X.index) - 1]

    pnl_B = pd.Series(index=X.index, dtype=float, name="pnl_B")

    # Rolling slicing helper
    if isinstance(cfg.lookback, int):
        def window_for_asof(asof: pd.Timestamp) -> pd.DataFrame:
            i = X.index.get_loc(asof)
            start_i = max(0, i - cfg.lookback + 1)
            return X.iloc[start_i:i+1]
    else:
        offset = pd.tseries.frequencies.to_offset(cfg.lookback)
        def window_for_asof(asof: pd.Timestamp) -> pd.DataFrame:
            start = asof - offset
            return X.loc[(X.index > start) & (X.index <= asof)]

    for asof in asofs:
        day_fits = fits_B[asof]
        # Union of selected instruments across PCs
        selected = []
        for pc, fit in day_fits.items():
            selected.extend(list(fit.selected))
        selected = sorted(list(set(selected)))
        if len(selected) == 0:
            continue

        Xw = window_for_asof(asof).dropna(axis=0, how="any")
        if Xw.shape[0] < cfg.min_obs:
            continue

        pnlw = pnl_full.loc[Xw.index]
        Z_w = build_selected_instrument_moves(Xw, selected)

        # Fit hedge weights on instruments (this is the actual hedging layer)
        w, b = fit_hedge_weights(Z_w, pnlw, ridge_alpha=cfg.ridge_alpha)

        # Apply one-step-ahead
        t_next = X.index[X.index.get_loc(asof) + 1]
        Z_next = build_selected_instrument_moves(X.loc[[t_next]], selected)
        pnl_B.loc[t_next] = float(b + (Z_next.iloc[0] * w).sum())

    resid_B = pnl_full - pnl_B
    resid_B.name = "resid_B"
    return {"pnl_full": pnl_full, "pnl_B": pnl_B, "resid_B": resid_B}


def summarize_backtest(
    pnl_full: pd.Series,
    pnl_hedge: pd.Series,
    resid: pd.Series,
) -> Dict[str, Union[float, HedgeStats]]:
    """
    Return summary metrics for hedge performance.
    """
    # Tracking quality
    c = corr_safe(pnl_full, pnl_hedge)

    stats = compute_stats(resid)
    stats = HedgeStats(
        rmse=stats.rmse,
        mae=stats.mae,
        corr=c,
        p95_abs=stats.p95_abs,
        p99_abs=stats.p99_abs,
    )

    # Additional desk-friendly metrics
    df = pd.concat([pnl_full.rename("full"), pnl_hedge.rename("hedge"), resid.rename("resid")], axis=1).dropna()
    if df.shape[0] > 0:
        gap_mean = float(df["resid"].mean())
        gap_std = float(df["resid"].std(ddof=1)) if df.shape[0] > 1 else np.nan
    else:
        gap_mean, gap_std = np.nan, np.nan

    return {
        "corr_full_vs_hedge": stats.corr,
        "rmse_residual": stats.rmse,
        "mae_residual": stats.mae,
        "p95_abs_residual": stats.p95_abs,
        "p99_abs_residual": stats.p99_abs,
        "mean_residual": gap_mean,
        "std_residual": gap_std,
        "n_days": int(df.shape[0]),
    }


# ----------------------------
# Example usage (glue)
# ----------------------------

if __name__ == "__main__":
    """
    You will plug this in after Step 3.

    Inputs you already have / will have:
      - changes_bp: DataFrame [dates x tenors] in bp (Step 1)
      - mappings_A: dict[asof] -> ImpliedPCAMapping (Step 3 Family A)
      - fits_B: dict[asof] -> dict[PC] -> FactorProxyFit (Step 3 Family B)
      - sensitivities:
          * sens_by_tenor: DataFrame [dates x tenors] DV01 per bp
            OR
          * sens_vec: Series [tenors] if constant

    Example:
      sens = align_sensitivities(changes_bp, sens_by_tenor=sens_df)

      bt_cfg = BacktestConfig(lookback="2Y", min_obs=252, ridge_alpha=1e-6)

      outA = run_backtest_family_A(changes_bp, sens, mappings_A, bt_cfg)
      statsA = summarize_backtest(outA["pnl_full"], outA["pnl_A"], outA["resid_A"])
      print("Family A stats:", statsA)

      outB = run_backtest_family_B(changes_bp, sens, fits_B, bt_cfg)
      statsB = summarize_backtest(outB["pnl_full"], outB["pnl_B"], outB["resid_B"])
      print("Family B stats:", statsB)
    """
    pass
