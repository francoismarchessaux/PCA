"""
Step 3 (Family A) — Implied PCA with reduced anchor tenors + stable daily mapping

Goal
----
Each day (as-of), we:
1) fit PCA on a rolling window of curve changes X (dates x tenors)  [Step 2 output]
2) choose a small set of "anchor tenors" S (e.g., |S| = k or user-fixed)
3) build a mapping that reconstructs full-curve moves from anchor moves:
       X_full ≈ X_anchor @ B
   where B is (|S| x N) and N is number of tenors in the curve.

Key desks constraints embedded
-----------------------------
- Robust covariance already handled upstream by Step 2 (Ledoit-Wolf recommended).
- Conditioning control: reject / penalize anchor sets that lead to unstable inverses.
- Stickiness: avoid changing anchors day-to-day unless there's a meaningful improvement.

This module is intentionally "production-shaped":
- deterministic outputs (sign convention + stable selection)
- guardrails (condition number threshold, improvement threshold)
- clear diagnostics per day (reconstruction score, cond number, anchor turnover)

Practical default
-----------------
- Choose k=4 anchors (e.g. [3Y, 5Y, 10Y, 20Y]) OR let algorithm pick
- Use rolling lookback "2Y"
- Use PCA computed on curve changes in bp
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal, Iterable, Union

import numpy as np
import pandas as pd

# Reuse Step 2
# from step2_pca_engine import RollingPCAModel, PCAFitResult

# ----------------------------
# Utilities
# ----------------------------

def _as_np(df: pd.DataFrame) -> np.ndarray:
    return df.to_numpy(dtype=float)

def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.nanmean(d * d)))

def _r2_like(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Returns 1 - ||y - yhat||^2 / ||y||^2 (Frobenius norm for matrices).
    This is a stable scalar score in [-inf, 1], higher is better.
    """
    num = float(np.linalg.norm(y_true - y_hat, ord="fro") ** 2)
    den = float(np.linalg.norm(y_true, ord="fro") ** 2)
    if den <= 1e-18:
        return np.nan
    return 1.0 - num / den

def _cond_number(A: np.ndarray) -> float:
    # Condition number in 2-norm
    try:
        return float(np.linalg.cond(A))
    except Exception:
        return float("inf")

def _pinv(A: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    return np.linalg.pinv(A, rcond=rcond)

def _intersect_columns(X: pd.DataFrame, cols: Iterable[str]) -> List[str]:
    cols = list(cols)
    return [c for c in cols if c in X.columns]

def _dropna_rows_for_cols(X: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return X.loc[:, cols].dropna(axis=0, how="any")


# ----------------------------
# Mapping math: anchors -> full curve
# ----------------------------

@dataclass(frozen=True)
class ImpliedPCAMapping:
    """
    Mapping that reconstructs full curve moves from anchor moves:
        X_full_hat = X_anchor @ B
    """
    asof: pd.Timestamp
    anchors: List[str]
    B: pd.DataFrame                 # (|S| x N) mapping anchors -> all tenors
    score: float                    # R2-like on calibration window
    rmse: float                     # RMSE on calibration window
    cond: float                     # condition number of L_S (PCA loadings restricted)
    pca_loadings: pd.DataFrame      # (N x k) loadings used at fit date
    explained_var_3: Tuple[float, float, float]  # EVR PC1-3 (or shorter)


def build_anchor_to_full_mapping(
    X_window: pd.DataFrame,
    loadings: pd.DataFrame,
    anchors: List[str],
    *,
    ridge_lambda: float = 0.0,
    rcond_pinv: float = 1e-12,
) -> Tuple[pd.DataFrame, float, float, float]:
    """
    Build B such that:
        X_full ≈ X_anchor @ B
    using the implied PCA formula:
        B = L_S @ pinv(L)^T      (or) more directly:
        B = L_S @ pinv(L)       depending on convention

    We'll use the clean derivation:
    - Let loadings L be (N x k), mapping factors -> tenors (columns are PCs).
    - Factors (scores) are F = X @ L  (if L orthonormal under covariance/corr basis).
    - We want a linear operator B so that X_hat = X_S @ B.

    A robust and practical approach that matches "implied PCA" spirit:
    1) Compute factor scores on window: F = X @ L  (T x k)
    2) Fit F from anchor moves: F ≈ X_S @ A   (A: |S| x k)  via least squares / ridge
    3) Then X_hat = F_hat @ L^T = (X_S @ A) @ L^T
       => B = A @ L^T   (|S| x N)

    This avoids directly inverting L_S and gives you an explicit conditioning control
    through the regression step. It is also more stable if you use ridge_lambda>0.

    Conditioning metric returned:
      cond(L_S) where L_S is loadings restricted to anchor rows (|S| x k).

    Returns:
      B (|S| x N) DataFrame
      score (R2-like)
      rmse
      cond(L_S)
    """
    anchors = list(anchors)
    # Align columns
    tenors = list(loadings.index)
    Xw = X_window.loc[:, tenors]
    Xw = Xw.dropna(axis=0, how="any")
    if Xw.shape[0] < 50:
        raise ValueError(f"Too few observations after NaN drop: {Xw.shape[0]}")

    # Extract anchor matrix
    anchors_in = _intersect_columns(Xw, anchors)
    if len(anchors_in) != len(anchors):
        missing = [a for a in anchors if a not in Xw.columns]
        raise ValueError(f"Anchors missing from X_window columns: {missing}")

    XS = Xw.loc[:, anchors]  # (T x |S|)
    L = loadings.loc[tenors, :]  # (N x k)
    k = L.shape[1]

    # Factor scores
    F = _as_np(Xw) @ _as_np(L)           # (T x k)

    # Fit A: minimize ||F - XS A||^2 + lambda ||A||^2
    XSm = _as_np(XS)                     # (T x |S|)
    # Ridge closed-form: A = (XS^T XS + λI)^-1 XS^T F
    XtX = XSm.T @ XSm                    # (|S| x |S|)
    if ridge_lambda > 0:
        XtX = XtX + ridge_lambda * np.eye(XtX.shape[0])
    A = np.linalg.solve(XtX, XSm.T @ F)  # (|S| x k)

    # Build B = A @ L^T  (|S| x N)
    Bm = A @ _as_np(L).T                 # (|S| x N)
    B = pd.DataFrame(Bm, index=anchors, columns=tenors)

    # Reconstruct
    X_hat = XSm @ Bm                     # (T x N)
    score = _r2_like(_as_np(Xw), X_hat)
    rmse = _rmse(_as_np(Xw), X_hat)

    # Conditioning of L_S (|S| x k)
    LS = _as_np(L.loc[anchors, :])       # anchors x k
    cond = _cond_number(LS)

    return B, score, rmse, cond


# ----------------------------
# Anchor selection (Greedy + conditioning guard + stickiness)
# ----------------------------

@dataclass
class AnchorSelectionConfig:
    k_anchors: int = 3
    candidate_tenors: Optional[List[str]] = None  # if None, use all columns
    ridge_lambda: float = 0.0                     # small ridge can stabilize mapping
    cond_max: float = 1e4                         # reject anchor sets above this condition number
    cond_penalty_lambda: float = 0.0              # optional: score -= lam * log(cond)
    min_improvement_to_switch: float = 0.002      # 0.2% R2 improvement required to change anchors
    max_anchor_changes_per_day: int = 0           # 0 means either keep all or switch all; >0 allows limited changes


def _penalized_score(score: float, cond: float, lam: float) -> float:
    if lam <= 0:
        return score
    if not np.isfinite(cond) or cond <= 0:
        return -np.inf
    return score - lam * float(np.log(cond))


def greedy_select_anchors(
    X_window: pd.DataFrame,
    loadings: pd.DataFrame,
    cfg: AnchorSelectionConfig,
) -> Tuple[List[str], ImpliedPCAMapping]:
    """
    Greedy forward selection:
    - Start empty
    - Add the tenor that maximizes penalized score at each step
    - Enforce conditioning constraint (cond(L_S) <= cfg.cond_max)

    Returns:
      best_anchors, mapping_info
    """
    tenors_all = list(loadings.index)
    candidates = cfg.candidate_tenors if cfg.candidate_tenors is not None else tenors_all
    candidates = _intersect_columns(pd.DataFrame(columns=tenors_all), candidates)  # keep only valid in loadings
    # Above line is a trick; we really want intersection with tenors_all:
    candidates = [c for c in candidates if c in tenors_all]
    if len(candidates) == 0:
        raise ValueError("No valid candidate tenors for anchor selection.")

    selected: List[str] = []
    best_map: Optional[ImpliedPCAMapping] = None

    # Basic explained variance diagnostics
    evr = getattr(loadings, "_explained_var_ratio", None)  # not used; we pass separately at call site

    for step in range(cfg.k_anchors):
        best_candidate = None
        best_candidate_map = None
        best_candidate_pscore = -np.inf

        remaining = [c for c in candidates if c not in selected]
        if len(remaining) == 0:
            raise ValueError("Ran out of candidates during selection.")

        for c in remaining:
            trial = selected + [c]
            try:
                B, score, rmse, cond = build_anchor_to_full_mapping(
                    X_window=X_window,
                    loadings=loadings,
                    anchors=trial,
                    ridge_lambda=cfg.ridge_lambda,
                )
            except Exception:
                continue

            if cond > cfg.cond_max:
                continue

            pscore = _penalized_score(score, cond, cfg.cond_penalty_lambda)
            if pscore > best_candidate_pscore:
                best_candidate_pscore = pscore
                best_candidate = c
                best_candidate_map = (B, score, rmse, cond)

        if best_candidate is None:
            raise ValueError(
                f"Could not find a feasible anchor at step {step+1}. "
                f"Try relaxing cond_max or adding ridge_lambda."
            )

        selected.append(best_candidate)

        B, score, rmse, cond = best_candidate_map  # type: ignore[misc]
        best_map = ImpliedPCAMapping(
            asof=pd.NaT,  # will be filled by caller
            anchors=list(selected),
            B=B,
            score=score,
            rmse=rmse,
            cond=cond,
            pca_loadings=loadings.copy(),
            explained_var_3=(np.nan, np.nan, np.nan),
        )

    assert best_map is not None
    return selected, best_map


def stickiness_update_anchors(
    prev_anchors: Optional[List[str]],
    prev_mapping: Optional[ImpliedPCAMapping],
    new_anchors: List[str],
    new_mapping: ImpliedPCAMapping,
    cfg: AnchorSelectionConfig,
) -> Tuple[List[str], ImpliedPCAMapping, bool]:
    """
    Apply a stickiness rule:
    - If we have previous anchors, keep them unless the new mapping score improves enough
      AND the new mapping is not worse in conditioning beyond tolerance.

    Returns:
      anchors_kept_or_new, mapping, switched (bool)
    """
    if prev_anchors is None or prev_mapping is None:
        return new_anchors, new_mapping, True

    # If previous anchors are not feasible today (e.g., missing tenors), we must switch.
    prev_set = list(prev_anchors)

    # Compare scores
    improvement = new_mapping.score - prev_mapping.score

    # Require meaningful improvement
    if improvement < cfg.min_improvement_to_switch:
        # Keep previous
        kept = prev_mapping
        kept = ImpliedPCAMapping(
            asof=new_mapping.asof,
            anchors=prev_set,
            B=prev_mapping.B,
            score=prev_mapping.score,
            rmse=prev_mapping.rmse,
            cond=prev_mapping.cond,
            pca_loadings=new_mapping.pca_loadings,  # update loadings to today's for downstream if needed
            explained_var_3=new_mapping.explained_var_3,
        )
        return prev_set, kept, False

    # Otherwise switch entirely (simple policy). You can implement "limited changes" later.
    return new_anchors, new_mapping, True


# ----------------------------
# Daily runner: PCA -> choose anchors -> mapping -> diagnostics
# ----------------------------

@dataclass
class DailyImpliedPCAConfig:
    # PCA settings
    n_components: int = 3
    pca_method: Literal["cov", "corr"] = "cov"
    pca_estimator: Literal["sample", "ledoitwolf"] = "ledoitwolf"
    lookback: Union[int, str] = "2Y"
    min_obs: int = 252

    # Anchor selection settings
    anchors_mode: Literal["auto", "fixed"] = "auto"
    fixed_anchors: Optional[List[str]] = None
    selection_cfg: AnchorSelectionConfig = AnchorSelectionConfig()

    # Output controls
    verbose: bool = False


def run_daily_implied_pca_family_A(
    X_changes: pd.DataFrame,
    cfg: DailyImpliedPCAConfig,
) -> Dict[pd.Timestamp, ImpliedPCAMapping]:
    """
    End-to-end daily engine for Family A:
    - rolling PCA
    - anchor selection (auto or fixed)
    - build mapping B: anchors -> full curve
    - apply stickiness to reduce day-to-day changes

    Returns:
      dict asof -> ImpliedPCAMapping
    """
    # Import locally to keep this file standalone if you paste pieces
    from sklearn.covariance import LedoitWolf

    def compute_cov_or_corr(Xw: pd.DataFrame) -> np.ndarray:
        Xw = Xw.dropna(axis=0, how="any")
        M = Xw.to_numpy(dtype=float)
        if cfg.pca_estimator == "ledoitwolf":
            cov = LedoitWolf().fit(M).covariance_
        else:
            cov = np.cov(M, rowvar=False, ddof=1)

        if cfg.pca_method == "cov":
            return cov
        # correlation conversion
        d = np.sqrt(np.clip(np.diag(cov), 1e-18, np.inf))
        invd = 1.0 / d
        corr = cov * invd[None, :] * invd[:, None]
        corr = 0.5 * (corr + corr.T)
        np.fill_diagonal(corr, 1.0)
        return corr

    def fit_pca_loadings(Xw: pd.DataFrame, tenors: List[str]) -> Tuple[pd.DataFrame, Tuple[float, float, float]]:
        Xw = Xw.loc[:, tenors].dropna(axis=0, how="any")
        S = compute_cov_or_corr(Xw)
        # symmetric eig
        eigvals, eigvecs = np.linalg.eigh(S)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # deterministic sign
        for j in range(eigvecs.shape[1]):
            col = eigvecs[:, j]
            k = int(np.argmax(np.abs(col)))
            if col[k] < 0:
                eigvecs[:, j] *= -1.0

        total = float(np.sum(eigvals))
        evr = eigvals / total if total > 0 else np.full_like(eigvals, np.nan)
        k = cfg.n_components
        pc_names = [f"PC{i+1}" for i in range(k)]
        L = pd.DataFrame(eigvecs[:, :k], index=tenors, columns=pc_names)

        ev3 = (float(evr[0]) if len(evr) > 0 else np.nan,
               float(evr[1]) if len(evr) > 1 else np.nan,
               float(evr[2]) if len(evr) > 2 else np.nan)
        return L, ev3

    X = X_changes.sort_index()
    if not isinstance(X.index, pd.DatetimeIndex):
        raise ValueError("X_changes must have a DatetimeIndex.")
    if X.shape[0] < cfg.min_obs:
        raise ValueError(f"Not enough rows in X_changes ({X.shape[0]}) for min_obs={cfg.min_obs}")

    tenors = list(X.columns)

    # Rolling slicing
    if isinstance(cfg.lookback, int):
        def window_slice(i: int) -> pd.DataFrame:
            start_i = max(0, i - cfg.lookback + 1)
            return X.iloc[start_i:i+1]
        asofs = [X.index[i] for i in range(cfg.min_obs, X.shape[0])]
    else:
        offset = pd.tseries.frequencies.to_offset(cfg.lookback)
        def window_slice(i: int) -> pd.DataFrame:
            asof = X.index[i]
            start = asof - offset
            return X.loc[(X.index > start) & (X.index <= asof)]
        asofs = [X.index[i] for i in range(cfg.min_obs, X.shape[0])]

    results: Dict[pd.Timestamp, ImpliedPCAMapping] = {}

    prev_anchors: Optional[List[str]] = None
    prev_mapping: Optional[ImpliedPCAMapping] = None

    for i, asof in enumerate(asofs):
        Xw = window_slice(cfg.min_obs + i) if isinstance(cfg.lookback, int) else window_slice(cfg.min_obs + i)
        Xw = Xw.dropna(axis=0, how="any")
        if Xw.shape[0] < cfg.min_obs:
            continue

        # Fit PCA loadings for this day
        L, ev3 = fit_pca_loadings(Xw, tenors)

        # Determine anchors
        if cfg.anchors_mode == "fixed":
            if not cfg.fixed_anchors:
                raise ValueError("anchors_mode='fixed' requires fixed_anchors.")
            anchors = cfg.fixed_anchors
            B, score, rmse, cond = build_anchor_to_full_mapping(
                X_window=Xw,
                loadings=L,
                anchors=anchors,
                ridge_lambda=cfg.selection_cfg.ridge_lambda,
            )
            new_map = ImpliedPCAMapping(
                asof=asof,
                anchors=list(anchors),
                B=B,
                score=score,
                rmse=rmse,
                cond=cond,
                pca_loadings=L.copy(),
                explained_var_3=ev3,
            )
            # optional stickiness doesn't apply if fixed (anchors don't change)
            results[asof] = new_map
            prev_anchors, prev_mapping = list(anchors), new_map
            if cfg.verbose:
                print(f"{asof.date()} FIXED anchors={anchors} score={score:.5f} rmse={rmse:.4f} cond={cond:.1f}")
            continue

        # AUTO: greedy select
        selection_cfg = cfg.selection_cfg
        # ensure candidate tenors are within curve tenors
        if selection_cfg.candidate_tenors is not None:
            selection_cfg.candidate_tenors = [t for t in selection_cfg.candidate_tenors if t in tenors]

        best_anchors, best_map = greedy_select_anchors(Xw, L, selection_cfg)
        best_map = ImpliedPCAMapping(
            asof=asof,
            anchors=best_anchors,
            B=best_map.B,
            score=best_map.score,
            rmse=best_map.rmse,
            cond=best_map.cond,
            pca_loadings=L.copy(),
            explained_var_3=ev3,
        )

        # Stickiness decision: compare with previous day's chosen anchors/mapping
        chosen_anchors, chosen_map, switched = stickiness_update_anchors(
            prev_anchors=prev_anchors,
            prev_mapping=prev_mapping,
            new_anchors=best_anchors,
            new_mapping=best_map,
            cfg=selection_cfg,
        )

        results[asof] = chosen_map
        prev_anchors, prev_mapping = chosen_anchors, chosen_map

        if cfg.verbose:
            tag = "SWITCH" if switched else "KEEP"
            print(
                f"{asof.date()} {tag} anchors={chosen_anchors} "
                f"EVR(1-3)={tuple(np.round(chosen_map.explained_var_3,4))} "
                f"score={chosen_map.score:.5f} rmse={chosen_map.rmse:.4f} cond={chosen_map.cond:.1f}"
            )

    return results


# ----------------------------
# How you use this (end of Step 3)
# ----------------------------

if __name__ == "__main__":
    """
    Expected workflow:

    Step 1:
      levels_bp, changes_bp, meta = build_market_matrices(cfg_step1)

    Step 3 (this file):
      cfgA = DailyImpliedPCAConfig(
          n_components=3,
          pca_method="cov",
          pca_estimator="ledoitwolf",
          lookback="2Y",
          min_obs=252,
          anchors_mode="auto", # or "fixed"
          fixed_anchors=["3Y","5Y","10Y","20Y"], # if fixed
          selection_cfg=AnchorSelectionConfig(
              k_anchors=4,                 # if auto
              ridge_lambda=1e-6,            # tiny ridge stabilizes A fit
              cond_max=1e4,                 # guardrail against explosion
              cond_penalty_lambda=0.0,      # or try 0.01
              min_improvement_to_switch=0.002,
          ),
          verbose=True,
      )

      mappings = run_daily_implied_pca_family_A(changes_bp, cfgA)

    The daily output:
      mappings[asof].anchors  -> selected tenors to hedge
      mappings[asof].B        -> mapping anchors -> full curve
      mappings[asof].score    -> reconstruction score (higher better)
      mappings[asof].rmse     -> reconstruction RMSE in bp units
      mappings[asof].cond     -> stability diagnostic
    """
    pass
