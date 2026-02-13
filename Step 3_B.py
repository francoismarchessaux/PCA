"""
Step 3 (Family B) — Sparse proxy instruments for PCA factors with Stability Selection

Concept
-------
We have curve changes X (dates x tenors).
From Step 2, we have PCA loadings L (tenors x k) and factor scores F = X @ L (dates x k).
Now we want to hedge factors with a SMALL SET of tradable "proxy instruments" built from tenors.

Family B builds a candidate instrument library Z:
- Outrights: r(T)
- Spreads: r(T_long) - r(T_short)
- Flies: r(T_short) - 2*r(T_mid) + r(T_long)  (curvature proxy)

Then for each PC j:
- Fit F[:, j] ~ Z * beta_j (sparse regression)
- Use Stability Selection:
    * Repeatedly subsample dates
    * Fit ElasticNet on each subsample with a fixed alpha grid (or a single alpha)
    * Track selection frequency per instrument
- Select instruments with high selection probability
- Refit final regression on the full window using only selected instruments (ridge/OLS)

Outputs
-------
Daily for each as-of date:
- selected instruments for PC1/PC2/PC3
- mapping from instruments -> factor scores (A matrix)
- (optional) mapping from instruments -> curve moves via A @ L^T

This block is meant to be used either:
- as an interpretability layer (level/slope/curv tradable proxies), or
- as an alternative hedging engine.

Dependencies
------------
- numpy, pandas
- scikit-learn

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal, Iterable, Union

import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler


# ----------------------------
# Instrument library building
# ----------------------------

@dataclass(frozen=True)
class InstrumentSpec:
    name: str
    kind: Literal["outright", "spread", "fly"]
    legs: Tuple[str, ...]
    weights: Tuple[float, ...]  # same length as legs


def build_instrument_library(
    tenors: List[str],
    *,
    include_outrights: bool = True,
    include_spreads: bool = True,
    include_flies: bool = True,
    max_spreads: Optional[int] = None,
    max_flies: Optional[int] = None,
    min_tenor_gap: int = 1,
) -> List[InstrumentSpec]:
    """
    Build a candidate library from a tenor grid (ordered list).
    - min_tenor_gap: minimum index separation between legs to avoid trivial near-duplicates
    - max_spreads / max_flies: optional cap (useful if you have many tenors)

    Naming conventions:
      OUT_<T>
      SPR_<short>_<long>  meaning r(long) - r(short)
      FLY_<short>_<mid>_<long> meaning r(short) - 2 r(mid) + r(long)
    """
    tenors = list(tenors)
    n = len(tenors)
    lib: List[InstrumentSpec] = []

    if include_outrights:
        for t in tenors:
            lib.append(InstrumentSpec(
                name=f"OUT_{t}",
                kind="outright",
                legs=(t,),
                weights=(1.0,),
            ))

    if include_spreads:
        count = 0
        for i in range(n):
            for j in range(i + min_tenor_gap, n):
                short, long = tenors[i], tenors[j]
                lib.append(InstrumentSpec(
                    name=f"SPR_{short}_{long}",
                    kind="spread",
                    legs=(short, long),
                    weights=(-1.0, 1.0),  # r(long) - r(short)
                ))
                count += 1
                if max_spreads is not None and count >= max_spreads:
                    break
            if max_spreads is not None and count >= max_spreads:
                break

    if include_flies:
        count = 0
        for i in range(n):
            for j in range(i + min_tenor_gap, n):
                for k in range(j + min_tenor_gap, n):
                    s, m, l = tenors[i], tenors[j], tenors[k]
                    lib.append(InstrumentSpec(
                        name=f"FLY_{s}_{m}_{l}",
                        kind="fly",
                        legs=(s, m, l),
                        weights=(1.0, -2.0, 1.0),
                    ))
                    count += 1
                    if max_flies is not None and count >= max_flies:
                        break
                if max_flies is not None and count >= max_flies:
                    break
            if max_flies is not None and count >= max_flies:
                break

    return lib


def instruments_to_matrix(
    X: pd.DataFrame,
    instruments: List[InstrumentSpec],
) -> pd.DataFrame:
    """
    Build Z (dates x instruments) from curve matrix X (dates x tenors).

    X should be curve changes (bp) or returns, consistent with your PCA.
    """
    if not isinstance(X.index, pd.DatetimeIndex):
        raise ValueError("X must have a DatetimeIndex.")
    missing = {leg for inst in instruments for leg in inst.legs if leg not in X.columns}
    if missing:
        raise ValueError(f"Missing required tenors in X columns: {sorted(missing)}")

    Z = pd.DataFrame(index=X.index)
    for inst in instruments:
        vals = np.zeros(len(X), dtype=float)
        for leg, w in zip(inst.legs, inst.weights):
            vals += w * X[leg].to_numpy(dtype=float)
        Z[inst.name] = vals
    return Z


# ----------------------------
# Stability selection engine
# ----------------------------

@dataclass
class StabilitySelectionConfig:
    n_subsamples: int = 100
    subsample_frac: float = 0.6
    random_state: int = 123

    # ElasticNet parameters
    alpha_grid: Tuple[float, ...] = (0.0005, 0.001, 0.002, 0.005, 0.01)
    l1_ratio: float = 0.9  # 1.0 = Lasso, 0.0 = Ridge-like

    # Selection threshold
    select_prob_threshold: float = 0.6  # keep features selected in >=60% subsamples
    max_features: int = 3               # cap number of instruments per factor

    # Final refit
    final_refit: Literal["ridge", "ols"] = "ridge"
    ridge_alpha: float = 1e-6

    # Standardization
    standardize_Z: bool = True
    standardize_y: bool = True


@dataclass(frozen=True)
class FactorProxyFit:
    asof: pd.Timestamp
    factor_name: str
    selected: List[str]                 # instrument names
    selection_prob: pd.Series           # selection probabilities (all instruments)
    beta: pd.Series                     # final weights on selected instruments
    intercept: float
    r2_in_window: float
    rmse_in_window: float


def _r2_rmse(y: np.ndarray, yhat: np.ndarray) -> Tuple[float, float]:
    y = y.reshape(-1)
    yhat = yhat.reshape(-1)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else np.nan
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return r2, rmse


def stability_select_single_factor(
    Z: pd.DataFrame,
    y: pd.Series,
    cfg: StabilitySelectionConfig,
) -> Tuple[pd.Series, List[str]]:
    """
    Run stability selection for a single target y:
    returns selection probabilities and chosen feature names.
    """
    # Align and drop NaNs
    df = pd.concat([Z, y.rename("y")], axis=1).dropna(axis=0, how="any")
    if df.shape[0] < 100:
        raise ValueError(f"Too few rows for stability selection: {df.shape[0]}")
    Zm = df[Z.columns].to_numpy(dtype=float)
    ym = df["y"].to_numpy(dtype=float)

    rng = np.random.default_rng(cfg.random_state)
    n = Zm.shape[0]
    m = Zm.shape[1]

    # Optional standardization
    Z_scaler = StandardScaler(with_mean=True, with_std=True) if cfg.standardize_Z else None
    y_scaler = StandardScaler(with_mean=True, with_std=True) if cfg.standardize_y else None

    if Z_scaler is not None:
        Zm = Z_scaler.fit_transform(Zm)
    if y_scaler is not None:
        ym = y_scaler.fit_transform(ym.reshape(-1, 1)).reshape(-1)

    # Track selections
    counts = np.zeros(m, dtype=int)

    n_sub = max(1, int(cfg.subsample_frac * n))
    idx_all = np.arange(n)

    # For speed and determinism, we pick one alpha per subsample from the grid (cycle)
    alphas = list(cfg.alpha_grid)
    if len(alphas) == 0:
        raise ValueError("alpha_grid must not be empty.")
    for b in range(cfg.n_subsamples):
        idx = rng.choice(idx_all, size=n_sub, replace=False)
        Zb = Zm[idx, :]
        yb = ym[idx]

        alpha = alphas[b % len(alphas)]
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=cfg.l1_ratio,
            fit_intercept=True,
            max_iter=20000,
            tol=1e-6,
            random_state=cfg.random_state,
        )
        model.fit(Zb, yb)
        sel = np.abs(model.coef_) > 1e-12
        counts += sel.astype(int)

    probs = counts / float(cfg.n_subsamples)
    prob_s = pd.Series(probs, index=Z.columns).sort_values(ascending=False)

    # Select by threshold then cap at max_features
    chosen = prob_s[prob_s >= cfg.select_prob_threshold].index.tolist()
    if len(chosen) > cfg.max_features:
        chosen = chosen[: cfg.max_features]

    # If none pass threshold, take top-1 as fallback (prevents empty hedge)
    if len(chosen) == 0:
        chosen = [prob_s.index[0]]

    return prob_s, chosen


def final_refit(
    Z: pd.DataFrame,
    y: pd.Series,
    selected: List[str],
    cfg: StabilitySelectionConfig,
) -> Tuple[pd.Series, float, float, float]:
    """
    Refit y on selected Z columns using ridge or OLS, return beta, intercept, r2, rmse.
    """
    df = pd.concat([Z[selected], y.rename("y")], axis=1).dropna(axis=0, how="any")
    Zm = df[selected].to_numpy(dtype=float)
    ym = df["y"].to_numpy(dtype=float)

    # Standardize for fit stability, then unscale beta back to original units
    if cfg.standardize_Z:
        Z_scaler = StandardScaler(with_mean=True, with_std=True).fit(Zm)
        Zs = Z_scaler.transform(Zm)
    else:
        Z_scaler = None
        Zs = Zm

    if cfg.standardize_y:
        y_scaler = StandardScaler(with_mean=True, with_std=True).fit(ym.reshape(-1, 1))
        ys = y_scaler.transform(ym.reshape(-1, 1)).reshape(-1)
    else:
        y_scaler = None
        ys = ym

    if cfg.final_refit == "ridge":
        model = Ridge(alpha=cfg.ridge_alpha, fit_intercept=True)
        model.fit(Zs, ys)
        coef_s = model.coef_.reshape(-1)
        intercept_s = float(model.intercept_)
    else:
        # OLS via lstsq with intercept
        X_design = np.column_stack([np.ones(len(Zs)), Zs])
        w, *_ = np.linalg.lstsq(X_design, ys, rcond=None)
        intercept_s = float(w[0])
        coef_s = w[1:].reshape(-1)

    # Convert coefficients back to original scale:
    # y = y_mean + y_std * (intercept_s + sum coef_s * ( (Z - Z_mean)/Z_std ))
    # => y = (y_mean + y_std*intercept_s - y_std*sum coef_s*(Z_mean/Z_std)) + sum (y_std*coef_s/Z_std) * Z
    if Z_scaler is not None:
        Z_mean = Z_scaler.mean_
        Z_std = np.clip(Z_scaler.scale_, 1e-18, np.inf)
    else:
        Z_mean = np.zeros_like(coef_s)
        Z_std = np.ones_like(coef_s)

    if y_scaler is not None:
        y_mean = float(y_scaler.mean_[0])
        y_std = float(np.clip(y_scaler.scale_[0], 1e-18, np.inf))
    else:
        y_mean = 0.0
        y_std = 1.0

    beta = (y_std * coef_s) / Z_std
    intercept = y_mean + y_std * intercept_s - float(np.sum((y_std * coef_s) * (Z_mean / Z_std)))

    # In-window fit metrics (original scale)
    yhat = intercept + Zm @ beta
    r2, rmse = _r2_rmse(ym, yhat)

    return pd.Series(beta, index=selected), intercept, r2, rmse


# ----------------------------
# Full Family B daily runner
# ----------------------------

@dataclass
class FamilyBConfig:
    # PCA
    n_components: int = 3
    lookback: Union[int, str] = "2Y"
    min_obs: int = 252

    # Instruments
    include_outrights: bool = True
    include_spreads: bool = True
    include_flies: bool = True
    max_spreads: Optional[int] = 3000   # cap if needed
    max_flies: Optional[int] = 6000     # cap if needed
    min_tenor_gap: int = 1

    # Stability selection
    stab_cfg: StabilitySelectionConfig = StabilitySelectionConfig()

    # Simple "intended mapping": by default
    # - PC1: allow outrights
    # - PC2: allow spreads
    # - PC3: allow flies
    restrict_by_factor_kind: bool = True

    verbose: bool = False


def _fit_pca_loadings_and_scores(
    Xw: pd.DataFrame,
    n_components: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Tuple[float, float, float]]:
    """
    Minimal PCA (symmetric eig on covariance) for this step, to keep file standalone.
    You can replace with your Step 2 engine if you prefer.

    Returns:
      L (N x k), F (T x k), EVR(1-3)
    """
    Xw = Xw.dropna(axis=0, how="any")
    M = Xw.to_numpy(dtype=float)
    cov = np.cov(M, rowvar=False, ddof=1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # deterministic sign
    for j in range(eigvecs.shape[1]):
        col = eigvecs[:, j]
        k0 = int(np.argmax(np.abs(col)))
        if col[k0] < 0:
            eigvecs[:, j] *= -1.0

    total = float(np.sum(eigvals))
    evr = eigvals / total if total > 0 else np.full_like(eigvals, np.nan)
    ev3 = (float(evr[0]) if len(evr) > 0 else np.nan,
           float(evr[1]) if len(evr) > 1 else np.nan,
           float(evr[2]) if len(evr) > 2 else np.nan)

    pcs = [f"PC{i+1}" for i in range(n_components)]
    L = pd.DataFrame(eigvecs[:, :n_components], index=Xw.columns, columns=pcs)
    F = pd.DataFrame(M @ L.to_numpy(), index=Xw.index, columns=pcs)
    return L, F, ev3


def run_daily_family_B(
    X_changes: pd.DataFrame,
    cfg: FamilyBConfig,
) -> Dict[pd.Timestamp, Dict[str, FactorProxyFit]]:
    """
    For each asof date:
      - fit PCA on window
      - compute Z instrument library on window
      - for each PC (PC1..PCk):
          * stability selection -> selected instruments
          * final refit -> beta
    Returns:
      dict asof -> { "PC1": FactorProxyFit, ... }
    """
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

    # Prebuild full instrument library specs (names fixed)
    full_lib = build_instrument_library(
        tenors=tenors,
        include_outrights=cfg.include_outrights,
        include_spreads=cfg.include_spreads,
        include_flies=cfg.include_flies,
        max_spreads=cfg.max_spreads,
        max_flies=cfg.max_flies,
        min_tenor_gap=cfg.min_tenor_gap,
    )

    # Split libraries by type (for factor-kind restriction)
    lib_out = [i for i in full_lib if i.kind == "outright"]
    lib_spr = [i for i in full_lib if i.kind == "spread"]
    lib_fly = [i for i in full_lib if i.kind == "fly"]

    results: Dict[pd.Timestamp, Dict[str, FactorProxyFit]] = {}

    for idx_asof, asof in enumerate(asofs):
        i_global = cfg.min_obs + idx_asof
        Xw = window_slice(i_global).dropna(axis=0, how="any")
        if Xw.shape[0] < cfg.min_obs:
            continue

        # PCA on window
        L, F, ev3 = _fit_pca_loadings_and_scores(Xw, cfg.n_components)

        # Build Z on window (full library once per day; can be heavy if flies huge)
        # You can speed up by restricting by kind per factor below.
        Z_full = instruments_to_matrix(Xw, full_lib)

        day_res: Dict[str, FactorProxyFit] = {}

        for pc in F.columns:
            y = F[pc]

            # Restrict candidates by factor kind (optional)
            if cfg.restrict_by_factor_kind:
                if pc == "PC1":
                    lib = lib_out
                elif pc == "PC2":
                    lib = lib_spr
                else:
                    lib = lib_fly
                Z = instruments_to_matrix(Xw, lib)
            else:
                Z = Z_full

            # Stability selection
            prob_s, chosen = stability_select_single_factor(Z, y, cfg.stab_cfg)

            # Final refit
            beta, intercept, r2, rmse = final_refit(Z, y, chosen, cfg.stab_cfg)

            fit = FactorProxyFit(
                asof=asof,
                factor_name=pc,
                selected=list(beta.index),
                selection_prob=prob_s,
                beta=beta,
                intercept=intercept,
                r2_in_window=r2,
                rmse_in_window=rmse,
            )
            day_res[pc] = fit

        results[asof] = day_res

        if cfg.verbose:
            s1 = day_res["PC1"].selected if "PC1" in day_res else []
            s2 = day_res["PC2"].selected if "PC2" in day_res else []
            s3 = day_res["PC3"].selected if "PC3" in day_res else []
            print(
                f"{asof.date()} EVR(1-3)={tuple(np.round(ev3,4))} "
                f"PC1={s1} PC2={s2} PC3={s3}"
            )

    return results


# ----------------------------
# Example usage
# ----------------------------

if __name__ == "__main__":
    # X_changes should be your curve changes in bp from Step 1
    X_changes = None  # replace with your DataFrame

    cfgB = FamilyBConfig(
        n_components=3,
        lookback="2Y",
        min_obs=252,
        include_outrights=True,
        include_spreads=True,
        include_flies=True,
        # If you have many tenors, cap spreads/flies to keep runtime sane:
        max_spreads=3000,
        max_flies=6000,
        min_tenor_gap=2,  # avoids too-tight legs (optional)
        stab_cfg=StabilitySelectionConfig(
            n_subsamples=120,
            subsample_frac=0.6,
            alpha_grid=(0.001, 0.002, 0.005, 0.01),
            l1_ratio=0.9,
            select_prob_threshold=0.6,
            max_features=1,     # if you want exactly 1 instrument per factor like your first code
            final_refit="ridge",
            ridge_alpha=1e-6,
        ),
        restrict_by_factor_kind=True,
        verbose=True,
    )

    # fits = run_daily_family_B(X_changes, cfgB)
    pass
