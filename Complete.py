"""
FULL PROJECT PIPELINE — Corrected version (includes RollingPCAModel)

This is a single-file implementation for Steps 1→4:
  Step 1: load/clean market data, convert units, compute changes
  Step 2: Rolling PCA (RollingPCAModel) with cov/corr + sample/LedoitWolf
  Step 3A: Family A (Implied PCA): fixed or auto anchors + stickiness
  Step 3B: Family B (Sparse proxy instruments): stability selection + ridge refit
  Step 4: Backtest / PnL tracking: full PnL vs hedge PnL (A and B)

Dependencies:
  numpy, pandas
  scikit-learn: LedoitWolf, ElasticNet, Ridge, StandardScaler
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union, Literal, Any

import re
import numpy as np
import pandas as pd

from sklearn.covariance import LedoitWolf
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler


# =============================================================================
# STEP 1 — Data plumbing
# =============================================================================

_TENOR_RE = re.compile(r"^\s*(?P<num>\d+(\.\d+)?)\s*(?P<unit>[DdWwMmYy])\s*$")


def tenor_to_years(label: str) -> float:
    s = str(label).strip()
    m = _TENOR_RE.match(s)
    if not m:
        raise ValueError(f"Unrecognized tenor format: {label!r} (expected '3M', '2Y', '1W', ...)")
    num = float(m.group("num"))
    unit = m.group("unit").upper()
    if unit == "D":
        return num / 365.0
    if unit == "W":
        return num * 7.0 / 365.0
    if unit == "M":
        return num / 12.0
    if unit == "Y":
        return num
    raise ValueError(f"Unsupported tenor unit in {label!r}")


def normalize_tenor_label(label: str) -> str:
    s = str(label).strip().replace(" ", "")
    m = _TENOR_RE.match(s)
    if not m:
        return s
    num = m.group("num")
    unit = m.group("unit").upper()
    if num.endswith(".0"):
        num = num[:-2]
    return f"{num}{unit}"


def sort_tenors(labels: Iterable[str]) -> List[str]:
    parsed = []
    unparsable = []
    for c in labels:
        try:
            y = tenor_to_years(normalize_tenor_label(c))
            parsed.append((y, c))
        except Exception:
            unparsable.append(c)
    parsed_sorted = [c for _, c in sorted(parsed, key=lambda t: t[0])]
    return parsed_sorted + list(unparsable)


@dataclass(frozen=True)
class RateUnits:
    """
    input_unit:
      - "pct": 2.35 means 2.35%
      - "dec": 0.0235 means 2.35%
      - "bp" : 235 means 2.35%
    target_unit: recommended "bp"
    """
    input_unit: str = "pct"
    target_unit: str = "bp"

    def __post_init__(self):
        iu = self.input_unit.lower()
        tu = self.target_unit.lower()
        if iu not in {"pct", "dec", "bp"}:
            raise ValueError(f"input_unit must be pct/dec/bp, got {self.input_unit!r}")
        if tu not in {"pct", "dec", "bp"}:
            raise ValueError(f"target_unit must be pct/dec/bp, got {self.target_unit!r}")


def convert_levels(df: pd.DataFrame, units: RateUnits) -> pd.DataFrame:
    iu = units.input_unit.lower()
    tu = units.target_unit.lower()

    x = df.astype(float)

    if iu == "bp":
        bp = x
    elif iu == "pct":
        bp = x * 100.0
    elif iu == "dec":
        bp = x * 10000.0
    else:
        raise ValueError(f"Unknown input_unit: {units.input_unit!r}")

    if tu == "bp":
        return bp
    if tu == "pct":
        return bp / 100.0
    if tu == "dec":
        return bp / 10000.0
    raise ValueError(f"Unknown target_unit: {units.target_unit!r}")


def _read_table(
    path: str,
    sheet_name: Optional[Union[str, int]] = None,
    date_col: Optional[str] = None,
    index_col: Optional[Union[str, int]] = 0,
) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        raw = pd.read_csv(path)
    elif path.lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(path, sheet_name=sheet_name)
    else:
        raise ValueError(f"Unsupported file type for {path!r}. Use .csv or .xlsx/.xls")

    if date_col is not None:
        if date_col not in raw.columns:
            raise ValueError(f"date_col {date_col!r} not found.")
        raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
        raw = raw.set_index(date_col)
    else:
        raw = raw.set_index(index_col)
        raw.index = pd.to_datetime(raw.index, errors="coerce")
    return raw


def clean_market_data(
    raw: pd.DataFrame,
    *,
    restrict_tenors: Optional[Iterable[str]] = None,
    min_valid_frac_per_row: float = 0.8,
    max_gap_ffill: int = 2,
    max_gap_bfill: int = 0,
    normalize_columns: bool = True,
    sort_columns_by_tenor: bool = True,
) -> pd.DataFrame:
    df = raw.copy()
    df = df[~df.index.isna()].copy()
    df = df[~df.index.duplicated(keep="last")].sort_index()

    if normalize_columns:
        df.columns = [normalize_tenor_label(c) for c in df.columns]

    if restrict_tenors is not None:
        restrict = [normalize_tenor_label(c) for c in restrict_tenors]
        missing = [c for c in restrict if c not in df.columns]
        if missing:
            raise ValueError(f"Requested tenors not found in data: {missing}")
        df = df.loc[:, restrict].copy()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(axis=1, how="all")
    if df.shape[1] == 0:
        raise ValueError("After cleaning, no columns left.")

    valid_frac = df.notna().mean(axis=1)
    df = df.loc[valid_frac >= min_valid_frac_per_row].copy()
    if df.shape[0] < 5:
        raise ValueError("Too few rows after cleaning; relax thresholds.")

    if max_gap_ffill and max_gap_ffill > 0:
        df = df.ffill(limit=max_gap_ffill)
    if max_gap_bfill and max_gap_bfill > 0:
        df = df.bfill(limit=max_gap_bfill)

    if sort_columns_by_tenor:
        df = df.loc[:, sort_tenors(df.columns)]

    return df


def build_market_matrices(
    *,
    path: Optional[str] = None,
    sheet_name: Optional[Union[str, int]] = None,
    date_col: Optional[str] = None,
    index_col: Optional[Union[str, int]] = 0,
    raw_df: Optional[pd.DataFrame] = None,
    restrict_tenors: Optional[List[str]] = None,
    units: RateUnits = RateUnits("pct", "bp"),
    min_valid_frac_per_row: float = 0.8,
    max_gap_ffill: int = 2,
    max_gap_bfill: int = 0,
    normalize_columns: bool = True,
    sort_columns_by_tenor: bool = True,
    compute_changes: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if raw_df is None and path is None:
        raise ValueError("Provide either raw_df or path.")
    raw = raw_df if raw_df is not None else _read_table(path, sheet_name=sheet_name, date_col=date_col, index_col=index_col)

    cleaned = clean_market_data(
        raw,
        restrict_tenors=restrict_tenors,
        min_valid_frac_per_row=min_valid_frac_per_row,
        max_gap_ffill=max_gap_ffill,
        max_gap_bfill=max_gap_bfill,
        normalize_columns=normalize_columns,
        sort_columns_by_tenor=sort_columns_by_tenor,
    )

    levels = convert_levels(cleaned, units)
    changes = levels.diff().iloc[1:].copy() if compute_changes else pd.DataFrame(index=levels.index, columns=levels.columns, dtype=float)

    meta = {
        "n_dates_levels": int(levels.shape[0]),
        "n_dates_changes": int(changes.shape[0]),
        "n_tenors": int(levels.shape[1]),
        "input_unit": units.input_unit,
        "target_unit": units.target_unit,
        "tenors": list(levels.columns),
    }
    return levels, changes, meta


# =============================================================================
# STEP 2 — Rolling PCA (THIS WAS THE MISSING CLASS)
# =============================================================================

@dataclass
class PCAFitResult:
    asof: pd.Timestamp
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    n_obs: int
    method: str
    estimator: str
    n_components: int
    explained_var_ratio: pd.Series      # PC1..PCN
    loadings: pd.DataFrame              # (tenors x k)
    scores_in_window: pd.DataFrame      # (dates x k)


class RollingPCAModel:
    """
    Rolling PCA engine:
      - cov/corr PCA with sample or LedoitWolf covariance
      - symmetric eigendecomposition
      - deterministic sign convention
      - rolling window by row-count or by pandas offset ("2Y", "18M", ...)
    """
    def __init__(
        self,
        n_components: int = 3,
        method: Literal["cov", "corr"] = "cov",
        estimator: Literal["sample", "ledoitwolf"] = "ledoitwolf",
        min_obs: int = 252,
    ):
        self.n_components = int(n_components)
        self.method = method
        self.estimator = estimator
        self.min_obs = int(min_obs)

    @staticmethod
    def _sym_eigh_sorted(S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        S = np.asarray(S, dtype=float)
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
        return eigvals, eigvecs

    def _compute_cov_or_corr(self, Xw: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        Xw = Xw.dropna(axis=0, how="any")
        if Xw.shape[0] < 10:
            raise ValueError(f"Too few observations after NaN drop: {Xw.shape[0]}")
        M = Xw.to_numpy(dtype=float)

        if self.estimator == "ledoitwolf":
            cov = LedoitWolf().fit(M).covariance_
        elif self.estimator == "sample":
            cov = np.cov(M, rowvar=False, ddof=1)
        else:
            raise ValueError(f"Unknown estimator: {self.estimator!r}")

        cov = np.asarray(cov, dtype=float)

        if self.method == "cov":
            return cov, Xw

        if self.method == "corr":
            d = np.sqrt(np.clip(np.diag(cov), 1e-18, np.inf))
            invd = 1.0 / d
            corr = cov * invd[None, :] * invd[:, None]
            corr = 0.5 * (corr + corr.T)
            np.fill_diagonal(corr, 1.0)
            return corr, Xw

        raise ValueError(f"Unknown method: {self.method!r}")

    def fit_on_window(self, X_window: pd.DataFrame, asof: pd.Timestamp) -> PCAFitResult:
        S, Xw = self._compute_cov_or_corr(X_window)
        eigvals, eigvecs = self._sym_eigh_sorted(S)

        total = float(np.sum(eigvals))
        if total <= 0 or not np.isfinite(total):
            raise ValueError("Invalid total variance from eigenvalues.")
        evr = eigvals / total

        k = self.n_components
        pcs = [f"PC{i+1}" for i in range(k)]
        loadings = pd.DataFrame(eigvecs[:, :k], index=Xw.columns, columns=pcs)
        scores = pd.DataFrame(Xw.to_numpy(dtype=float) @ loadings.to_numpy(dtype=float), index=Xw.index, columns=pcs)

        evr_s = pd.Series(evr, index=[f"PC{i+1}" for i in range(len(evr))])

        return PCAFitResult(
            asof=pd.Timestamp(asof),
            window_start=pd.Timestamp(Xw.index.min()),
            window_end=pd.Timestamp(Xw.index.max()),
            n_obs=int(Xw.shape[0]),
            method=self.method,
            estimator=self.estimator,
            n_components=k,
            explained_var_ratio=evr_s,
            loadings=loadings,
            scores_in_window=scores,
        )

    def fit_rolling(
        self,
        X: pd.DataFrame,
        lookback: Union[int, str] = "2Y",
        step: int = 1,
        verbose: bool = False,
    ) -> Dict[pd.Timestamp, PCAFitResult]:
        X = X.sort_index()
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a DatetimeIndex.")
        if X.shape[0] < self.min_obs:
            raise ValueError(f"Not enough rows in X ({X.shape[0]}) for min_obs={self.min_obs}")

        results: Dict[pd.Timestamp, PCAFitResult] = {}

        if isinstance(lookback, int):
            for i in range(self.min_obs, X.shape[0], step):
                asof = X.index[i]
                start_i = max(0, i - lookback + 1)
                Xw = X.iloc[start_i:i + 1]
                if Xw.dropna(axis=0, how="any").shape[0] < self.min_obs:
                    continue
                try:
                    res = self.fit_on_window(Xw, asof=asof)
                    results[asof] = res
                    if verbose:
                        ev3 = res.explained_var_ratio.iloc[:3].to_numpy()
                        print(f"{asof.date()} EVR(1-3)={tuple(np.round(ev3,4))}")
                except Exception as e:
                    if verbose:
                        print(f"[WARN] asof={asof.date()} failed: {e}")
            return results

        offset = pd.tseries.frequencies.to_offset(lookback)
        for i in range(self.min_obs, X.shape[0], step):
            asof = X.index[i]
            start = asof - offset
            Xw = X.loc[(X.index > start) & (X.index <= asof)]
            if Xw.dropna(axis=0, how="any").shape[0] < self.min_obs:
                continue
            try:
                res = self.fit_on_window(Xw, asof=asof)
                results[asof] = res
                if verbose:
                    ev3 = res.explained_var_ratio.iloc[:3].to_numpy()
                    print(f"{asof.date()} EVR(1-3)={tuple(np.round(ev3,4))}")
            except Exception as e:
                if verbose:
                    print(f"[WARN] asof={asof.date()} failed: {e}")

        return results


# =============================================================================
# STEP 3A — Family A (Implied PCA): anchors
# =============================================================================

def _r2_like(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    num = float(np.linalg.norm(y_true - y_hat, ord="fro") ** 2)
    den = float(np.linalg.norm(y_true, ord="fro") ** 2)
    if den <= 1e-18:
        return np.nan
    return 1.0 - num / den


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.nanmean(d * d)))


def _cond_number(A: np.ndarray) -> float:
    try:
        return float(np.linalg.cond(A))
    except Exception:
        return float("inf")


@dataclass
class AnchorSelectionConfig:
    k_anchors: int = 4
    anchors_mode: Literal["auto", "fixed"] = "fixed"
    fixed_anchors: Optional[List[str]] = None
    candidate_tenors: Optional[List[str]] = None

    ridge_lambda: float = 1e-6
    cond_max: float = 1e4
    cond_penalty_lambda: float = 0.0
    min_improvement_to_switch: float = 0.002


@dataclass(frozen=True)
class FamilyAResult:
    asof: pd.Timestamp
    anchors: List[str]
    mapping_B: pd.DataFrame  # anchors x tenors
    score: float
    rmse: float
    cond: float


def _build_familyA_mapping(
    Xw: pd.DataFrame,
    L: pd.DataFrame,
    anchors: List[str],
    ridge_lambda: float,
) -> Tuple[pd.DataFrame, float, float, float]:
    tenors = list(L.index)
    Xw2 = Xw.loc[:, tenors].dropna(axis=0, how="any")
    XS = Xw2.loc[:, anchors]
    F = Xw2.to_numpy(dtype=float) @ L.to_numpy(dtype=float)  # (T x k)

    XSm = XS.to_numpy(dtype=float)
    XtX = XSm.T @ XSm + ridge_lambda * np.eye(XSm.shape[1])
    A = np.linalg.solve(XtX, XSm.T @ F)        # (|S| x k)
    Bm = A @ L.to_numpy(dtype=float).T         # (|S| x N)
    B = pd.DataFrame(Bm, index=anchors, columns=tenors)

    Xhat = XSm @ Bm
    score = _r2_like(Xw2.to_numpy(dtype=float), Xhat)
    rmse = _rmse(Xw2.to_numpy(dtype=float), Xhat)

    LS = L.loc[anchors, :].to_numpy(dtype=float)
    cond = _cond_number(LS)
    return B, score, rmse, cond


def _penalized(score: float, cond: float, lam: float) -> float:
    if lam <= 0:
        return score
    if not np.isfinite(cond) or cond <= 0:
        return -np.inf
    return score - lam * float(np.log(cond))


def _greedy_auto_anchors(
    Xw: pd.DataFrame,
    L: pd.DataFrame,
    cfg: AnchorSelectionConfig,
) -> FamilyAResult:
    tenors_all = list(L.index)
    candidates = cfg.candidate_tenors if cfg.candidate_tenors is not None else tenors_all
    candidates = [t for t in candidates if t in tenors_all]
    if not candidates:
        raise ValueError("No candidates available for auto anchor selection.")

    selected: List[str] = []
    best_pack = None

    for _ in range(cfg.k_anchors):
        best_pscore = -np.inf
        best_c_pack = None

        for c in candidates:
            if c in selected:
                continue
            trial = selected + [c]
            try:
                B, score, rmse, cond = _build_familyA_mapping(Xw, L, trial, cfg.ridge_lambda)
            except Exception:
                continue
            if cond > cfg.cond_max:
                continue
            pscore = _penalized(score, cond, cfg.cond_penalty_lambda)
            if pscore > best_pscore:
                best_pscore = pscore
                best_c_pack = (trial, B, score, rmse, cond)

        if best_c_pack is None:
            raise ValueError("Could not find feasible anchors under constraints.")
        selected.append(best_c_pack[0][-1])
        best_pack = best_c_pack

    anchors, B, score, rmse, cond = best_pack  # type: ignore[misc]
    return FamilyAResult(asof=pd.NaT, anchors=anchors, mapping_B=B, score=score, rmse=rmse, cond=cond)


# =============================================================================
# STEP 3B — Family B (Sparse proxies)
# =============================================================================

@dataclass(frozen=True)
class InstrumentSpec:
    name: str
    kind: Literal["outright", "spread", "fly"]
    legs: Tuple[str, ...]
    weights: Tuple[float, ...]


def build_instrument_library(
    tenors: List[str],
    *,
    include_outrights: bool = True,
    include_spreads: bool = True,
    include_flies: bool = True,
    max_spreads: Optional[int] = 3000,
    max_flies: Optional[int] = 6000,
    min_tenor_gap: int = 2,
) -> List[InstrumentSpec]:
    tenors = list(tenors)
    n = len(tenors)
    lib: List[InstrumentSpec] = []

    if include_outrights:
        for t in tenors:
            lib.append(InstrumentSpec(f"OUT_{t}", "outright", (t,), (1.0,)))

    if include_spreads:
        count = 0
        for i in range(n):
            for j in range(i + min_tenor_gap, n):
                s, l = tenors[i], tenors[j]
                lib.append(InstrumentSpec(f"SPR_{s}_{l}", "spread", (s, l), (-1.0, 1.0)))
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
                    lib.append(InstrumentSpec(f"FLY_{s}_{m}_{l}", "fly", (s, m, l), (1.0, -2.0, 1.0)))
                    count += 1
                    if max_flies is not None and count >= max_flies:
                        break
                if max_flies is not None and count >= max_flies:
                    break
            if max_flies is not None and count >= max_flies:
                break

    return lib


def instruments_to_matrix(X: pd.DataFrame, instruments: List[InstrumentSpec]) -> pd.DataFrame:
    Z = pd.DataFrame(index=X.index)
    for inst in instruments:
        vals = np.zeros(len(X), dtype=float)
        for leg, w in zip(inst.legs, inst.weights):
            vals += w * X[leg].to_numpy(dtype=float)
        Z[inst.name] = vals
    return Z


@dataclass
class StabilitySelectionConfig:
    n_subsamples: int = 120
    subsample_frac: float = 0.6
    random_state: int = 123
    alpha_grid: Tuple[float, ...] = (0.001, 0.002, 0.005, 0.01)
    l1_ratio: float = 0.9
    select_prob_threshold: float = 0.6
    max_features: int = 1
    ridge_alpha: float = 1e-6
    standardize_Z: bool = True
    standardize_y: bool = True


@dataclass(frozen=True)
class FactorProxyFit:
    asof: pd.Timestamp
    factor_name: str
    selected: List[str]
    beta: pd.Series
    intercept: float
    selection_prob: pd.Series


def _stability_select(Z: pd.DataFrame, y: pd.Series, cfg: StabilitySelectionConfig) -> Tuple[pd.Series, List[str]]:
    df = pd.concat([Z, y.rename("y")], axis=1).dropna(axis=0, how="any")
    if df.shape[0] < 100:
        raise ValueError(f"Too few rows for stability selection: {df.shape[0]}")
    Zm = df[Z.columns].to_numpy(dtype=float)
    ym = df["y"].to_numpy(dtype=float)

    rng = np.random.default_rng(cfg.random_state)
    n, m = Zm.shape

    if cfg.standardize_Z:
        Zm = StandardScaler(with_mean=True, with_std=True).fit_transform(Zm)
    if cfg.standardize_y:
        ym = StandardScaler(with_mean=True, with_std=True).fit_transform(ym.reshape(-1, 1)).reshape(-1)

    counts = np.zeros(m, dtype=int)
    n_sub = max(1, int(cfg.subsample_frac * n))
    idx_all = np.arange(n)
    alphas = list(cfg.alpha_grid)
    if not alphas:
        raise ValueError("alpha_grid must not be empty.")

    for b in range(cfg.n_subsamples):
        idx = rng.choice(idx_all, size=n_sub, replace=False)
        Zb, yb = Zm[idx, :], ym[idx]
        alpha = alphas[b % len(alphas)]
        model = ElasticNet(alpha=alpha, l1_ratio=cfg.l1_ratio, fit_intercept=True, max_iter=20000, tol=1e-6)
        model.fit(Zb, yb)
        counts += (np.abs(model.coef_) > 1e-12).astype(int)

    probs = counts / float(cfg.n_subsamples)
    prob_s = pd.Series(probs, index=Z.columns).sort_values(ascending=False)

    chosen = prob_s[prob_s >= cfg.select_prob_threshold].index.tolist()
    if len(chosen) > cfg.max_features:
        chosen = chosen[: cfg.max_features]
    if not chosen:
        chosen = [prob_s.index[0]]

    return prob_s, chosen


def _refit_ridge(Z: pd.DataFrame, y: pd.Series, selected: List[str], ridge_alpha: float) -> Tuple[pd.Series, float]:
    df = pd.concat([Z[selected], y.rename("y")], axis=1).dropna(axis=0, how="any")
    Zm = df[selected].to_numpy(dtype=float)
    ym = df["y"].to_numpy(dtype=float)

    model = Ridge(alpha=ridge_alpha, fit_intercept=True)
    model.fit(Zm, ym)
    beta = pd.Series(model.coef_.reshape(-1), index=selected)
    intercept = float(model.intercept_)
    return beta, intercept


# =============================================================================
# STEP 4 — Backtest / PnL tracking
# =============================================================================

def align_sensitivities(
    X_changes: pd.DataFrame,
    sens_by_tenor: Optional[pd.DataFrame] = None,
    sens_vec: Optional[pd.Series] = None,
) -> pd.DataFrame:
    tenors = list(X_changes.columns)
    idx = X_changes.index

    if sens_by_tenor is not None:
        S = sens_by_tenor.copy()
        if not isinstance(S.index, pd.DatetimeIndex):
            raise ValueError("sens_by_tenor must have a DatetimeIndex.")
        S = S.reindex(idx).ffill()
        missing = [c for c in tenors if c not in S.columns]
        if missing:
            raise ValueError(f"sens_by_tenor missing columns: {missing}")
        return S.loc[:, tenors].astype(float)

    if sens_vec is not None:
        sv = pd.Series(sens_vec).copy()
        if all(t in sv.index for t in tenors):
            sv = sv.reindex(tenors)
        else:
            if len(sv) != len(tenors):
                raise ValueError("sens_vec length mismatch.")
            sv.index = tenors
        return pd.DataFrame(np.tile(sv.to_numpy(), (len(idx), 1)), index=idx, columns=tenors).astype(float)

    raise ValueError("Provide sens_by_tenor or sens_vec.")


def compute_full_pnl(X_changes: pd.DataFrame, sens: pd.DataFrame) -> pd.Series:
    return (X_changes.astype(float) * sens.astype(float)).sum(axis=1).rename("pnl_full")


def fit_hedge_weights(Z: pd.DataFrame, pnl: pd.Series, ridge_alpha: float) -> Tuple[pd.Series, float]:
    df = pd.concat([Z, pnl.rename("pnl")], axis=1).dropna(axis=0, how="any")
    if df.shape[0] < 50:
        raise ValueError("Too few rows to fit hedge weights.")
    X = df[Z.columns].to_numpy(dtype=float)
    y = df["pnl"].to_numpy(dtype=float)
    model = Ridge(alpha=ridge_alpha, fit_intercept=True)
    model.fit(X, y)
    w = pd.Series(model.coef_.reshape(-1), index=Z.columns)
    b = float(model.intercept_)
    return w, b


def parse_instrument(name: str) -> Tuple[List[str], List[float]]:
    if name.startswith("OUT_"):
        t = name.replace("OUT_", "", 1)
        return [t], [1.0]
    if name.startswith("SPR_"):
        _, s, l = name.split("_", 2)
        return [s, l], [-1.0, 1.0]
    if name.startswith("FLY_"):
        _, s, m, l = name.split("_", 3)
        return [s, m, l], [1.0, -2.0, 1.0]
    raise ValueError(f"Unknown instrument name: {name}")


def build_selected_moves(X_changes: pd.DataFrame, selected: List[str]) -> pd.DataFrame:
    Z = pd.DataFrame(index=X_changes.index)
    for inst in selected:
        legs, ww = parse_instrument(inst)
        vals = np.zeros(len(X_changes), dtype=float)
        for leg, w in zip(legs, ww):
            vals += w * X_changes[leg].to_numpy(dtype=float)
        Z[inst] = vals
    return Z


@dataclass
class BacktestConfig:
    lookback: Union[int, str] = "2Y"
    min_obs: int = 252
    ridge_alpha: float = 1e-6


def _summary(pnl_full: pd.Series, pnl_hedge: pd.Series) -> Dict[str, float]:
    df = pd.concat([pnl_full.rename("full"), pnl_hedge.rename("hedge")], axis=1).dropna()
    if df.shape[0] == 0:
        return {k: np.nan for k in ["rmse", "mae", "corr", "p95_abs", "p99_abs", "n"]}
    resid = df["full"] - df["hedge"]
    r = resid.to_numpy(dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean(r * r))),
        "mae": float(np.mean(np.abs(r))),
        "corr": float(df["full"].corr(df["hedge"])) if df.shape[0] >= 20 else np.nan,
        "p95_abs": float(np.quantile(np.abs(r), 0.95)),
        "p99_abs": float(np.quantile(np.abs(r), 0.99)),
        "n": float(df.shape[0]),
    }


def make_synthetic_sensitivities(
    X_changes: pd.DataFrame,
    center_tenor: str = "10Y",
    width: float = 2.0,
    total_risk: float = 1e6,
) -> pd.Series:
    tenors = list(X_changes.columns)
    years = []
    for t in tenors:
        try:
            years.append(tenor_to_years(t))
        except Exception:
            years.append(np.nan)
    years = np.array(years, dtype=float)

    if center_tenor in tenors:
        center_year = tenor_to_years(center_tenor)
    else:
        center_year = float(np.nanmedian(years[np.isfinite(years)]))

    w = np.exp(-0.5 * ((years - center_year) / max(width, 1e-6)) ** 2)
    w[~np.isfinite(w)] = 0.0
    w = w / (np.sum(np.abs(w)) + 1e-18) * total_risk
    return pd.Series(w, index=tenors, name="sens")


# =============================================================================
# ONE MAIN BUILDER FUNCTION
# =============================================================================

@dataclass
class PCAHedgeOptions:
    # Data
    data_mode: Literal["path", "raw_df", "x_changes"] = "path"
    path: Optional[str] = None
    raw_df: Optional[pd.DataFrame] = None
    x_changes: Optional[pd.DataFrame] = None

    sheet_name: Optional[Union[str, int]] = None
    date_col: Optional[str] = None
    index_col: Optional[Union[str, int]] = 0
    restrict_tenors: Optional[List[str]] = None
    units: RateUnits = RateUnits("pct", "bp")
    min_valid_frac_per_row: float = 0.8
    max_gap_ffill: int = 2
    max_gap_bfill: int = 0
    normalize_columns: bool = True
    sort_columns_by_tenor: bool = True

    # PCA
    n_components: int = 3
    pca_method: Literal["cov", "corr"] = "cov"
    pca_estimator: Literal["sample", "ledoitwolf"] = "ledoitwolf"
    lookback: Union[int, str] = "2Y"
    min_obs: int = 252
    step: int = 1

    # Families
    run_familyA: bool = True
    run_familyB: bool = True
    familyA: AnchorSelectionConfig = AnchorSelectionConfig()
    familyB: StabilitySelectionConfig = StabilitySelectionConfig()
    familyB_max_spreads: Optional[int] = 3000
    familyB_max_flies: Optional[int] = 6000
    familyB_min_tenor_gap: int = 2
    familyB_restrict_by_kind: bool = True  # PC1->outrights, PC2->spreads, PC3->flies

    # Backtest
    run_backtest: bool = False
    sens_by_tenor: Optional[pd.DataFrame] = None
    sens_vec: Optional[pd.Series] = None
    backtest: BacktestConfig = BacktestConfig()

    verbose: bool = False


def build_pca_hedge_model(opts: PCAHedgeOptions) -> Dict[str, Any]:
    # 1) Build X_changes
    levels = None
    meta: Dict[str, Any] = {}

    if opts.data_mode == "x_changes":
        if opts.x_changes is None:
            raise ValueError("data_mode='x_changes' requires x_changes.")
        X_changes = opts.x_changes.copy()
        if not isinstance(X_changes.index, pd.DatetimeIndex):
            raise ValueError("x_changes must have a DatetimeIndex.")
        X_changes = X_changes.sort_index()
        X_changes.columns = [normalize_tenor_label(c) for c in X_changes.columns]
        X_changes = X_changes.loc[:, sort_tenors(X_changes.columns)]
        meta["source"] = "x_changes"
    else:
        levels, X_changes, meta = build_market_matrices(
            path=opts.path if opts.data_mode == "path" else None,
            raw_df=opts.raw_df if opts.data_mode == "raw_df" else None,
            sheet_name=opts.sheet_name,
            date_col=opts.date_col,
            index_col=opts.index_col,
            restrict_tenors=opts.restrict_tenors,
            units=opts.units,
            min_valid_frac_per_row=opts.min_valid_frac_per_row,
            max_gap_ffill=opts.max_gap_ffill,
            max_gap_bfill=opts.max_gap_bfill,
            normalize_columns=opts.normalize_columns,
            sort_columns_by_tenor=opts.sort_columns_by_tenor,
            compute_changes=True,
        )
        meta["source"] = "path/raw_df"

    if X_changes.shape[0] < opts.min_obs:
        raise ValueError(f"Not enough rows in X_changes ({X_changes.shape[0]}) for min_obs={opts.min_obs}")

    # 2) Rolling PCA
    pca_engine = RollingPCAModel(
        n_components=opts.n_components,
        method=opts.pca_method,
        estimator=opts.pca_estimator,
        min_obs=opts.min_obs,
    )
    pca_states = pca_engine.fit_rolling(X_changes, lookback=opts.lookback, step=opts.step, verbose=False)

    if opts.verbose and pca_states:
        last_asof = list(pca_states.keys())[-1]
        ev3 = pca_states[last_asof].explained_var_ratio.iloc[:3].to_numpy()
        print(f"[PCA] last asof={last_asof.date()} EVR(1-3)={tuple(np.round(ev3,4))}")

    out: Dict[str, Any] = {
        "levels": levels,
        "changes": X_changes,
        "meta": meta,
        "pca_states": pca_states,
    }

    # Prebuild Family B instrument library
    tenors = list(X_changes.columns)
    lib_full = build_instrument_library(
        tenors=tenors,
        include_outrights=True,
        include_spreads=True,
        include_flies=True,
        max_spreads=opts.familyB_max_spreads,
        max_flies=opts.familyB_max_flies,
        min_tenor_gap=opts.familyB_min_tenor_gap,
    )
    lib_out = [i for i in lib_full if i.kind == "outright"]
    lib_spr = [i for i in lib_full if i.kind == "spread"]
    lib_fly = [i for i in lib_full if i.kind == "fly"]

    # 3) Family A + B (per asof)
    familyA_out: Dict[pd.Timestamp, FamilyAResult] = {}
    familyB_out: Dict[pd.Timestamp, Dict[str, FactorProxyFit]] = {}

    prevA: Optional[FamilyAResult] = None

    for asof, state in pca_states.items():
        # Recreate the same window slice used for that asof (for selection/backtest consistency)
        i = X_changes.index.get_loc(asof)
        if isinstance(opts.lookback, int):
            start_i = max(0, i - opts.lookback + 1)
            Xw = X_changes.iloc[start_i:i + 1]
        else:
            offset = pd.tseries.frequencies.to_offset(opts.lookback)
            start = asof - offset
            Xw = X_changes.loc[(X_changes.index > start) & (X_changes.index <= asof)]
        Xw = Xw.dropna(axis=0, how="any")
        if Xw.shape[0] < opts.min_obs:
            continue

        L = state.loadings
        F = state.scores_in_window

        # --- Family A
        if opts.run_familyA:
            cfgA = opts.familyA

            if cfgA.anchors_mode == "fixed":
                if not cfgA.fixed_anchors:
                    raise ValueError("FamilyA fixed mode requires fixed_anchors.")
                anchors = [normalize_tenor_label(a) for a in cfgA.fixed_anchors]
                missing = [a for a in anchors if a not in X_changes.columns]
                if missing:
                    raise ValueError(f"Fixed anchors not in curve columns: {missing}")
                B, score, rmse, cond = _build_familyA_mapping(Xw, L, anchors, cfgA.ridge_lambda)
                newA = FamilyAResult(asof=asof, anchors=anchors, mapping_B=B, score=score, rmse=rmse, cond=cond)
            else:
                autoA = _greedy_auto_anchors(Xw, L, cfgA)
                newA = FamilyAResult(asof=asof, anchors=autoA.anchors, mapping_B=autoA.mapping_B, score=autoA.score, rmse=autoA.rmse, cond=autoA.cond)

            if prevA is None:
                chosenA = newA
                switched = True
            else:
                improvement = newA.score - prevA.score
                if improvement < cfgA.min_improvement_to_switch:
                    chosenA = FamilyAResult(asof=asof, anchors=prevA.anchors, mapping_B=prevA.mapping_B,
                                           score=prevA.score, rmse=prevA.rmse, cond=prevA.cond)
                    switched = False
                else:
                    chosenA = newA
                    switched = True

            familyA_out[asof] = chosenA
            prevA = chosenA

            if opts.verbose:
                tag = "SWITCH" if switched else "KEEP"
                print(f"[A] {asof.date()} {tag} anchors={chosenA.anchors} score={chosenA.score:.5f} cond={chosenA.cond:.1f}")

        # --- Family B
        if opts.run_familyB:
            cfgB = opts.familyB

            # Ensure alignment between Xw and PCA scores (scores were computed on the window after dropna)
            XwB = Xw.loc[F.index]

            day_res: Dict[str, FactorProxyFit] = {}
            for pc in F.columns:
                y = F[pc]

                if opts.familyB_restrict_by_kind:
                    if pc == "PC1":
                        Z = instruments_to_matrix(XwB, lib_out)
                    elif pc == "PC2":
                        Z = instruments_to_matrix(XwB, lib_spr)
                    else:
                        Z = instruments_to_matrix(XwB, lib_fly)
                else:
                    Z = instruments_to_matrix(XwB, lib_full)

                prob_s, chosen = _stability_select(Z, y, cfgB)
                beta, intercept = _refit_ridge(Z, y, chosen, cfgB.ridge_alpha)

                day_res[pc] = FactorProxyFit(
                    asof=asof,
                    factor_name=pc,
                    selected=list(beta.index),
                    beta=beta,
                    intercept=intercept,
                    selection_prob=prob_s,
                )

            familyB_out[asof] = day_res

            if opts.verbose:
                s1 = day_res.get("PC1").selected if "PC1" in day_res else []
                s2 = day_res.get("PC2").selected if "PC2" in day_res else []
                s3 = day_res.get("PC3").selected if "PC3" in day_res else []
                print(f"[B] {asof.date()} PC1={s1} PC2={s2} PC3={s3}")

    if opts.run_familyA:
        out["familyA"] = familyA_out
    if opts.run_familyB:
        out["familyB"] = familyB_out

    # 4) Backtest
    if opts.run_backtest:
        sens = align_sensitivities(X_changes, sens_by_tenor=opts.sens_by_tenor, sens_vec=opts.sens_vec)
        pnl_full = compute_full_pnl(X_changes, sens)
        bt = opts.backtest

        if opts.run_familyA:
            pnl_A = pd.Series(index=X_changes.index, dtype=float, name="pnl_A")
            for asof, Ares in familyA_out.items():
                if asof not in X_changes.index:
                    continue
                i = X_changes.index.get_loc(asof)
                if i >= len(X_changes.index) - 1:
                    continue
                # window
                if isinstance(bt.lookback, int):
                    start_i = max(0, i - bt.lookback + 1)
                    Xw = X_changes.iloc[start_i:i + 1]
                else:
                    offset = pd.tseries.frequencies.to_offset(bt.lookback)
                    start = asof - offset
                    Xw = X_changes.loc[(X_changes.index > start) & (X_changes.index <= asof)]
                Xw = Xw.dropna(axis=0, how="any")
                if Xw.shape[0] < bt.min_obs:
                    continue

                Z_w = Xw.loc[:, Ares.anchors]
                w, b = fit_hedge_weights(Z_w, pnl_full.loc[Xw.index], bt.ridge_alpha)

                t_next = X_changes.index[i + 1]
                Z_next = X_changes.loc[[t_next], Ares.anchors]
                pnl_A.loc[t_next] = float(b + (Z_next.iloc[0] * w).sum())

            out["backtestA"] = {"pnl_full": pnl_full, "pnl_A": pnl_A, "resid_A": pnl_full - pnl_A}
            out["statsA"] = _summary(pnl_full, pnl_A)

        if opts.run_familyB:
            pnl_B = pd.Series(index=X_changes.index, dtype=float, name="pnl_B")
            for asof, Bres in familyB_out.items():
                if asof not in X_changes.index:
                    continue
                i = X_changes.index.get_loc(asof)
                if i >= len(X_changes.index) - 1:
                    continue
                # window
                if isinstance(bt.lookback, int):
                    start_i = max(0, i - bt.lookback + 1)
                    Xw = X_changes.iloc[start_i:i + 1]
                else:
                    offset = pd.tseries.frequencies.to_offset(bt.lookback)
                    start = asof - offset
                    Xw = X_changes.loc[(X_changes.index > start) & (X_changes.index <= asof)]
                Xw = Xw.dropna(axis=0, how="any")
                if Xw.shape[0] < bt.min_obs:
                    continue

                selected = sorted(set(inst for fit in Bres.values() for inst in fit.selected))
                if not selected:
                    continue

                Z_w = build_selected_moves(Xw, selected)
                w, b = fit_hedge_weights(Z_w, pnl_full.loc[Xw.index], bt.ridge_alpha)

                t_next = X_changes.index[i + 1]
                Z_next = build_selected_moves(X_changes.loc[[t_next]], selected)
                pnl_B.loc[t_next] = float(b + (Z_next.iloc[0] * w).sum())

            out["backtestB"] = {"pnl_full": pnl_full, "pnl_B": pnl_B, "resid_B": pnl_full - pnl_B}
            out["statsB"] = _summary(pnl_full, pnl_B)

    return out


# =============================================================================
# Example
# =============================================================================
if __name__ == "__main__":
    # Example usage (edit your paths):
    opts = PCAHedgeOptions(
        data_mode="path",
        path="curve_timeseries.csv",
        units=RateUnits("pct", "bp"),
        n_components=3,
        pca_method="cov",
        pca_estimator="ledoitwolf",
        lookback="2Y",
        min_obs=252,
        run_familyA=True,
        run_familyB=True,
        familyA=AnchorSelectionConfig(
            anchors_mode="fixed",
            fixed_anchors=["3Y", "5Y", "10Y", "20Y"],
            k_anchors=4,
            ridge_lambda=1e-6,
            cond_max=1e4,
            min_improvement_to_switch=0.002,
        ),
        run_backtest=False,
        verbose=True,
    )

    # model = build_pca_hedge_model(opts)
    # print(model.keys())
    print("Corrected full code loaded (includes RollingPCAModel). Set paths and uncomment the call.")
