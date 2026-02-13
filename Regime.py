from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# 1) Regime labeling
# ----------------------------

@dataclass(frozen=True)
class RegimeConfig:
    # Rolling windows in business days
    vol_window: int = 60
    slope_window: int = 1  # usually instantaneous slope is fine
    level_tenor: str = "10Y"
    slope_tenors: Tuple[str, str] = ("2Y", "10Y")

    # Quantile cutoffs
    vol_q: float = 0.8         # top 20% = high vol
    slope_hi_q: float = 0.7    # top 30% = steep
    slope_lo_q: float = 0.3    # bottom 30% = flat/inverted-ish
    level_hi_q: float = 0.7
    level_lo_q: float = 0.3

    # Which regime axes to include
    use_vol: bool = True
    use_slope: bool = True
    use_level: bool = False    # optional; often slope+vol is enough


def _safe_zscore(x: pd.Series) -> pd.Series:
    s = x.std(ddof=0)
    if s == 0 or np.isnan(s):
        return x * 0.0
    return (x - x.mean()) / s


def label_regimes_threshold(
    rates: pd.DataFrame,
    cfg: RegimeConfig,
) -> pd.Series:
    """
    rates: DataFrame indexed by date, columns are tenors (e.g. "2Y","5Y","10Y"...)
          values are yields/rates levels (in % or bp; consistent units).
    returns: Series of regime labels per date
    """

    # 1) Build features
    # Daily changes
    dr = rates.diff()

    # Vol proxy: rolling RMS of changes on a representative tenor (10Y default)
    vol = dr[cfg.level_tenor].rolling(cfg.vol_window).std()

    # Slope proxy: (10Y - 2Y) level slope
    t_short, t_long = cfg.slope_tenors
    slope = (rates[t_long] - rates[t_short]).rolling(cfg.slope_window).mean()

    # Level proxy: representative tenor level
    level = rates[cfg.level_tenor]

    # 2) Compute rolling / expanding quantile thresholds
    # Use expanding quantiles for stability (doesn't "peek" into future if used properly)
    # For backtests, shift(1) to ensure thresholds only use past data.
    vol_hi = vol.expanding(min_periods=cfg.vol_window).quantile(cfg.vol_q).shift(1)

    slope_hi = slope.expanding(min_periods=cfg.vol_window).quantile(cfg.slope_hi_q).shift(1)
    slope_lo = slope.expanding(min_periods=cfg.vol_window).quantile(cfg.slope_lo_q).shift(1)

    level_hi = level.expanding(min_periods=cfg.vol_window).quantile(cfg.level_hi_q).shift(1)
    level_lo = level.expanding(min_periods=cfg.vol_window).quantile(cfg.level_lo_q).shift(1)

    # 3) Bucketize
    labels = []

    for t in rates.index:
        parts = []

        if cfg.use_vol:
            if pd.isna(vol.loc[t]) or pd.isna(vol_hi.loc[t]):
                parts.append("VOL_UNK")
            else:
                parts.append("HIGHVOL" if vol.loc[t] >= vol_hi.loc[t] else "LOWVOL")

        if cfg.use_slope:
            if pd.isna(slope.loc[t]) or pd.isna(slope_hi.loc[t]) or pd.isna(slope_lo.loc[t]):
                parts.append("SLOPE_UNK")
            else:
                if slope.loc[t] >= slope_hi.loc[t]:
                    parts.append("STEEP")
                elif slope.loc[t] <= slope_lo.loc[t]:
                    parts.append("FLAT")
                else:
                    parts.append("MID")

        if cfg.use_level:
            if pd.isna(level.loc[t]) or pd.isna(level_hi.loc[t]) or pd.isna(level_lo.loc[t]):
                parts.append("LEVEL_UNK")
            else:
                if level.loc[t] >= level_hi.loc[t]:
                    parts.append("HIGHLEVEL")
                elif level.loc[t] <= level_lo.loc[t]:
                    parts.append("LOWLEVEL")
                else:
                    parts.append("MIDLEVEL")

        labels.append("_".join(parts))

    return pd.Series(labels, index=rates.index, name="regime")


# ----------------------------
# 2) Regime-aware trainer
# ----------------------------

@dataclass
class RegimeModelBundle:
    models: Dict[str, object]             # your estimator per regime
    regime_counts: Dict[str, int]         # how many samples per regime
    fallback_regime: str                  # used when unseen/rare regimes appear
    regime_config: RegimeConfig


class RegimeAwareTrainer:
    """
    Wraps your existing estimator class and trains one copy per regime.

    You provide:
      - estimator_factory(): returns a *fresh* unfit estimator instance
      - fit/predict interface matches your current code (adapt in 2 lines if needed)
    """

    def __init__(
        self,
        estimator_factory,
        min_samples_per_regime: int = 250,
        fallback: str = "GLOBAL",
    ):
        self.estimator_factory = estimator_factory
        self.min_samples_per_regime = min_samples_per_regime
        self.fallback = fallback

    def fit(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        regimes: pd.Series,
        regime_cfg: RegimeConfig,
    ) -> RegimeModelBundle:
        # Align indices
        common_idx = X.index.intersection(Y.index).intersection(regimes.index)
        Xc, Yc, Rc = X.loc[common_idx], Y.loc[common_idx], regimes.loc[common_idx]

        models: Dict[str, object] = {}
        counts: Dict[str, int] = {}

        # 1) Always train a global fallback model
        global_model = self.estimator_factory()
        global_model.fit(Xc, Yc)
        models[self.fallback] = global_model
        counts[self.fallback] = len(common_idx)

        # 2) Train per regime if enough samples
        for reg, idx in Rc.groupby(Rc).groups.items():
            idx = pd.Index(idx)
            n = len(idx)
            counts[reg] = n

            if n < self.min_samples_per_regime:
                continue

            m = self.estimator_factory()
            m.fit(Xc.loc[idx], Yc.loc[idx])
            models[reg] = m

        # 3) Define fallback regime = the most frequent regime that has a model, else GLOBAL
        candidate = [r for r in counts.keys() if r in models and r != self.fallback]
        if candidate:
            fallback_regime = max(candidate, key=lambda r: counts[r])
        else:
            fallback_regime = self.fallback

        return RegimeModelBundle(
            models=models,
            regime_counts=counts,
            fallback_regime=fallback_regime,
            regime_config=regime_cfg,
        )

    def predict_hard(
        self,
        bundle: RegimeModelBundle,
        X: pd.DataFrame,
        regimes: pd.Series,
    ) -> pd.DataFrame:
        """
        Hard routing: each date uses its regime model if available; else fallback.
        """
        idx = X.index.intersection(regimes.index)
        Xc = X.loc[idx]
        Rc = regimes.loc[idx]

        preds = []
        for t in idx:
            reg = Rc.loc[t]
            model = bundle.models.get(reg, bundle.models[bundle.fallback_regime])
            yhat_t = model.predict(Xc.loc[[t]])
            preds.append(yhat_t)

        return pd.concat(preds, axis=0)

    def predict_soft(
        self,
        bundle: RegimeModelBundle,
        X: pd.DataFrame,
        regime_prob: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Soft routing: weighted average of each regime model prediction.
        regime_prob: DataFrame indexed by date, columns are regime labels, rows sum to 1.
        """
        idx = X.index.intersection(regime_prob.index)
        Xc = X.loc[idx]
        Pc = regime_prob.loc[idx]

        # Keep only regimes we have models for
        usable_regs = [c for c in Pc.columns if c in bundle.models]
        if not usable_regs:
            # If nothing usable, use fallback
            return bundle.models[bundle.fallback_regime].predict(Xc)

        Pc = Pc[usable_regs].div(Pc[usable_regs].sum(axis=1), axis=0).fillna(0.0)

        y_sum = None
        for reg in usable_regs:
            w = Pc[reg].values.reshape(-1, 1)  # (T,1)
            y_reg = bundle.models[reg].predict(Xc).values  # (T, n_targets)
            contrib = w * y_reg
            y_sum = contrib if y_sum is None else (y_sum + contrib)

        return pd.DataFrame(y_sum, index=idx, columns=bundle.models[usable_regs[0]].predict(Xc).columns)


# Whereoot plug it:
# rates: levels (date x tenor)
reg_cfg = RegimeConfig(
    vol_window=60,
    vol_q=0.8,
    use_vol=True,
    use_slope=True,
    use_level=False,
    level_tenor="10Y",
    slope_tenors=("2Y", "10Y")
)

regimes = label_regimes_threshold(rates, reg_cfg)

# X, Y as you already build them in your project:
# X could be selected-tenor changes, or full curve changes, etc.
# Y is target-tenor changes you want to replicate/hedge.

trainer = RegimeAwareTrainer(
    estimator_factory=lambda: ImpliedSparsePCREstimator(your_config),
    min_samples_per_regime=250,
    fallback="GLOBAL",
)

bundle = trainer.fit(X, Y, regimes, reg_cfg)

# In backtest / production:
yhat = trainer.predict_hard(bundle, X_live, regimes_live)
