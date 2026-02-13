from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class ClusterFeatureConfig:
    # Tenors used for feature construction
    level_tenor: str = "10Y"
    slope_tenors: Tuple[str, str] = ("2Y", "10Y")     # long-short
    curvature_tenors: Tuple[str, str, str] = ("2Y", "10Y", "30Y")  # short, belly, long

    # Vol proxy window
    vol_window: int = 60

    # Include which features
    use_level: bool = True
    use_slope: bool = True
    use_curvature: bool = True
    use_realized_vol: bool = True

    # Optional: standardize to bps-like scale if your data is in %
    # (No need if you're consistent; the scaler will handle it anyway.)
    scale_to_bp: bool = False


def build_regime_features(rates_levels: pd.DataFrame, cfg: ClusterFeatureConfig) -> pd.DataFrame:
    """
    rates_levels: DataFrame of yield/rate LEVELS (date x tenor), e.g. in %.
    Returns: DataFrame (date x features).
    """

    r = rates_levels.copy()

    if cfg.scale_to_bp:
        r = r * 100.0  # % -> bp

    # Daily changes
    dr = r.diff()

    feats = {}

    # Level
    if cfg.use_level:
        feats["level"] = r[cfg.level_tenor]

    # Slope = long - short
    if cfg.use_slope:
        t_short, t_long = cfg.slope_tenors
        feats["slope"] = r[t_long] - r[t_short]

    # Curvature (common proxy): 2*belly - short - long
    if cfg.use_curvature:
        t_s, t_b, t_l = cfg.curvature_tenors
        feats["curvature"] = 2.0 * r[t_b] - r[t_s] - r[t_l]

    # Realized vol proxy on representative tenor change
    if cfg.use_realized_vol:
        feats["rv"] = dr[cfg.level_tenor].rolling(cfg.vol_window).std()

    F = pd.DataFrame(feats, index=r.index)

    # You can also add more sophisticated features if you want:
    # - PCA1 score volatility, cross-tenor dispersion, butterfly spreads, etc.
    return F

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans


@dataclass
class KMeansRegimeClustering:
    n_clusters: int = 4
    random_state: int = 42
    n_init: int = 20

    model_: Optional[Pipeline] = None
    label_names_: Optional[Dict[int, str]] = None  # optional pretty names

    def fit(self, F_train: pd.DataFrame) -> "KMeansRegimeClustering":
        """
        Fit clustering model on training feature matrix.
        """
        pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=self.n_init
            ))
        ])
        pipe.fit(F_train.values)
        self.model_ = pipe
        self.label_names_ = {k: f"CL{k}" for k in range(self.n_clusters)}
        return self

    def predict_labels(self, F: pd.DataFrame) -> pd.Series:
        """
        Returns a regime label per date.
        """
        if self.model_ is None:
            raise RuntimeError("Clustering model not fit. Call fit() first.")

        lab = self.model_.predict(F.values)
        # map ints to stable string labels
        names = pd.Series(lab, index=F.index).map(self.label_names_)
        names.name = "regime"
        return names


from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture


@dataclass
class GMMRegimeClustering:
    n_components: int = 4
    random_state: int = 42
    covariance_type: str = "full"  # "diag" can be more stable with fewer data
    reg_covar: float = 1e-6

    model_: Optional[Pipeline] = None
    label_names_: Optional[Dict[int, str]] = None

    def fit(self, F_train: pd.DataFrame) -> "GMMRegimeClustering":
        gmm = GaussianMixture(
            n_components=self.n_components,
            random_state=self.random_state,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar
        )
        pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("gmm", gmm)
        ])
        pipe.fit(F_train.values)

        self.model_ = pipe
        self.label_names_ = {k: f"CL{k}" for k in range(self.n_components)}
        return self

    def predict_proba(self, F: pd.DataFrame) -> pd.DataFrame:
        if self.model_ is None:
            raise RuntimeError("Clustering model not fit. Call fit() first.")
        P = self.model_.predict_proba(F.values)  # (T, K)
        cols = [self.label_names_[k] for k in range(P.shape[1])]
        return pd.DataFrame(P, index=F.index, columns=cols)

    def predict_labels(self, F: pd.DataFrame) -> pd.Series:
        if self.model_ is None:
            raise RuntimeError("Clustering model not fit. Call fit() first.")
        lab = self.model_.predict(F.values)
        names = pd.Series(lab, index=F.index).map(self.label_names_)
        names.name = "regime"
        return names

# Integration 1:
# 1) Build features
feat_cfg = ClusterFeatureConfig(
    level_tenor="10Y",
    slope_tenors=("2Y", "10Y"),
    curvature_tenors=("2Y", "10Y", "30Y"),
    vol_window=60,
    scale_to_bp=False
)
F = build_regime_features(rates_levels, feat_cfg).dropna()

# 2) Choose training window to avoid leakage (example)
train_end = "2023-12-29"
F_train = F.loc[:train_end]

# 3) Fit KMeans
clusterer = KMeansRegimeClustering(n_clusters=4, random_state=42).fit(F_train)

# 4) Label regimes for all dates
regimes = clusterer.predict_labels(F)

# 5) Train your regime models
bundle = trainer.fit(X.loc[F.index], Y.loc[F.index], regimes, regime_cfg=None)  # regime_cfg unused here

# 6) Predict / hedge routing
yhat = trainer.predict_hard(bundle, X_live, regimes_live)


# Integration 2:
# 1) Features
F = build_regime_features(rates_levels, feat_cfg).dropna()
F_train = F.loc[:train_end]

# 2) Fit GMM
gmm = GMMRegimeClustering(n_components=4, random_state=42, covariance_type="full").fit(F_train)

# 3) Probabilities for all dates
regime_prob = gmm.predict_proba(F)

# 4) You can still train models per "most likely" regime (hard assignment for training)
regimes_hard = gmm.predict_labels(F)

bundle = trainer.fit(X.loc[F.index], Y.loc[F.index], regimes_hard, regime_cfg=None)

# 5) Soft prediction (mixture of regime models)
yhat_soft = trainer.predict_soft(bundle, X.loc[F.index], regime_prob)


# Wrapper:
def get_cluster_regimes(
    rates_levels: pd.DataFrame,
    method: str = "gmm",  # "kmeans" or "gmm"
    n: int = 4,
    train_end: Optional[str] = None,
    feat_cfg: Optional[ClusterFeatureConfig] = None,
):
    feat_cfg = feat_cfg or ClusterFeatureConfig()
    F = build_regime_features(rates_levels, feat_cfg).dropna()

    if train_end is None:
        # Default: fit on all available history (OK for pure descriptive clustering,
        # NOT OK for strict backtests). Better to set train_end in backtests.
        F_train = F
    else:
        F_train = F.loc[:train_end]

    if method.lower() == "kmeans":
        cl = KMeansRegimeClustering(n_clusters=n).fit(F_train)
        regimes = cl.predict_labels(F)
        return {"features": F, "regimes": regimes, "regime_prob": None, "clusterer": cl}

    if method.lower() == "gmm":
        cl = GMMRegimeClustering(n_components=n).fit(F_train)
        regimes = cl.predict_labels(F)
        prob = cl.predict_proba(F)
        return {"features": F, "regimes": regimes, "regime_prob": prob, "clusterer": cl}

    raise ValueError("method must be 'kmeans' or 'gmm'")
