import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def _safe_zscore(x: pd.Series, window: int = 252) -> pd.Series:
    mu = x.rolling(window).mean()
    sd = x.rolling(window).std(ddof=0)
    return (x - mu) / (sd.replace(0.0, np.nan))


@dataclass
class IPCADailyResult:
    cov: np.ndarray              # Σ_t
    eigvals: np.ndarray          # λ_t sorted desc
    eigvecs: np.ndarray          # columns are eigenvectors
    evr: np.ndarray              # explained variance ratios


class ImpliedPCABuilder:
    """
    Build implied covariance Σ_t from implied vols + implied correlations,
    then compute eigenvalues/eigenvectors + explained variance ratios (EVR).
    """

    def __init__(self, tenors: list[str], eps: float = 1e-12):
        self.tenors = list(tenors)
        self.eps = eps

    def build_cov(self, vol_row: pd.Series, corr_mat: pd.DataFrame) -> np.ndarray:
        v = vol_row.reindex(self.tenors).astype(float).values
        # guardrails
        v = np.clip(v, self.eps, None)

        C = corr_mat.reindex(index=self.tenors, columns=self.tenors).astype(float).values
        # enforce symmetry + diag=1
        C = 0.5 * (C + C.T)
        np.fill_diagonal(C, 1.0)

        # Σ = D C D
        D = np.diag(v)
        Sigma = D @ C @ D
        # numerical symmetry
        Sigma = 0.5 * (Sigma + Sigma.T)
        return Sigma

    def eigen(self, Sigma: np.ndarray) -> IPCADailyResult:
        # eigh for symmetric matrices
        w, V = np.linalg.eigh(Sigma)
        # sort descending
        idx = np.argsort(w)[::-1]
        w = w[idx]
        V = V[:, idx]

        tr = float(np.trace(Sigma))
        tr = max(tr, self.eps)
        evr = w / tr
        return IPCADailyResult(cov=Sigma, eigvals=w, eigvecs=V, evr=evr)

    def run(self, imp_vol: pd.DataFrame, imp_corr: dict[pd.Timestamp, pd.DataFrame]) -> dict[pd.Timestamp, IPCADailyResult]:
        out = {}
        for dt, vol_row in imp_vol.iterrows():
            if dt not in imp_corr:
                continue
            Sigma = self.build_cov(vol_row, imp_corr[dt])
            out[dt] = self.eigen(Sigma)
        return out


class JPMRegimePredictor:
    """
    Predict next-day regime probabilities from i-PCA diagnostics.
    Trains on your clustering labels (unsupervised -> used as targets).
    """

    def __init__(self, n_regimes: int, feature_lags: int = 1):
        self.n_regimes = n_regimes
        self.feature_lags = feature_lags

        # Multinomial logit is a good first pass: stable + interpretable.
        self.model = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(
                multi_class="multinomial",
                max_iter=2000,
                C=1.0,
                solver="lbfgs"
            ))
        ])

        self.feature_columns_: list[str] | None = None

    def build_features(self, ipca_results: dict[pd.Timestamp, IPCADailyResult]) -> pd.DataFrame:
        """
        Feature set centered on JPM statement:
        - EVR1 level and sharp declines in EVR1
        plus a few stabilizers.
        """
        idx = sorted(ipca_results.keys())
        evr1 = pd.Series({d: ipca_results[d].evr[0] for d in idx}, name="evr1")
        evr2 = pd.Series({d: ipca_results[d].evr[1] for d in idx}, name="evr2")

        d_evr1 = evr1.diff()
        z_d_evr1 = _safe_zscore(d_evr1, window=252)

        # concentration / "how one-factorish is the world?"
        spread12 = (evr1 - evr2)
        # a simple entropy measure using first m components
        m = 5
        ent = []
        for d in idx:
            p = np.clip(ipca_results[d].evr[:m], 1e-12, None)
            p = p / p.sum()
            ent.append(-np.sum(p * np.log(p)))
        ent = pd.Series(ent, index=idx, name="evr_entropy_5")

        X = pd.concat([
            evr1, evr2,
            d_evr1.rename("d_evr1"),
            z_d_evr1.rename("z_d_evr1"),
            spread12.rename("evr1_minus_evr2"),
            ent
        ], axis=1)

        # Optional: add lags (helps if regimes respond with small delay)
        if self.feature_lags > 0:
            for lag in range(1, self.feature_lags + 1):
                X = pd.concat([X, X.shift(lag).add_suffix(f"_lag{lag}")], axis=1)

        return X.dropna()

    def fit(self, X: pd.DataFrame, regime_labels: pd.Series):
        y = regime_labels.reindex(X.index)
        mask = y.notna()
        X_fit = X.loc[mask]
        y_fit = y.loc[mask].astype(int)

        if y_fit.nunique() < 2:
            raise ValueError("Need at least 2 regimes present in training sample.")

        self.feature_columns_ = list(X_fit.columns)
        self.model.fit(X_fit.values, y_fit.values)
        return self

    def predict_proba_next(self, X_today: pd.DataFrame) -> pd.Series:
        """
        Returns probability of each regime for *next* step.
        Here we assume X_today already corresponds to time t, and you're using it as predictor for t+1.
        """
        if self.feature_columns_ is None:
            raise RuntimeError("Model not fitted yet.")

        x = X_today[self.feature_columns_].iloc[-1:].values
        p = self.model.predict_proba(x)[0]
        # ensure full regime vector in fixed order 0..K-1
        classes = self.model.named_steps["clf"].classes_
        out = pd.Series(0.0, index=np.arange(self.n_regimes))
        out.loc[classes] = p
        return out


# --- Example: wiring it into your daily run ---

def train_regime_predictor(
    tenors: list[str],
    imp_vol: pd.DataFrame,
    imp_corr: dict[pd.Timestamp, pd.DataFrame],
    regime_labels: pd.Series,
    n_regimes: int
) -> tuple[ImpliedPCABuilder, JPMRegimePredictor, pd.DataFrame]:
    ipca = ImpliedPCABuilder(tenors=tenors)
    res = ipca.run(imp_vol=imp_vol, imp_corr=imp_corr)
    X = JPMRegimePredictor(n_regimes=n_regimes).build_features(res)

    pred = JPMRegimePredictor(n_regimes=n_regimes, feature_lags=1)
    pred.fit(X, regime_labels)
    return ipca, pred, X


def daily_regime_probabilities(
    ipca: ImpliedPCABuilder,
    pred: JPMRegimePredictor,
    imp_vol: pd.DataFrame,
    imp_corr: dict[pd.Timestamp, pd.DataFrame]
) -> pd.Series:
    res = ipca.run(imp_vol=imp_vol, imp_corr=imp_corr)
    X = pred.build_features(res)
    return pred.predict_proba_next(X)


# integration:
p_next = daily_regime_probabilities(ipca, pred, imp_vol, imp_corr)  # length K

# Option A: pick the most likely regime (hard)
r_star = int(p_next.idxmax())
h = hedge_model[r_star].compute_hedge(today_inputs)

# Option B: probability blend (soft / stabilizing)
h = 0.0
for r, pr in p_next.items():
    h += pr * hedge_model[int(r)].compute_hedge(today_inputs)
