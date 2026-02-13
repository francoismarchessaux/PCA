import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class IPCADaily:
    date: pd.Timestamp
    tenors: list[str]
    sigma: np.ndarray           # implied vol vector
    corr: np.ndarray            # implied corr matrix
    cov: np.ndarray             # implied covariance Σ
    eigvals: np.ndarray         # λ sorted desc
    eigvecs: np.ndarray         # V columns sorted desc
    implied_pcs: np.ndarray     # V * sqrt(λ) (JPM appendix step 4)
    evr: np.ndarray             # explained variance ratios


def _make_psd(mat: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Eigenvalue clipping PSD projection."""
    w, V = np.linalg.eigh(0.5 * (mat + mat.T))
    w = np.maximum(w, eps)
    return (V * w) @ V.T


class JPMImpliedPCA:
    """
    Implements JPM i-PCA construction:
    Σ_ij = ρ_ij σ_i σ_j, eigen-decompose, EVR, and implied PCs = V * sqrt(λ).
    """

    def __init__(self, tenors: list[str], psd_clip: bool = True, eps: float = 1e-12):
        self.tenors = list(tenors)
        self.psd_clip = psd_clip
        self.eps = eps

    def build_cov(self, vol_row: pd.Series, corr_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sigma = vol_row.reindex(self.tenors).astype(float).values
        sigma = np.clip(sigma, self.eps, None)

        corr = corr_df.reindex(index=self.tenors, columns=self.tenors).astype(float).values
        corr = 0.5 * (corr + corr.T)
        np.fill_diagonal(corr, 1.0)

        # Σ = D * Corr * D
        D = np.diag(sigma)
        cov = D @ corr @ D
        cov = 0.5 * (cov + cov.T)

        if self.psd_clip:
            cov = _make_psd(cov, eps=self.eps)

        return sigma, corr, cov

    def decompose(self, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        w, V = np.linalg.eigh(cov)
        idx = np.argsort(w)[::-1]
        w = w[idx]
        V = V[:, idx]

        # JPM “implied principal components” scaling: V * sqrt(λ)
        implied_pcs = V * np.sqrt(np.maximum(w, self.eps))

        evr = w / max(np.trace(cov), self.eps)
        return w, V, implied_pcs, evr

    def run_one(self, date: pd.Timestamp, vol_row: pd.Series, corr_df: pd.DataFrame) -> IPCADaily:
        sigma, corr, cov = self.build_cov(vol_row, corr_df)
        eigvals, eigvecs, implied_pcs, evr = self.decompose(cov)
        return IPCADaily(date=date, tenors=self.tenors, sigma=sigma, corr=corr, cov=cov,
                        eigvals=eigvals, eigvecs=eigvecs, implied_pcs=implied_pcs, evr=evr)

    def run_panel(self, imp_vol: pd.DataFrame, imp_corr: dict[pd.Timestamp, pd.DataFrame]) -> dict[pd.Timestamp, IPCADaily]:
        out = {}
        for dt, vol_row in imp_vol.iterrows():
            if dt not in imp_corr:
                continue
            out[dt] = self.run_one(dt, vol_row, imp_corr[dt])
        return out


# Regime prediction model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def rolling_zscore(s: pd.Series, window: int = 252) -> pd.Series:
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0).replace(0.0, np.nan)
    return (s - mu) / sd

class JPMRegimePredictor:
    def __init__(self, n_regimes: int, z_window: int = 252, lags: int = 2):
        self.n_regimes = n_regimes
        self.z_window = z_window
        self.lags = lags
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=2000,
                C=1.0
            ))
        ])
        self.feature_cols_ = None

    def build_features(self, ipca_panel: dict[pd.Timestamp, IPCADaily]) -> pd.DataFrame:
        idx = sorted(ipca_panel.keys())
        evr1 = pd.Series({d: ipca_panel[d].evr[0] for d in idx}, name="evr1")
        evr2 = pd.Series({d: ipca_panel[d].evr[1] for d in idx}, name="evr2")

        d_evr1 = evr1.diff().rename("d_evr1")
        z_d_evr1 = rolling_zscore(d_evr1, self.z_window).rename("z_d_evr1")
        evr_gap = (evr1 - evr2).rename("evr1_minus_evr2")

        # entropy over first m EVRs (diffuse structure => higher entropy)
        m = 5
        entropy = []
        for d in idx:
            p = np.clip(ipca_panel[d].evr[:m], 1e-12, None)
            p = p / p.sum()
            entropy.append(-np.sum(p * np.log(p)))
        entropy = pd.Series(entropy, index=idx, name="evr_entropy_5")

        X = pd.concat([evr1, evr2, d_evr1, z_d_evr1, evr_gap, entropy], axis=1)

        # add a few lags (predictive in practice; also helps regime smoothing)
        for L in range(1, self.lags + 1):
            X = pd.concat([X, X.shift(L).add_suffix(f"_lag{L}")], axis=1)

        return X.dropna()

    def fit(self, X: pd.DataFrame, regime_labels: pd.Series) -> "JPMRegimePredictor":
        # predict next regime: y_{t+1} from X_t
        y_next = regime_labels.shift(-1).reindex(X.index)
        mask = y_next.notna()
        X_fit = X.loc[mask]
        y_fit = y_next.loc[mask].astype(int)

        self.feature_cols_ = list(X_fit.columns)
        self.model.fit(X_fit.values, y_fit.values)
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_cols_ is None:
            raise RuntimeError("Predictor not fitted.")
        probs = self.model.predict_proba(X[self.feature_cols_].values)
        classes = self.model.named_steps["clf"].classes_
        out = pd.DataFrame(0.0, index=X.index, columns=np.arange(self.n_regimes))
        out.loc[:, classes] = probs
        return out

    def predict_next_proba_from_latest(self, X: pd.DataFrame) -> pd.Series:
        P = self.predict_proba(X.iloc[[-1]])
        return P.iloc[0]

# integration:
# 1) build iPCA panel (or update incrementally)
ipca_engine = JPMImpliedPCA(tenors=tenor_universe, psd_clip=True)
ipca_panel = ipca_engine.run_panel(imp_vol_df, imp_corr_dict)

# 2) features
pred = JPMRegimePredictor(n_regimes=K, z_window=252, lags=2)
X = pred.build_features(ipca_panel)

# 3) train (rolling / expanding window recommended in production)
pred.fit(X, regime_labels)

# 4) predict next-day regime probabilities (latest row)
p_next = pred.predict_next_proba_from_latest(X)

# 5) hedge computation
# Option B: probability blend
h = sum(p_next[r] * hedge_model[r].compute_hedge(today_inputs) for r in range(K))

# The trigger check
z_d_evr1 < -2
