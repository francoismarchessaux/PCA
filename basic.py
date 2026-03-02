from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import mean_squared_error


@dataclass
class PCADiagnostics:
    explained_variance_raw: np.ndarray
    explained_variance_ratio_raw: np.ndarray
    cumulative_explained_variance_raw: np.ndarray
    explained_variance_final_total: float
    reconstruction_mse_total: float
    reconstruction_rmse_by_tenor: pd.Series
    explained_variance_by_tenor: pd.Series
    residual_variance_by_tenor: pd.Series


class GuidedSparsePCARatesEngine:
    """
    Guided sparse PCA engine for USD rates curve changes.

    Universe:
        Any tenor grid supplied as DataFrame columns, e.g.
        ['2Y', '3Y', ..., '15Y', '20Y', '25Y', '30Y', '40Y', '50Y']

    Principal hedge tenors:
        Default: ['2Y', '5Y', '10Y', '30Y']

    Main workflow implemented:
        3. Compute covariance/correlation structure
        4. Run unconstrained PCA
        5. Retain top K factors
        6. Inspect raw PCA factors
        7. Define hedge tenor basis
        8. Map full tenor universe to hedge tenor basis
        9. Impose sparsity
        10. Translate into hedgeable trade structures
        11. Normalize factors
        12. Compute factor scores
        13. Reconstruct curve
        14. Measure diagnostics

    Notes:
        - Input X must already be daily changes.
        - Demeaning is assumed to be required.
        - Standardization is optional.
        - The sparse mapping is implemented via Lasso/ElasticNet.
        - "Trade factors" are constructed in anchor space, then lifted back
          to the full tenor universe through the full->anchor mapping.
    """

    def __init__(
        self,
        hedge_tenors: Optional[List[str]] = None,
        n_factors_raw: int = 3,
        standardize: bool = False,
        sparse_method: str = "lasso",
        sparse_alpha_universe_to_anchor: float = 1e-4,
        sparse_alpha_factor_to_anchor: float = 1e-4,
        elasticnet_l1_ratio: float = 0.95,
        use_trade_templates: bool = True,
        trade_templates: Optional[Dict[str, Dict[str, float]]] = None,
        normalize_factor_signs: bool = True,
        ridge_eps: float = 1e-10,
    ) -> None:
        self.hedge_tenors = hedge_tenors or ["2Y", "5Y", "10Y", "30Y"]
        self.n_factors_raw = n_factors_raw
        self.standardize = standardize
        self.sparse_method = sparse_method.lower()
        self.sparse_alpha_universe_to_anchor = sparse_alpha_universe_to_anchor
        self.sparse_alpha_factor_to_anchor = sparse_alpha_factor_to_anchor
        self.elasticnet_l1_ratio = elasticnet_l1_ratio
        self.use_trade_templates = use_trade_templates
        self.normalize_factor_signs = normalize_factor_signs
        self.ridge_eps = ridge_eps

        # Default hedgeable templates in anchor space.
        # These are only defaults and should be considered configurable.
        # Anchor order is hedge_tenors = ['2Y', '5Y', '10Y', '30Y']
        self.trade_templates = trade_templates or {
            "level": {"2Y": 1.0},
            "slope_2s30s": {"2Y": -1.0, "30Y": 1.0},
            "curvature_2s10s30s": {"2Y": -1.0, "10Y": 2.0, "30Y": -1.0},
        }

        # Fitted attributes
        self.columns_: Optional[List[str]] = None
        self.anchor_idx_: Optional[List[int]] = None

        self.mean_: Optional[pd.Series] = None
        self.std_: Optional[pd.Series] = None

        self.X_original_: Optional[pd.DataFrame] = None
        self.X_processed_: Optional[pd.DataFrame] = None

        self.covariance_: Optional[pd.DataFrame] = None

        self.eigenvalues_: Optional[np.ndarray] = None
        self.eigenvectors_: Optional[np.ndarray] = None

        self.raw_loadings_: Optional[pd.DataFrame] = None
        self.raw_scores_: Optional[pd.DataFrame] = None

        self.anchor_mapping_: Optional[pd.DataFrame] = None
        self.guided_anchor_factor_weights_: Optional[pd.DataFrame] = None
        self.guided_full_loadings_: Optional[pd.DataFrame] = None

        self.trade_anchor_weights_: Optional[pd.DataFrame] = None
        self.final_loadings_: Optional[pd.DataFrame] = None
        self.final_scores_: Optional[pd.DataFrame] = None
        self.reconstruction_: Optional[pd.DataFrame] = None
        self.residuals_: Optional[pd.DataFrame] = None

        self.diagnostics_: Optional[PCADiagnostics] = None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def fit(self, X: pd.DataFrame) -> "GuidedSparsePCARatesEngine":
        """
        Fit the full engine on daily curve changes X.

        Parameters
        ----------
        X : pd.DataFrame
            Rows = dates
            Columns = tenors
            Values = daily changes
        """
        self._validate_input(X)
        self.X_original_ = X.copy()
        self.columns_ = list(X.columns)
        self.anchor_idx_ = [self.columns_.index(t) for t in self.hedge_tenors]

        # Step 3: preprocess + covariance structure
        Xp = self._preprocess(X)
        self.X_processed_ = Xp
        self.covariance_ = self._compute_covariance(Xp)

        # Step 4-5: raw PCA + retain top K
        self._fit_raw_pca()

        # Step 6: raw factor inspection is accessible through raw_loadings_
        # Step 7-8-9: map full universe to anchors + sparse factor approximation
        self._fit_sparse_universe_to_anchor_mapping()
        self._fit_guided_sparse_factor_mapping()

        # Step 10-11: optional conversion to hedgeable trade structures + normalize
        self._build_final_factor_basis()

        # Step 12: factor scores
        self._compute_final_scores()

        # Step 13: reconstruction
        self._reconstruct_curve()

        # Step 14: diagnostics
        self._compute_diagnostics()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Project new daily changes onto the final factor basis.
        """
        self._check_is_fitted()
        Xp = self._apply_preprocess_with_fitted_params(X)
        scores = Xp.values @ self.final_loadings_.values
        return pd.DataFrame(
            scores,
            index=X.index,
            columns=list(self.final_loadings_.columns),
        )

    def reconstruct(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Reconstruct X from the final factor basis.
        If X is None, return in-sample reconstruction.
        """
        self._check_is_fitted()

        if X is None:
            return self.reconstruction_.copy()

        Xp = self._apply_preprocess_with_fitted_params(X)
        scores = Xp.values @ self.final_loadings_.values
        Xhat = scores @ self.final_loadings_.values.T
        return pd.DataFrame(Xhat, index=X.index, columns=X.columns)

    def get_raw_loadings(self) -> pd.DataFrame:
        self._check_is_fitted()
        return self.raw_loadings_.copy()

    def get_guided_loadings(self) -> pd.DataFrame:
        self._check_is_fitted()
        return self.guided_full_loadings_.copy()

    def get_final_loadings(self) -> pd.DataFrame:
        self._check_is_fitted()
        return self.final_loadings_.copy()

    def get_anchor_mapping(self) -> pd.DataFrame:
        self._check_is_fitted()
        return self.anchor_mapping_.copy()

    def get_diagnostics(self) -> PCADiagnostics:
        self._check_is_fitted()
        return self.diagnostics_

    def summary(self) -> pd.DataFrame:
        """
        Return a compact summary table of main diagnostics.
        """
        self._check_is_fitted()
        d = self.diagnostics_

        out = pd.DataFrame({
            "raw_explained_variance_ratio": d.explained_variance_ratio_raw,
            "raw_cumulative_explained_variance": d.cumulative_explained_variance_raw,
        }, index=self.raw_loadings_.columns)

        out.loc["FINAL_BASIS_TOTAL", "raw_explained_variance_ratio"] = d.explained_variance_final_total
        out.loc["FINAL_BASIS_TOTAL", "raw_cumulative_explained_variance"] = np.nan
        return out

    # ---------------------------------------------------------------------
    # Step 3: preprocessing and covariance
    # ---------------------------------------------------------------------

    def _validate_input(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if X.isnull().any().any():
            raise ValueError("X contains NaNs. Input must be clean.")
        missing = [t for t in self.hedge_tenors if t not in X.columns]
        if missing:
            raise ValueError(f"Missing hedge tenors in columns: {missing}")
        if self.n_factors_raw < 1:
            raise ValueError("n_factors_raw must be >= 1.")
        if self.n_factors_raw > X.shape[1]:
            raise ValueError("n_factors_raw cannot exceed number of tenors.")

    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        self.mean_ = X.mean(axis=0)
        Xp = X - self.mean_

        if self.standardize:
            std = Xp.std(axis=0, ddof=1).replace(0.0, 1.0)
            self.std_ = std
            Xp = Xp / std
        else:
            self.std_ = pd.Series(1.0, index=X.columns)

        return Xp

    def _apply_preprocess_with_fitted_params(self, X: pd.DataFrame) -> pd.DataFrame:
        if list(X.columns) != self.columns_:
            raise ValueError("X columns must match the fitted tenor universe exactly.")

        Xp = X - self.mean_
        if self.standardize:
            Xp = Xp / self.std_
        return Xp

    def _compute_covariance(self, Xp: pd.DataFrame) -> pd.DataFrame:
        cov = np.cov(Xp.values, rowvar=False, ddof=1)
        return pd.DataFrame(cov, index=Xp.columns, columns=Xp.columns)

    # ---------------------------------------------------------------------
    # Step 4-5: raw PCA
    # ---------------------------------------------------------------------

    def _fit_raw_pca(self) -> None:
        cov = self.covariance_.values

        # eigh because covariance matrix is symmetric
        eigvals, eigvecs = np.linalg.eigh(cov)

        # descending order
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # retain top K
        eigvals_k = eigvals[:self.n_factors_raw]
        eigvecs_k = eigvecs[:, :self.n_factors_raw]

        # PCA loadings convention:
        # columns are factors; rows are tenors
        factor_names = [f"PC{i+1}" for i in range(self.n_factors_raw)]

        self.eigenvalues_ = eigvals
        self.eigenvectors_ = eigvecs

        self.raw_loadings_ = pd.DataFrame(
            eigvecs_k,
            index=self.columns_,
            columns=factor_names,
        )

        raw_scores = self.X_processed_.values @ eigvecs_k
        self.raw_scores_ = pd.DataFrame(
            raw_scores,
            index=self.X_processed_.index,
            columns=factor_names,
        )

    # ---------------------------------------------------------------------
    # Step 7-8-9: sparse full-universe -> anchor mapping
    # ---------------------------------------------------------------------

    def _make_sparse_regressor(self, alpha: float):
        if self.sparse_method == "lasso":
            return Lasso(alpha=alpha, fit_intercept=False, max_iter=20000)
        elif self.sparse_method == "elasticnet":
            return ElasticNet(
                alpha=alpha,
                l1_ratio=self.elasticnet_l1_ratio,
                fit_intercept=False,
                max_iter=20000,
            )
        else:
            raise ValueError(f"Unknown sparse_method: {self.sparse_method}")

    def _fit_sparse_universe_to_anchor_mapping(self) -> None:
        """
        Regress each tenor series on the anchor tenor series:
            X_full ≈ X_anchor @ C
        where:
            X_full   : T x N
            X_anchor : T x M
            C        : M x N

        This creates a sparse mapping from the hedge tenor basis to the full universe.
        """
        Xp = self.X_processed_
        Xa = Xp[self.hedge_tenors].values  # T x M
        N = Xp.shape[1]
        M = Xa.shape[1]

        C = np.zeros((M, N))

        for j, tenor in enumerate(Xp.columns):
            y = Xp[tenor].values
            reg = self._make_sparse_regressor(self.sparse_alpha_universe_to_anchor)
            reg.fit(Xa, y)
            C[:, j] = reg.coef_

        self.anchor_mapping_ = pd.DataFrame(
            C,
            index=self.hedge_tenors,
            columns=Xp.columns,
        )

    def _fit_guided_sparse_factor_mapping(self) -> None:
        """
        Approximate each raw PCA loading vector using the hedge tenor basis.

        Let C be the anchor mapping such that:
            X ≈ X_anchor @ C

        A factor represented in anchor space by vector a (M,)
        lifts to full-universe loading approximately as:
            b_full ≈ C.T @ a

        So for each raw loading vector b_raw, solve:
            b_raw ≈ C.T @ a
        with sparse regression over a.
        """
        C = self.anchor_mapping_.values  # M x N
        design = C.T                     # N x M

        K = self.raw_loadings_.shape[1]
        M = C.shape[0]

        A = np.zeros((M, K))
        guided_full = np.zeros((len(self.columns_), K))

        for k, factor_name in enumerate(self.raw_loadings_.columns):
            y = self.raw_loadings_[factor_name].values
            reg = self._make_sparse_regressor(self.sparse_alpha_factor_to_anchor)
            reg.fit(design, y)
            a = reg.coef_
            A[:, k] = a
            guided_full[:, k] = design @ a

        factor_names = [f"GUIDED_{i+1}" for i in range(K)]

        self.guided_anchor_factor_weights_ = pd.DataFrame(
            A,
            index=self.hedge_tenors,
            columns=factor_names,
        )

        self.guided_full_loadings_ = pd.DataFrame(
            guided_full,
            index=self.columns_,
            columns=factor_names,
        )

        self.guided_full_loadings_ = self._orthonormalize_loadings(self.guided_full_loadings_)
        if self.normalize_factor_signs:
            self.guided_full_loadings_ = self._normalize_loading_signs(self.guided_full_loadings_)

    # ---------------------------------------------------------------------
    # Step 10-11: hedgeable factor translation + normalization
    # ---------------------------------------------------------------------

    def _build_trade_template_matrix(self) -> pd.DataFrame:
        """
        Build factor weights in anchor space from explicit trade templates.

        Example templates in anchor space:
            level             : {'2Y': 1.0}
            slope_2s30s       : {'2Y': -1.0, '30Y': 1.0}
            curvature_2s10s30s: {'2Y': -1.0, '10Y': 2.0, '30Y': -1.0}
        """
        out = pd.DataFrame(
            0.0,
            index=self.hedge_tenors,
            columns=list(self.trade_templates.keys()),
        )

        for factor_name, weights in self.trade_templates.items():
            for tenor, w in weights.items():
                if tenor not in out.index:
                    raise ValueError(
                        f"Trade template tenor '{tenor}' not in hedge_tenors={self.hedge_tenors}"
                    )
                out.loc[tenor, factor_name] = w

        return out

    def _build_final_factor_basis(self) -> None:
        """
        Final factor basis:
            - If use_trade_templates=True:
                build trade factors in anchor space, lift them to full universe
            - Else:
                use guided sparse PCA loadings directly
        """
        if self.use_trade_templates:
            trade_anchor = self._build_trade_template_matrix()
            self.trade_anchor_weights_ = trade_anchor.copy()

            C = self.anchor_mapping_.values  # M x N
            # Lift anchor-space trade factors to full-universe loadings
            # full_loadings = C.T @ trade_anchor
            full_loadings = C.T @ trade_anchor.values  # N x K_trade

            final_loadings = pd.DataFrame(
                full_loadings,
                index=self.columns_,
                columns=trade_anchor.columns,
            )
        else:
            self.trade_anchor_weights_ = None
            final_loadings = self.guided_full_loadings_.copy()

        final_loadings = self._orthonormalize_loadings(final_loadings)

        if self.normalize_factor_signs:
            final_loadings = self._normalize_loading_signs(final_loadings)

        self.final_loadings_ = final_loadings

    def _orthonormalize_loadings(self, loadings: pd.DataFrame) -> pd.DataFrame:
        """
        Euclidean orthonormalization of factor columns using QR.
        """
        Q, _ = np.linalg.qr(loadings.values)
        cols = list(loadings.columns)
        return pd.DataFrame(Q[:, :len(cols)], index=loadings.index, columns=cols)

    def _normalize_loading_signs(self, loadings: pd.DataFrame) -> pd.DataFrame:
        """
        Fix arbitrary sign flips for consistency.

        Heuristic:
            - level-like factor: positive average loading
            - slope-like factor: positive if 30Y loading > 2Y loading
            - curvature-like factor: positive if 10Y dominates average of 2Y/30Y
            - otherwise: positive largest-abs loading
        """
        out = loadings.copy()

        for col in out.columns:
            vec = out[col].copy()

            if "level" in col.lower():
                sgn = 1.0 if vec.mean() >= 0 else -1.0
            elif "slope" in col.lower() and {"2Y", "30Y"}.issubset(set(out.index)):
                sgn = 1.0 if vec.loc["30Y"] - vec.loc["2Y"] >= 0 else -1.0
            elif "curvature" in col.lower() and {"2Y", "10Y", "30Y"}.issubset(set(out.index)):
                belly = vec.loc["10Y"]
                wings = 0.5 * (vec.loc["2Y"] + vec.loc["30Y"])
                sgn = 1.0 if belly - wings >= 0 else -1.0
            else:
                largest = vec.abs().idxmax()
                sgn = 1.0 if vec.loc[largest] >= 0 else -1.0

            out[col] = sgn * vec

        return out

    # ---------------------------------------------------------------------
    # Step 12-13: scores and reconstruction
    # ---------------------------------------------------------------------

    def _compute_final_scores(self) -> None:
        scores = self.X_processed_.values @ self.final_loadings_.values
        self.final_scores_ = pd.DataFrame(
            scores,
            index=self.X_processed_.index,
            columns=self.final_loadings_.columns,
        )

    def _reconstruct_curve(self) -> None:
        Xhat = self.final_scores_.values @ self.final_loadings_.values.T
        self.reconstruction_ = pd.DataFrame(
            Xhat,
            index=self.X_processed_.index,
            columns=self.columns_,
        )
        self.residuals_ = self.X_processed_ - self.reconstruction_

    # ---------------------------------------------------------------------
    # Step 14: diagnostics
    # ---------------------------------------------------------------------

    def _compute_diagnostics(self) -> None:
        eigvals = self.eigenvalues_
        evr_raw = eigvals[:self.n_factors_raw] / eigvals.sum()
        cum_evr_raw = np.cumsum(evr_raw)

        X = self.X_processed_.values
        Xhat = self.reconstruction_.values
        resid = X - Xhat

        total_var = np.var(X, axis=0, ddof=1).sum()
        residual_var = np.var(resid, axis=0, ddof=1).sum()
        explained_var_final_total = 1.0 - residual_var / (total_var + self.ridge_eps)

        mse_total = mean_squared_error(X, Xhat)

        rmse_by_tenor = np.sqrt(np.mean((X - Xhat) ** 2, axis=0))
        var_by_tenor = np.var(X, axis=0, ddof=1)
        resid_var_by_tenor = np.var(resid, axis=0, ddof=1)
        explained_var_by_tenor = 1.0 - resid_var_by_tenor / (var_by_tenor + self.ridge_eps)

        self.diagnostics_ = PCADiagnostics(
            explained_variance_raw=eigvals[:self.n_factors_raw],
            explained_variance_ratio_raw=evr_raw,
            cumulative_explained_variance_raw=cum_evr_raw,
            explained_variance_final_total=float(explained_var_final_total),
            reconstruction_mse_total=float(mse_total),
            reconstruction_rmse_by_tenor=pd.Series(rmse_by_tenor, index=self.columns_),
            explained_variance_by_tenor=pd.Series(explained_var_by_tenor, index=self.columns_),
            residual_variance_by_tenor=pd.Series(resid_var_by_tenor, index=self.columns_),
        )

    # ---------------------------------------------------------------------
    # Utility
    # ---------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        attrs = [
            self.X_processed_,
            self.covariance_,
            self.raw_loadings_,
            self.anchor_mapping_,
            self.guided_full_loadings_,
            self.final_loadings_,
            self.final_scores_,
            self.reconstruction_,
            self.diagnostics_,
        ]
        if any(a is None for a in attrs):
            raise RuntimeError("Engine is not fitted yet. Call fit(X) first.")
