# pca_report.py
from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors


# =============================================================================
# Data adapters (adjust these minimal parts if your objects differ)
# =============================================================================

def _to_series_explained_var(explained_var: Any, pc_names: List[str]) -> pd.Series:
    """
    Normalize explained variance to a pandas Series indexed by PC names.
    Accepts: numpy array, list, pd.Series.
    """
    if explained_var is None:
        return pd.Series(dtype=float)

    if isinstance(explained_var, pd.Series):
        # If already named PC1.., keep, else map.
        if explained_var.index.dtype == object and all(str(x).startswith("PC") for x in explained_var.index):
            return explained_var.astype(float)
        return pd.Series(explained_var.values, index=pc_names[: len(explained_var)], dtype=float)

    arr = np.asarray(explained_var, dtype=float).ravel()
    return pd.Series(arr, index=pc_names[: len(arr)], dtype=float)


def _pc_names_from_loadings(loadings: pd.DataFrame, n_pcs_fallback: int = 3) -> List[str]:
    if loadings is not None and isinstance(loadings, pd.DataFrame) and loadings.shape[1] > 0:
        return list(loadings.columns)
    return [f"PC{i+1}" for i in range(n_pcs_fallback)]


# =============================================================================
# Core metrics for PCA quality
# =============================================================================

def compute_reconstruction_error(
    data_matrix: pd.DataFrame,
    loadings: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.Series:
    """
    Reconstruct X_hat = scores @ loadings.T  (assuming columns match PCs)
    Returns per-date RMSE across tenors.
    """
    common_pcs = [pc for pc in scores.columns if pc in loadings.columns]
    if len(common_pcs) == 0:
        return pd.Series(index=data_matrix.index, dtype=float)

    x = data_matrix.astype(float)
    s = scores[common_pcs].astype(float)
    l = loadings[common_pcs].astype(float)

    x_hat = pd.DataFrame(s.values @ l.values.T, index=x.index, columns=x.columns)

    err = x - x_hat
    rmse = np.sqrt((err ** 2).mean(axis=1))
    rmse.name = "RMSE"
    return rmse


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return np.nan
    return float(np.dot(a, b) / denom)


def compute_loading_cosine_stability_over_time(loadings_by_date: Dict[pd.Timestamp, pd.DataFrame]) -> pd.DataFrame:
    """
    For each consecutive date, compute cosine similarity per PC between loading vectors.
    loadings_by_date: dict(date -> DataFrame index=tenors, columns=PCs)
    Returns DataFrame indexed by date with columns cosine_PC1, cosine_PC2, ...
    """
    dates = sorted(loadings_by_date.keys())
    if len(dates) < 2:
        return pd.DataFrame()

    pcs = list(loadings_by_date[dates[0]].columns)
    rows = []
    idx = []
    for t in range(1, len(dates)):
        d_prev, d_cur = dates[t - 1], dates[t]
        L_prev = loadings_by_date[d_prev]
        L_cur = loadings_by_date[d_cur]

        # Align tenors
        common_tenors = L_prev.index.intersection(L_cur.index)
        L_prev = L_prev.loc[common_tenors, pcs]
        L_cur = L_cur.loc[common_tenors, pcs]

        row = {}
        for pc in pcs:
            row[f"cosine_{pc}"] = cosine_similarity(L_prev[pc].values, L_cur[pc].values)
        rows.append(row)
        idx.append(d_cur)

    return pd.DataFrame(rows, index=pd.to_datetime(idx)).sort_index()


def subspace_distance_frobenius(V_prev: np.ndarray, V_cur: np.ndarray) -> float:
    """
    Computes || P_prev - P_cur ||_F where P = V V^T (projector on subspace)
    V matrices are (n_tenors x K) with orthonormal columns ideally.
    """
    P_prev = V_prev @ V_prev.T
    P_cur = V_cur @ V_cur.T
    return float(np.linalg.norm(P_prev - P_cur, ord="fro"))


def compute_subspace_stability_over_time(
    eigvecs_by_date: Dict[pd.Timestamp, np.ndarray],
    n_components: int,
) -> pd.Series:
    """
    eigvecs_by_date: dict(date -> eigenvectors matrix shape (n_tenors, n_pcs))
    returns Series of subspace distance for top-K between consecutive windows.
    """
    dates = sorted(eigvecs_by_date.keys())
    if len(dates) < 2:
        return pd.Series(dtype=float)

    distances = []
    idx = []
    for t in range(1, len(dates)):
        d_prev, d_cur = dates[t - 1], dates[t]
        V_prev = np.asarray(eigvecs_by_date[d_prev])[:, :n_components]
        V_cur = np.asarray(eigvecs_by_date[d_cur])[:, :n_components]

        # Orthonormalize defensively (QR)
        V_prev, _ = np.linalg.qr(V_prev)
        V_cur, _ = np.linalg.qr(V_cur)

        distances.append(subspace_distance_frobenius(V_prev, V_cur))
        idx.append(d_cur)

    s = pd.Series(distances, index=pd.to_datetime(idx), name=f"subspace_dist_top{n_components}")
    return s.sort_index()


# =============================================================================
# Plot helpers (matplotlib only)
# =============================================================================

def _savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_scree(explained_var: pd.Series, outpath: str, title: str) -> None:
    plt.figure(figsize=(8, 4))
    if explained_var.empty:
        plt.text(0.5, 0.5, "No explained variance available", ha="center", va="center")
        _savefig(outpath)
        return

    x = np.arange(1, len(explained_var) + 1)
    plt.plot(x, explained_var.values, marker="o")
    plt.xticks(x, explained_var.index, rotation=0)
    plt.title(title)
    plt.ylabel("Explained variance ratio")
    plt.xlabel("Principal components")
    plt.grid(True, alpha=0.3)
    _savefig(outpath)


def plot_cum_explained_var(explained_var: pd.Series, outpath: str, title: str) -> None:
    plt.figure(figsize=(8, 4))
    if explained_var.empty:
        plt.text(0.5, 0.5, "No explained variance available", ha="center", va="center")
        _savefig(outpath)
        return

    cum = explained_var.cumsum()
    x = np.arange(1, len(cum) + 1)
    plt.plot(x, cum.values, marker="o")
    plt.xticks(x, explained_var.index, rotation=0)
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.ylabel("Cumulative explained variance")
    plt.xlabel("Principal components")
    plt.grid(True, alpha=0.3)
    _savefig(outpath)


def plot_loadings_heatmap(loadings: pd.DataFrame, outpath: str, title: str) -> None:
    plt.figure(figsize=(9, 5))
    if loadings is None or loadings.empty:
        plt.text(0.5, 0.5, "No loadings available", ha="center", va="center")
        _savefig(outpath)
        return

    # Simple heatmap via imshow
    mat = loadings.values
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="Loading")

    plt.yticks(np.arange(loadings.shape[0]), loadings.index)
    plt.xticks(np.arange(loadings.shape[1]), loadings.columns)
    plt.title(title)
    plt.xlabel("PC")
    plt.ylabel("Tenor")
    _savefig(outpath)


def plot_loadings_curves(loadings: pd.DataFrame, outpath: str, title: str, pcs: Optional[List[str]] = None) -> None:
    plt.figure(figsize=(9, 4))
    if loadings is None or loadings.empty:
        plt.text(0.5, 0.5, "No loadings available", ha="center", va="center")
        _savefig(outpath)
        return

    pcs_to_plot = pcs if pcs is not None else list(loadings.columns[:3])
    x = np.arange(loadings.shape[0])

    for pc in pcs_to_plot:
        if pc in loadings.columns:
            plt.plot(x, loadings[pc].values, marker="o", label=pc)

    plt.xticks(x, loadings.index, rotation=45)
    plt.title(title)
    plt.xlabel("Tenor")
    plt.ylabel("Loading")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _savefig(outpath)


def plot_timeseries(df: pd.DataFrame, outpath: str, title: str, ylabel: str) -> None:
    plt.figure(figsize=(9, 4))
    if df is None or df.empty:
        plt.text(0.5, 0.5, "No timeseries available", ha="center", va="center")
        _savefig(outpath)
        return

    for col in df.columns:
        plt.plot(df.index, df[col].values, label=str(col))
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    _savefig(outpath)


def plot_series(s: pd.Series, outpath: str, title: str, ylabel: str) -> None:
    plt.figure(figsize=(9, 4))
    if s is None or s.empty:
        plt.text(0.5, 0.5, "No data available", ha="center", va="center")
        _savefig(outpath)
        return

    plt.plot(s.index, s.values)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    _savefig(outpath)


# =============================================================================
# PDF assembly (ReportLab)
# =============================================================================

def _draw_title(c: canvas.Canvas, text: str, y: float) -> float:
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, y, text)
    return y - 1.0 * cm


def _draw_paragraph(c: canvas.Canvas, text: str, y: float, max_width: float = 17.5 * cm) -> float:
    c.setFont("Helvetica", 10)
    words = text.split()
    line = ""
    for w in words:
        test = (line + " " + w).strip()
        if c.stringWidth(test, "Helvetica", 10) > max_width:
            c.drawString(2 * cm, y, line)
            y -= 0.5 * cm
            line = w
        else:
            line = test
    if line:
        c.drawString(2 * cm, y, line)
        y -= 0.7 * cm
    return y


def _draw_table(c: canvas.Canvas, data: List[List[str]], y: float) -> float:
    table = Table(data, colWidths=[6 * cm, 11 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    w, h = table.wrapOn(c, 18 * cm, 10 * cm)
    table.drawOn(c, 2 * cm, y - h)
    return y - h - 0.8 * cm


def _draw_image(c: canvas.Canvas, img_path: str, y: float, height: float = 7.2 * cm) -> float:
    if not os.path.exists(img_path):
        return y
    # Preserve aspect ratio by using a fixed height
    from PIL import Image

    im = Image.open(img_path)
    w, h = im.size
    aspect = w / h if h else 1.0
    width = height * aspect
    max_width = 18 * cm
    if width > max_width:
        width = max_width
        height = width / aspect

    c.drawImage(img_path, 2 * cm, y - height, width=width, height=height, preserveAspectRatio=True, mask="auto")
    return y - height - 0.8 * cm


# =============================================================================
# Main report generator
# =============================================================================

@dataclass
class PCAReportConfig:
    report_title: str = "PCA Quality Report"
    n_pcs_to_show: int = 5
    pcs_to_plot_loadings: int = 3
    out_dir: str = "pca_report_output"
    pdf_filename: str = "pca_quality_report.pdf"


class PCAReportGenerator:
    """
    Generates:
      - PNG charts
      - PDF report embedding the charts
    Works for:
      - Static PCAResult
      - Rolling PCA RollingResult with .results dict(date -> PCAResult)
    """

    def __init__(self, config: PCAReportConfig):
        self.config = config
        os.makedirs(self.config.out_dir, exist_ok=True)
        self.fig_dir = os.path.join(self.config.out_dir, "figures")
        os.makedirs(self.fig_dir, exist_ok=True)

    # ---------- Public API ----------

    def generate_from_static(
        self,
        pca_result: Any,
        market_data: Optional[pd.DataFrame] = None,
    ) -> str:
        """
        pca_result must have at least: loadings (DataFrame), explained_var (array-like), scores (DataFrame)
        market_data is optional, used for reconstruction error if provided.
        Returns path to PDF.
        """
        loadings: pd.DataFrame = getattr(pca_result, "loadings", None)
        scores: pd.DataFrame = getattr(pca_result, "scores", None)

        pc_names = _pc_names_from_loadings(loadings, n_pcs_fallback=self.config.n_pcs_to_show)
        explained_var = _to_series_explained_var(getattr(pca_result, "explained_var", None), pc_names)

        figures = []
        figures.append(self._make_scree_and_cum_plots(explained_var, prefix="static"))
        figures.append(self._make_loading_plots(loadings, prefix="static"))

        reconstruction_rmse = None
        if market_data is not None and loadings is not None and scores is not None:
            reconstruction_rmse = compute_reconstruction_error(market_data, loadings, scores)
            rmse_path = os.path.join(self.fig_dir, "static_reconstruction_rmse.png")
            plot_series(reconstruction_rmse, rmse_path, "Reconstruction RMSE (per date)", "RMSE")
            figures.append([rmse_path])

        summary = self._build_static_summary(explained_var, loadings, reconstruction_rmse)
        pdf_path = self._build_pdf(figures=figures, summary_table=summary, subtitle="Static PCA")
        return pdf_path

    def generate_from_rolling(
        self,
        rolling_result: Any,
        market_data: Optional[pd.DataFrame] = None,
        top_k_subspace: int = 3,
    ) -> str:
        """
        rolling_result.results must be a dict(date -> PCAResult)
        Each PCAResult should have: loadings, explained_var, scores, eigvecs (optional but recommended)
        market_data optional: if provided, reconstruction RMSE per window can be computed approximately,
        but requires aligning window scores/index with market_data.

        Returns path to PDF.
        """
        results_dict: Dict[pd.Timestamp, Any] = getattr(rolling_result, "results", None)
        if results_dict is None or len(results_dict) == 0:
            raise ValueError("rolling_result.results is empty or missing.")

        # Collect explained variance, loadings, scores over time
        dates = sorted(pd.to_datetime(list(results_dict.keys())))
        first_res = results_dict[dates[0]]

        first_loadings: pd.DataFrame = getattr(first_res, "loadings", None)
        pc_names = _pc_names_from_loadings(first_loadings, n_pcs_fallback=self.config.n_pcs_to_show)

        explained_var_panel = []
        loadings_by_date: Dict[pd.Timestamp, pd.DataFrame] = {}
        eigvecs_by_date: Dict[pd.Timestamp, np.ndarray] = {}

        for d in dates:
            res = results_dict[d]
            ev = _to_series_explained_var(getattr(res, "explained_var", None), pc_names)
            explained_var_panel.append(ev.rename(d))

            L = getattr(res, "loadings", None)
            if isinstance(L, pd.DataFrame) and not L.empty:
                loadings_by_date[d] = L.copy()

            V = getattr(res, "eigvecs", None)
            if V is not None:
                eigvecs_by_date[d] = np.asarray(V)

        explained_var_df = pd.DataFrame(explained_var_panel).sort_index()
        explained_var_df.index = pd.to_datetime(explained_var_df.index)

        figures = []

        # Explained variance stability (time series)
        exp_var_path = os.path.join(self.fig_dir, "rolling_explained_var_timeseries.png")
        plot_timeseries(
            explained_var_df.iloc[:, : self.config.n_pcs_to_show],
            exp_var_path,
            f"Explained variance ratio over time (top {self.config.n_pcs_to_show})",
            "Explained var ratio",
        )
        figures.append([exp_var_path])

        # Cumulative explained variance over time (top K)
        cum = explained_var_df.cumsum(axis=1)
        cum_k = cum.iloc[:, min(self.config.n_pcs_to_show, cum.shape[1]) - 1].rename("CumExpVarTopK")
        cum_path = os.path.join(self.fig_dir, "rolling_cum_expvar_topk.png")
        plot_series(cum_k, cum_path, f"Cumulative explained variance (top {self.config.n_pcs_to_show})", "Cum explained var")
        figures.append([cum_path])

        # Loadings stability: cosine similarity per PC
        if len(loadings_by_date) >= 2:
            cosine_df = compute_loading_cosine_stability_over_time(loadings_by_date)
            cos_path = os.path.join(self.fig_dir, "rolling_loading_cosine.png")
            plot_timeseries(
                cosine_df.iloc[:, : min(self.config.n_pcs_to_show, cosine_df.shape[1])],
                cos_path,
                "Loading stability (cosine similarity vs previous window)",
                "Cosine similarity",
            )
            figures.append([cos_path])

            # Also plot a few snapshot loading curves (early/mid/late)
            snapshot_paths = self._make_loading_snapshots(loadings_by_date, prefix="rolling")
            figures.append(snapshot_paths)

        # Subspace stability (recommended)
        if len(eigvecs_by_date) >= 2:
            subspace_s = compute_subspace_stability_over_time(eigvecs_by_date, n_components=top_k_subspace)
            subspace_path = os.path.join(self.fig_dir, f"rolling_subspace_dist_top{top_k_subspace}.png")
            plot_series(subspace_s, subspace_path, f"Subspace distance (top {top_k_subspace} PCs)", "Frobenius distance")
            figures.append([subspace_path])

        # Build summary table
        summary = self._build_rolling_summary(explained_var_df, loadings_by_date)
        pdf_path = self._build_pdf(figures=figures, summary_table=summary, subtitle="Rolling PCA")
        return pdf_path

    # ---------- Figure builders ----------

    def _make_scree_and_cum_plots(self, explained_var: pd.Series, prefix: str) -> List[str]:
        scree_path = os.path.join(self.fig_dir, f"{prefix}_scree.png")
        plot_scree(explained_var.iloc[: self.config.n_pcs_to_show], scree_path, f"Scree plot (top {self.config.n_pcs_to_show})")

        cum_path = os.path.join(self.fig_dir, f"{prefix}_cum_explained.png")
        plot_cum_explained_var(
            explained_var.iloc[: self.config.n_pcs_to_show],
            cum_path,
            f"Cumulative explained variance (top {self.config.n_pcs_to_show})",
        )
        return [scree_path, cum_path]

    def _make_loading_plots(self, loadings: pd.DataFrame, prefix: str) -> List[str]:
        heat_path = os.path.join(self.fig_dir, f"{prefix}_loadings_heatmap.png")
        plot_loadings_heatmap(loadings.iloc[:, : self.config.n_pcs_to_show] if loadings is not None else loadings,
                              heat_path,
                              f"Loadings heatmap (top {self.config.n_pcs_to_show})")

        curve_path = os.path.join(self.fig_dir, f"{prefix}_loadings_curves.png")
        pcs = list(loadings.columns[: self.config.pcs_to_plot_loadings]) if isinstance(loadings, pd.DataFrame) else None
        plot_loadings_curves(loadings, curve_path, f"Loadings curves (top {self.config.pcs_to_plot_loadings})", pcs=pcs)

        return [heat_path, curve_path]

    def _make_loading_snapshots(self, loadings_by_date: Dict[pd.Timestamp, pd.DataFrame], prefix: str) -> List[str]:
        dates = sorted(loadings_by_date.keys())
        pick = []
        if len(dates) >= 1:
            pick.append(dates[0])
        if len(dates) >= 3:
            pick.append(dates[len(dates) // 2])
        if len(dates) >= 2:
            pick.append(dates[-1])

        paths = []
        for d in pick:
            L = loadings_by_date[d]
            out = os.path.join(self.fig_dir, f"{prefix}_loadings_snapshot_{pd.Timestamp(d).date()}.png")
            pcs = list(L.columns[: self.config.pcs_to_plot_loadings])
            plot_loadings_curves(L, out, f"Loadings snapshot @ {pd.Timestamp(d).date()}", pcs=pcs)
            paths.append(out)
        return paths

    # ---------- Summary tables ----------

    def _build_static_summary(
        self,
        explained_var: pd.Series,
        loadings: Optional[pd.DataFrame],
        reconstruction_rmse: Optional[pd.Series],
    ) -> List[List[str]]:
        top = explained_var.iloc[: self.config.n_pcs_to_show] if not explained_var.empty else explained_var
        cum_top = float(top.cumsum().iloc[-1]) if len(top) else float("nan")

        rows = [
            ["Metric", "Value"],
            [f"Cumulative explained var (top {self.config.n_pcs_to_show})", f"{cum_top:.3f}" if np.isfinite(cum_top) else "n/a"],
        ]

        if loadings is not None and not loadings.empty:
            # Simple “level-like” heuristic: fraction of tenors with same sign as median for PC1
            pc1 = loadings.iloc[:, 0]
            sign_agreement = float((np.sign(pc1) == np.sign(np.nanmedian(pc1))).mean())
            rows.append(["PC1 sign agreement across tenors (level-likeness proxy)", f"{sign_agreement:.2%}"])

        if reconstruction_rmse is not None and not reconstruction_rmse.empty:
            rows.append(["Reconstruction RMSE (mean)", f"{reconstruction_rmse.mean():.6f}"])
            rows.append(["Reconstruction RMSE (95% quantile)", f"{reconstruction_rmse.quantile(0.95):.6f}"])

        return rows

    def _build_rolling_summary(
        self,
        explained_var_df: pd.DataFrame,
        loadings_by_date: Dict[pd.Timestamp, pd.DataFrame],
    ) -> List[List[str]]:
        rows = [["Metric", "Value"]]

        if explained_var_df is not None and not explained_var_df.empty:
            k = min(self.config.n_pcs_to_show, explained_var_df.shape[1])
            cum_topk = explained_var_df.iloc[:, :k].sum(axis=1)
            rows.append([f"Cum explained var top {k} (mean)", f"{cum_topk.mean():.3f}"])
            rows.append([f"Cum explained var top {k} (5% / 95%)", f"{cum_topk.quantile(0.05):.3f} / {cum_topk.quantile(0.95):.3f}"])

            # Eigenvalue ratio proxies if available as explained_var; otherwise skip
            if explained_var_df.shape[1] >= 2:
                ratio = explained_var_df.iloc[:, 0] / explained_var_df.iloc[:, 1].replace(0.0, np.nan)
                rows.append(["PC1/PC2 explained-var ratio (median)", f"{ratio.median():.3f}"])

        if len(loadings_by_date) >= 2:
            cosine_df = compute_loading_cosine_stability_over_time(loadings_by_date)
            if not cosine_df.empty:
                for col in cosine_df.columns[: min(3, cosine_df.shape[1])]:
                    rows.append([f"{col} (mean)", f"{cosine_df[col].mean():.3f}"])
                    rows.append([f"{col} (5% quantile)", f"{cosine_df[col].quantile(0.05):.3f}"])

        rows.append(["# Rolling windows", str(explained_var_df.shape[0]) if explained_var_df is not None else "n/a"])
        return rows

    # ---------- PDF builder ----------

    def _build_pdf(self, figures: List[List[str]], summary_table: List[List[str]], subtitle: str) -> str:
        pdf_path = os.path.join(self.config.out_dir, self.config.pdf_filename)
        c = canvas.Canvas(pdf_path, pagesize=A4)
        width, height = A4

        y = height - 2 * cm
        y = _draw_title(c, self.config.report_title, y)
        y = _draw_paragraph(c, subtitle, y)

        y = _draw_paragraph(
            c,
            "This report summarizes PCA quality using explained variance, loading interpretability, "
            "and rolling stability diagnostics. Use the stability charts to detect regime changes "
            "and avoid unstable hedge ratios.",
            y,
        )

        y = _draw_table(c, summary_table, y)

        # Figures pages
        for block in figures:
            c.showPage()
            y = height - 2 * cm
            y = _draw_title(c, "Charts", y)
            for img_path in block:
                y = _draw_image(c, img_path, y, height=7.2 * cm)
                if y < 4 * cm:
                    c.showPage()
                    y = height - 2 * cm

        c.save()
        return pdf_path


# =============================================================================
# Example usage
# =============================================================================
if __name__ == "__main__":
    # Example (pseudo) usage:
    #
    # from your_module import MarketDataManager, StaticPCA, RollingPCA, PCAConfig, RollingConfig
    # mkt_data = MarketDataManager(...).fetch_data()
    # rolling = RollingPCA(pca_config, roll_config).fit(mkt_data)
    #
    # gen = PCAReportGenerator(PCAReportConfig(report_title="EUR Curve PCA Report", out_dir="out"))
    # pdf_path = gen.generate_from_rolling(rolling, market_data=mkt_data, top_k_subspace=3)
    # print("Generated:", pdf_path)
    #
    pass
