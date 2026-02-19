import numpy as np

def align_eigenvectors_to_previous(
    current_eigenvectors: np.ndarray,   # shape (n_tenors, n_pcs)
    previous_eigenvectors: np.ndarray,  # shape (n_tenors, n_pcs)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align current eigenvectors to previous eigenvectors:
    - match components by maximum absolute cosine similarity
    - flip signs so dot(prev, curr) >= 0 for matched components

    Returns:
        aligned_eigenvectors, permutation_indices
    """
    if previous_eigenvectors is None:
        return current_eigenvectors, np.arange(current_eigenvectors.shape[1])

    n_components = min(previous_eigenvectors.shape[1], current_eigenvectors.shape[1])

    prev = previous_eigenvectors[:, :n_components]
    curr = current_eigenvectors[:, :n_components]

    # Normalize columns
    prev_norm = prev / (np.linalg.norm(prev, axis=0, keepdims=True) + 1e-12)
    curr_norm = curr / (np.linalg.norm(curr, axis=0, keepdims=True) + 1e-12)

    # Similarity matrix: abs cosine between all pairs
    similarity = np.abs(prev_norm.T @ curr_norm)  # (n_components, n_components)

    # Greedy matching (ok for small n_components); for 5-10 PCs this is fine.
    # If you go higher, use Hungarian algorithm.
    permutation = -np.ones(n_components, dtype=int)
    used = set()
    for i in range(n_components):
        j = int(np.argmax(similarity[i, :]))
        while j in used:
            similarity[i, j] = -np.inf
            j = int(np.argmax(similarity[i, :]))
        permutation[i] = j
        used.add(j)

    # Reorder
    curr_reordered = curr[:, permutation]

    # Sign fix: enforce dot(prev, curr) >= 0
    signs = np.sign(np.sum(prev_norm * curr_reordered, axis=0))
    signs[signs == 0] = 1.0
    aligned = curr_reordered * signs

    # Put back in a full matrix if needed
    out = current_eigenvectors.copy()
    out[:, :n_components] = aligned
    return out, permutation



# Where to plug it
# In your _eigen_decomposition() path:
# after eigvals, eigvecs = np.linalg.eigh(matrix) and sorting,
# do eigvecs, perm = align_eigenvectors_to_previous(eigvecs, prev_eigvecs),
# then proceed (loadings/scores).



#######################
### EWMA covariance ###
#######################

import numpy as np

def ewma_covariance_matrix(
    data_matrix: np.ndarray,  # shape (n_obs, n_tenors)
    half_life: float,
) -> np.ndarray:
    """
    Exponentially weighted covariance with RiskMetrics-style decay.
    half_life in observations (e.g., 20 days).
    """
    n_obs = data_matrix.shape[0]
    lam = np.exp(-np.log(2.0) / half_life)
    weights = (1 - lam) * lam ** np.arange(n_obs - 1, -1, -1)  # newest gets largest weight
    weights = weights / np.sum(weights)

    mean = np.sum(data_matrix * weights[:, None], axis=0)
    centered = data_matrix - mean
    cov = (centered * weights[:, None]).T @ centered
    cov = 0.5 * (cov + cov.T)
    return cov

"""
Then in _compute_matrix():
if matrix_method == "ewma" use that instead of np.cov / LedoitWolf.
This materially improves robustness around turning points, because the estimator “forgets” old regimes faster.
"""
