import numpy as np

from scipy.stats import norm
from numba import njit


@njit(fastmath=True)
def gaussian_pdf_numba(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)


@njit(fastmath=True)
def lcv_inner(D_loo, h_grid):
    """
    Compute leave-one-out log-likelihood for 1D continuous KDE.
    """
    n = D_loo.shape[0]
    m = len(h_grid)
    lcv_scores = np.empty(m)

    for k in range(m):
        h = h_grid[k]
        inv_h = 1.0 / h
        norm_const = inv_h / (n - 1)

        s = 0.0
        for i in range(n):
            # sum of leave-one-out kernels
            ks = 0.0
            for j in range(n - 1):
                z = D_loo[i, j] * inv_h
                ks += gaussian_pdf_numba(z)
            f_loo = norm_const * ks
            s += np.log(f_loo)
        lcv_scores[k] = s / n

    return lcv_scores


@njit(fastmath=True)
def pairwise_diffs2d(X):
    """
    Compute pairwise differences for 2D data.
    """
    n = X.shape[0]
    Dx = np.empty((n, n))
    Dy = np.empty((n, n))
    for i in range(n):
        xi0 = X[i, 0]
        xi1 = X[i, 1]
        for j in range(n):
            Dx[i, j] = xi0 - X[j, 0]
            Dy[i, j] = xi1 - X[j, 1]
    return Dx, Dy


@njit(fastmath=True)
def lcv_inner_2d_full(Dx, Dy, h_grid):
    """
    Compute leave-one-out log-likelihood for 2D continuous KDE.
    """
    n = Dx.shape[0]
    m = len(h_grid)
    scores = np.empty(m)

    for k in range(m):
        h = h_grid[k]
        inv_h = 1.0 / h
        inv_h2 = inv_h * inv_h

        s = 0.0
        for i in range(n):
            acc = 0.0
            for j in range(n):
                if i == j:
                    continue
                zx = Dx[i, j] * inv_h
                zy = Dy[i, j] * inv_h
                acc += gaussian_pdf_numba(zx) * gaussian_pdf_numba(zy) * inv_h2
            f_loo = acc / (n - 1)

            s += np.log(f_loo)
        scores[k] = s / n

    return scores


def lcv_continuous(X, h_grid, d=1, numba=False, max_n=5000, seed=1):
    """
    Likelihood cross-validation for bandwidth selection for continuous (Gaussian) KDEs.

    X : array of samples (either 1D or 2D)
    h_grid : array of candidate bandwidths
    d : dimension (1 or 2)
    numba : whether to use numba-optimized LOO computations
    max_n : if len(X) > max_n, randomly subsample max_n points before LCV. This prevents O(n^2) memory blowups.
    """
    X = np.asarray(X)
    n = len(X)

    # cap sample size to avoid n^2 allocations
    if (max_n is not None) and (n > max_n):
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_n, replace=False)
        X = X[idx]
        n = len(X)

    n = len(X)
    if d == 1:
        # Precompute pairwise differences
        D = X[:, None] - X[None, :]

        # Build leave-one-out difference matrix
        mask = ~np.eye(n, dtype=bool)
        if numba:
            D_loo = D[mask].reshape(n, n - 1)

            # Compute LCV scores via numba-optimized inner loop
            lcv_scores = lcv_inner(D_loo, np.asarray(h_grid))

            h_opt = h_grid[np.argmax(lcv_scores)]
            return h_opt, lcv_scores
        else:
            lcv_scores = []
            
            for h in h_grid:
                # Compute kernel matrix for this h
                K = norm.pdf(D / h) / h   # (n, n)
                
                # LOO density estimate for each i
                f_loo = (K[mask].reshape(n, n-1)).sum(axis=1) / (n - 1)
                
                # Log-likelihood
                lcv_scores.append(np.mean(np.log(f_loo)))

            h_opt = h_grid[np.argmax(lcv_scores)]
            return h_opt, lcv_scores
    elif d == 2:        
        if numba:
            Dx, Dy = pairwise_diffs2d(X)  # Numba-compiled
            scores = lcv_inner_2d_full(Dx, Dy, h_grid)

            h_opt = h_grid[np.argmax(scores)]
            return h_opt, scores
        else:
            D = X[:, None] - X[None, :]  # shape (n, n, 2)
            mask = ~np.eye(n, dtype=bool)  # Mask to remove diagonal (leave-one-out)
            lcv_scores = []
            for h in h_grid:
                Kx = norm.pdf(D[:, :, 0] / h) / h
                Ky = norm.pdf(D[:, :, 1] / h) / h
                K = Kx * Ky  # Product of kernels for x and y
                f_loo = (K[mask].reshape(n, n - 1)).sum(axis=1) / (n - 1)
                lcv_scores.append(np.mean(np.log(f_loo)))

            h_opt = h_grid[np.argmax(lcv_scores)]
            return h_opt, lcv_scores