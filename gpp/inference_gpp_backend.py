import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from scipy.spatial import cKDTree
from scipy.special import ndtr
from joblib import Parallel, delayed
from sklearn.neighbors import KernelDensity

try:
    from gpp.lcv_gpp import lcv_continuous
except ImportError:
    from lcv_gpp import lcv_continuous

# ----------------------------------
# Constants
# ----------------------------------
EPS_SIN = 1e-12  # to avoid log(0) in dgs_s calculations

# ----------------------------------
# Window and grid setup
# ----------------------------------
class Window:
    # standard order: (xmin, ymin, xmax, ymax)
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def erode(self, r):
        if r <= 0:
            return self
        return Window(
            self.xmin + r,
            self.ymin + r,
            self.xmax - r,
            self.ymax - r
        )

    def contains(self, points):
        x, y = points[:, 0], points[:, 1]
        return (x >= self.xmin) & (x <= self.xmax) & (y >= self.ymin) & (y <= self.ymax)
    

def _as_window(w, configs=None, pad=0.):
    """ If w is already a Window, return it
    Otherwise, expect w as (xmin, ymin, xmax, ymax) and convert to Window
    Else, given a set of configurations, generate a window enclosing them 
    as the smallest rectangle (with optional padding)"""
    if w is not None:
        if isinstance(w, Window):
            return w
        else:
            return Window(*w)
    xs = np.concatenate([c[:, 0] for c in configs if len(c) > 0])
    ys = np.concatenate([c[:, 1] for c in configs if len(c) > 0])
    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)
    return Window(xmin - pad, ymin - pad, xmax + pad, ymax + pad)


class Grid:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny


def grid_points(window, grid):
    """return grid points on Window, as well as the area of the cells and their dimensions"""
    nx, ny = grid.nx, grid.ny
    dx = (window.xmax - window.xmin) / nx
    dy = (window.ymax - window.ymin) / ny
    if dx <= 0 or dy <= 0:
        raise ValueError("Invalid window dimensions for the given grid size.")
    xs = window.xmin + (np.arange(nx) + 0.5) * dx
    ys = window.ymin + (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(xs, ys)
    points = np.column_stack([X.ravel(), Y.ravel()])
    a_cell = dx * dy
    return points, a_cell, dx, dy


# ----------------------------------
# Inhomogeneous Intensity specification and evaluation
# ----------------------------------
class IntensitySpec:
    """
    type : 'function' or 'image'
    mode : 'offset' (adds the intensity to the predictor) or 'covariate' (adds a column to the design matrix, to estimate a coefficient)
    takes_log : if True, apply log to strictly positive values
    fn : callable(points) -> values for type='function'
    image : 2D array (ny, nx) for type='image'
    extent: (xmin, ymin, xmax, ymax) for type='image' mapping image to space
    """
    def __init__(self, type, mode="offset", takes_log=True, fn=None, image=None, extent=None):
        self.type = type
        self.mode = mode
        self.takes_log = takes_log
        self.fn = fn
        self.image = image
        self.extent = extent


def bilinear_sample(image, extent, points):
    """
    returns intensity value at each point given the intensity as an image
    
    :param image: image of intensity values
    :param extent: extent of image in space (xmin, ymin, xmax, ymax)
    :param points: points at which to sample the image
    """
    ny, nx = image.shape
    xmin, ymin, xmax, ymax = extent
    cx = (points[:, 0] - xmin) / max(xmax - xmin, 1e-10) * (nx - 1)
    ry = (points[:, 1] - ymin) / max(ymax - ymin, 1e-10) * (ny - 1)
    cx = np.clip(cx, 0, nx - 1 - 1e-6)
    ry = np.clip(ry, 0, ny - 1 - 1e-6)
    x0 = np.floor(cx).astype(int)
    x1 = x0 + 1
    y0 = np.floor(ry).astype(int)
    y1 = y0 + 1
    x1 = np.clip(x1, 0, nx - 1)
    y1 = np.clip(y1, 0, ny - 1)
    dx = cx - x0
    dy = ry - y0
    v00 = image[y0, x0]
    v10 = image[y0, x1]
    v01 = image[y1, x0]
    v11 = image[y1, x1]
    v0 = v00 * (1 - dx) + v10 * dx
    v1 = v01 * (1 - dx) + v11 * dx
    return v0 * (1 - dy) + v1 * dy


def intensity_terms(u_points, spec):
    """
    Returns (offset, covariate_column) either may be None.
    offset is added to the linear predictor, covariate is a column in the design matrix.
    
    :param u_points: quadrature points
    :param spec: Intensity specification
    """
    if spec is None:
        return None, None
    if spec.type == "function":
        assert spec.fn is not None, "Function must be provided for intensity of type 'function'"
        values = np.asarray(spec.fn(u_points), dtype=float).reshape(-1)
    elif spec.type == "image":
        assert spec.image is not None and spec.extent is not None, "Image and extent must be provided for intensity of type 'image'"
        values = bilinear_sample(spec.image, spec.extent, u_points)
    else:
        raise ValueError("Invalid intensity type. Must be 'function' or 'image'.")
    
    if spec.takes_log:
        values = np.log(np.clip(values, 1e-10, np.inf))

    if spec.mode == "offset":
        return values, None
    elif spec.mode == "covariate":
        return None, values
    else:
        raise ValueError("Invalid intensity mode. Must be 'offset' or 'covariate'.")
    

# ----------------------------------
# Estimation of the inhomogeneous intensity via KDE
# ----------------------------------
def _edge_factor_gauss_rect(u, window, h):
    """
    Diggle's edge correction factor e(u) = ∫_W Kσ(u - v) dv on a rectangular window
    using a Gaussian kernel with bandwidth h. For rectangles, e(u) factorizes into
    the product of 1D CDF differences.
    
    :param u: point at which to evaluate the edge factor
    :param window: rectangular window
    :param h: bandwidth of the Gaussian kernel
    :return: edge correction factor
    """
    x, y = u[:, 0], u[:, 1]
    sx = (ndtr((window.xmax - x) / h) - ndtr((window.xmin - x) / h))
    sy = (ndtr((window.ymax - y) / h) - ndtr((window.ymin - y) / h))
    return sx * sy  # already accounts for the kernel normalisation


def _select_bw_lcv(points, h_grid):
    """
    Wrapper for LCV bandwidth selection using Loader's algorithm, using our own implementation of LCV for continuous bandwidths.
    """
    h_opt, _ = lcv_continuous(points, h_grid, d=2, numba=True, max_n=2000)
    return h_opt


def estimate_poisson_kde_intensity(
        configs,
        window,
        grid,
        bandwidth=None,
        hs=None
):
    """
    Estimate the inhomogeneous intensity using a KDE approach.
    Bandwidth can be specified or is otherwise estimated using a Poisson-likelihood 
    cross-validation approach.
    
    :param configs: List of point configurations
    :param window: rectangular observation window (xmin, ymin, xmax, ymax) or Window
    :param grid: regular grid
    :param bandwidth: Bandwidth for the Gaussian kernel. If None, it will be estimated.
    :param rmax_mult: Multiplier of the bandwidth to define the maximum radius for KDE evaluation (default: 6.0 contains over 99.7% of Gaussian mass)
    :param npixel: Number of pixels per dimension for the internal grid used in bandwidth selection (default: 256)
    :param hs: array of bandwidths to consider for LCV bandwidth selection, if bandwidth is not provided
    :return: Description
    """
    # ensure correct window format
    W = _as_window(window, configs = configs, pad = 0.) if window is not None else _as_window(None, configs = configs, pad = 0.)
    pts = []
    for config in configs:
        if len(config):
            config_in = config[_as_window((W.xmin, W.ymin, W.xmax, W.ymax), [config]).contains(config)]
            if len(config_in):
                pts.append(config_in)
    pool = np.vstack(pts) if len(pts) else np.zeros((0, 2))
    if len(pool) == 0:
        # Empty baseline, return constant 0 image
        ny, nx = grid.ny, grid.nx
        img = np.zeros((ny, nx), dtype=float)
        return img, (W.xmin, W.ymin, W.xmax, W.ymax), 0.0
    
    # bandwidth selection
    if bandwidth is not None and bandwidth > 0:
        h = float(bandwidth)
    else:
        h = _select_bw_lcv(pool, hs)


    # Evaluate intensity on grid points (same u_points)
    u_points, _, _, _ = grid_points(W, grid)

    # Edge factor
    e_u = _edge_factor_gauss_rect(u_points, W, h)

    # Exact Gaussian KDE in compiled code
    kde = KernelDensity(kernel="gaussian", bandwidth=h, algorithm="kd_tree")
    kde.fit(pool)

    # score_samples gives log( (1/N) * sum K )
    log_d = kde.score_samples(u_points)
    dens_sum = np.exp(log_d) * len(pool)

    intensities = dens_sum / np.maximum(e_u, 1e-10)
    img = intensities.reshape(grid.ny, grid.nx)
    return img, (W.xmin, W.ymin, W.xmax, W.ymax), h


def make_baseline_intensity_spec(
        img,
        extent,
        mode="offset",
        takes_log=True
):
    """
    Wrap a raster as an IntensitySpec ready for fitting.
    
    :param img: intensity image
    :param extent: extent of intensity image in (xmin, ymin, xmax, ymax)
    :param mode: 'offset' or 'covariate'
    :param takes_log: Whether the intensity values are in log scale
    :return: Intensity specification object
    """
    return IntensitySpec(
        type="image",
        image=img.astype(float),
        extent=extent,
        mode=mode,
        takes_log=takes_log
    )


# -----------------------------------
# modified Diggle-Gates-Stibbard interaction
# -----------------------------------
class DGSOptions:
    def __init__(self, window=None, grid_nx=50, grid_ny=50, exclude_self=True):
        self.window = window
        self.grid_nx = grid_nx
        self.grid_ny = grid_ny
        self.exclude_self = exclude_self  # whether to exclude the point itself when u coincides with a point in x in the sufficient statistic


def dgs_delta_from_dist(dist, r, eps=EPS_SIN):
    # delta(d) = 2*log(sin(pi d / (2r))) for d<=r
    s = np.sin(np.pi * dist / (2.0 * r))
    s = np.clip(s, eps, None)
    return 2.0 * np.log(s)


def _dgs_correction_pairs_x_to_u(x_points, u_points, r):
    """
    Return flat arrays describing all pairs (i, j) with ||x_i - u_j|| <= r.
      owner[k] = i  (data index)
      u_idx[k] = j  (quadrature index)
      delta[k] = 2*log(sin(pi d/(2r)))
    Uses sparse_distance_matrix for vectorized neighbor enumeration.
    """
    if len(x_points) == 0 or len(u_points) == 0:
        return (np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64))

    Tx = cKDTree(x_points)
    Tu = cKDTree(u_points)

    # coo_matrix with row=data index, col=grid index, data=distance
    coo = Tx.sparse_distance_matrix(Tu, max_distance=r, output_type='coo_matrix')

    owner = coo.row.astype(np.int64, copy=False)
    u_idx = coo.col.astype(np.int64, copy=False)
    dist  = coo.data.astype(np.float64, copy=False)

    delta = dgs_delta_from_dist(dist, r)
    return owner, u_idx, delta


def _dgs_s_corr_from_pairs(owner,
                           u_idx,
                           delta,
                           p,
                           alpha,
                           n_points):
    """
    Vectorized s_corr[i] = sum_{j in N(i)} p[j] * (exp(-alpha*delta_ij) - 1)
    implemented with expm1 + bincount.
    Keeps the same numerical guards as the original loop.
    """
    if owner.size == 0:
        return np.zeros(n_points, dtype=np.float64)

    exp_arg = -alpha * delta
    exp_arg = np.clip(exp_arg, -745.0, 700.0)
    term = np.expm1(exp_arg)
    term = np.clip(term, -1.0, 1e300)

    w = p[u_idx] * term
    s_corr = np.bincount(owner, weights=w, minlength=n_points).astype(np.float64)
    return s_corr


def dgs_s_fast(u_points, x_points, r, exclude_self=False):
    """
    Vectorized DGS sufficient statistic:
      stats[j] = sum_{i: ||u_j-x_i||<=r} 2*log(sin(pi d/(2r)))
    """
    m = len(u_points)
    if len(x_points) == 0 or m == 0:
        return np.zeros(m, dtype=float)

    Tu = cKDTree(u_points)
    Tx = cKDTree(x_points)

    # sparse distance matrix: rows = u index, cols = x index
    coo = Tu.sparse_distance_matrix(Tx, max_distance=r, output_type='coo_matrix')
    row = coo.row
    dist = coo.data

    # If exclude_self is requested and u_points==x_points (s_num case),
    # filter out zero-distance matches.
    if exclude_self:
        dist_mask = dist > 0.0
        row = row[dist_mask]
        dist = dist[dist_mask]

    delta = dgs_delta_from_dist(dist, r)  # vectorized

    stats = np.bincount(row, weights=delta, minlength=m).astype(float)
    return stats


# -----------------------------------
# Conditional log-pseudolikelihood for DGS
# Note: these are not fully optimized for speed, but can be useful for diagnostics
# -----------------------------------
def dgs_conditional_log_pseudolikelihood_fast(
        config,
        u_points,
        intensity,
        r,
        alpha,
        options,
        erode=True
):
    """
    Same as dgs_conditional_log_pseudolikelihood, but vectorizes the per-point correction.
    """
    n = len(config)
    if n == 0:
        return 0.0

    s_full = dgs_s_fast(u_points, config, r, exclude_self=False)
    s_num  = dgs_s_fast(config, config, r, exclude_self=True)

    # Baseline intensity terms (offset mode)
    u_intensity = intensity_terms(u_points, intensity)[0]
    x_intensity = intensity_terms(config, intensity)[0]

    W_full = _as_window(options.window, configs=[config], pad=0.)
    W_eroded = W_full.erode(r) if erode else W_full
    dx = (W_eroded.xmax - W_eroded.xmin) / options.grid_nx
    dy = (W_eroded.ymax - W_eroded.ymin) / options.grid_ny
    a_cell = dx * dy

    # Z0 and normalized weights p
    A0 = u_intensity + alpha * s_full
    A0_max = np.max(A0)
    expA0 = np.exp(A0 - A0_max)
    denom = a_cell * np.sum(expA0)
    Z0 = np.log(denom) + A0_max
    p = (a_cell * expA0) / denom

    # Vectorized correction pairs + bincount aggregation
    owner, u_idx, delta = _dgs_correction_pairs_x_to_u(config, u_points, r)
    s_corr = _dgs_s_corr_from_pairs(owner, u_idx, delta, p, alpha, n)

    s = 1.0 + s_corr
    s = np.maximum(s, 1e-300)
    Zi = Z0 + np.log(s)

    log_cpl = np.sum(x_intensity + alpha * s_num - Zi)
    return float(log_cpl)


def dgs_total_conditional_log_pseudolikelihood(
        configs,
        intensity,
        r,
        alpha,
        options,
        erode=True
):
    """
    Calculate total conditional log-pseudolikelihood for DGS interaction Gibbs point
    process model across multiple configurations.
    
    :param configs: list of point configurations
    :param intensity: intensity specification
    :param r: interaction radius 
    :param alpha: interaction strength
    :param options: DGS options, should include window, grid size and optionally erosion radius and whether to exclude self-interaction
    :param erode: whether to erode the window by r to avoid edge effects in the cache (default: True)
    :return: total conditional log-pseudolikelihood value across all configurations
    """
    
    W_full = _as_window(options.window, configs=configs, pad=0.)
    W_eroded = W_full.erode(r) if erode else W_full
    u_points, _, _, _ = grid_points(W_eroded, Grid(options.grid_nx, options.grid_ny))

    total_log_cpl = 0.0
    for config in configs:
        total_log_cpl += dgs_conditional_log_pseudolikelihood_fast(config, u_points, intensity, r, alpha, options, erode=erode)
    
    return total_log_cpl


# -----------------------------------
# Cached pseudolikelihood calculations for fast grid search
# -----------------------------------
class DGSConfigCache:
    # per-config cache for fixed r
    def __init__(self, config, u_intensity, x_intensity, s_full, s_num, idx_list, delta_list):
        self.config = config            # config of data points inside (eroded) window
        self.u_intensity = u_intensity  # intensity at quadrature points, shape (m,)
        self.x_intensity = x_intensity  # intensity at data points, shape (n,)
        self.s_full = s_full            # s(u_j; X) on grid (length M)
        self.s_num = s_num              # s(x_i; X\{x_i}) at data (length n)
        self.idx_list = idx_list        # grid neighbor indices for each data point
        self.delta_list = delta_list    # 2*log(sin(pi d / (2R))) per data point on its neighbors


class DGSCacheForR:
    # shared pieces across configs for given r
    def __init__(self, r, window_full, window_eroded, grid, u_points, a_cell, configs):
        self.r = r
        self.window_full = window_full
        self.window_eroded = window_eroded
        self.grid = grid
        self.u_points = u_points
        self.a_cell = a_cell
        self.configs = configs


def _prepare_cpl_dgs_for_r(
        configs,
        intensity,
        r,
        options,
        erode=True,
        pinned_grid=None,
        fixed_erosion_radius=None
):
    """
    Build all caches needed to evaluate the conditional log-pseudolikelihood 
    quickly for any alpha at a fixed r
    
    :param configs: list of point configurations
    :param intensity: intensity specification
    :param r: interaction radius
    :param options: DGS options, should include window, grid size and optionally erosion radius and whether to exclude self-interaction
    :param erode: whether to erode the window by r to avoid edge effects in the cache (default: True)
    :param pinned_grid: optional precomputed grid (u_points, a_cell, window_eroded) to use instead of computing it from the options and r.
    :param fixed_erosion_radius: if provided, use this fixed erosion radius instead of r for all caches.
    :return: cache object containing all precomputed values for given r
    """
    window_full = _as_window(options.window, configs=configs, pad=0.)
    if pinned_grid is not None:
        U_all, a_cell_full, window_ref = pinned_grid
        rr = (fixed_erosion_radius if (fixed_erosion_radius is not None) else r)
        if erode and rr > 0:
            # Mask points at least rr away from all borders
            mask = (
                (U_all[:, 0] >= window_ref.xmin + rr) &
                (U_all[:, 0] <= window_ref.xmax - rr) &
                (U_all[:, 1] >= window_ref.ymin + rr) &
                (U_all[:, 1] <= window_ref.ymax - rr)
            )
            u_points = U_all[mask]
            a_cell = float(a_cell_full) # keep cell size constant across r
            window_eroded = window_ref.erode(rr)
        else:
            u_points = U_all
            a_cell = float(a_cell_full)
            window_eroded = window_ref
        grid = Grid(options.grid_nx, options.grid_ny)
    else:
        grid = Grid(options.grid_nx, options.grid_ny)
        window_eroded = window_full.erode(r) if erode else window_full
        u_points, a_cell, _, _ = grid_points(window_eroded, grid)

    u_intensity = intensity_terms(u_points, intensity)[0]  # shape (m,)
    if u_intensity is None:
        u_intensity = np.zeros(len(u_points), dtype=float)
    
    cached_configs = []
    tree_grid = cKDTree(u_points)
    for config in configs:
        # restrict data to eroded window
        config_in = config[window_eroded.contains(config)] if len(config) else config
        n = len(config_in)
        x_intensity = intensity_terms(config_in, intensity)[0] if n > 0 else np.zeros(0) # shape (n,)
        if x_intensity is None:
            x_intensity = np.zeros(n, dtype=float)
        if n == 0:
            # empty config, cache minimal placeholders
            cached_configs.append(DGSConfigCache(
                config=config_in,
                u_intensity=u_intensity,
                x_intensity=x_intensity,
                s_full=np.zeros(len(u_points), dtype=float),
                s_num=np.zeros(0, dtype=float),
                idx_list=[],
                delta_list=[]
            ))
            continue
        s_full = dgs_s_fast(u_points, config_in, r, exclude_self=False)
        s_num = dgs_s_fast(config_in, config_in, r, exclude_self=True)
        idx_list, delta_list = [], []
        for xi in config_in:
            idx = np.array(tree_grid.query_ball_point(xi, r=r), dtype=int)
            if idx.size:
                dist = np.linalg.norm(u_points[idx] - xi, axis=1)
                s = np.clip(np.sin(np.pi * dist / (2 * r)), EPS_SIN, None)
                delta = 2. * np.log(s) # shape (len(idx),)
            else:
                delta = np.zeros(0, dtype=float)
            idx_list.append(idx)
            delta_list.append(delta)
        
        cached_configs.append(DGSConfigCache(
            config=config_in,
            u_intensity=u_intensity,
            x_intensity=x_intensity,
            s_full=s_full.astype(float),
            s_num=s_num.astype(float),
            idx_list=idx_list,
            delta_list=delta_list
        ))

    return DGSCacheForR(
        r=r,
        window_full=window_full,
        window_eroded=window_eroded,
        grid=grid,
        u_points=u_points,
        a_cell=float(a_cell),
        configs=cached_configs
    )


def _cpl_value_from_cache_for_alpha(cacheR, alpha):
    """
    Evaluate the conditional log-pseudolikelihood for all patterns at a fixed r and
    given alpha, using the precomputed cache.
    
    :param cacheR: cached values for given r
    :param alpha: interaction strength
    :return: conditional log-pseudolikelihood value for given alpha and cached r
    """
    total = 0.0
    a_cell = cacheR.a_cell

    for X in cacheR.configs:
        Xi = X.config
        n = len(Xi)
        if n == 0:
            continue
        A0 = X.u_intensity + alpha * X.s_full
        A0_max = np.max(A0)
        expA0 = np.exp(A0 - A0_max)
        denom = a_cell * np.sum(expA0)
        Z0 = np.log(denom) + A0_max
        p = (a_cell * expA0) / denom

        for i in range(n):
            idx = X.idx_list[i]
            delta = X.delta_list[i]
            # Numerically safe computation of exp(-alpha * delta) - 1
            exp_arg = -alpha * delta

            # Prevent overflow in exp for float64 (exp(>~709) overflows)
            exp_arg = np.clip(exp_arg, -745.0, 700.0)

            # expm1 is more accurate than exp(x)-1 for small |x|
            term = np.expm1(exp_arg)

            # Optional extra guard: avoid inf propagation in multiply/sum if term gets enormous
            term = np.clip(term, -1.0, 1e300)

            s_corr = np.sum(p[idx] * term) if idx.size else 0.0
            s = 1. + s_corr
            if s <= 1e-300:
                s = 1e-300 
            Zi = Z0 + np.log(s)
            total += (X.x_intensity[i] + alpha * X.s_num[i] - Zi)

    return float(total)


# -----------------------------------
# Grid search over alpha, r
# -----------------------------------
def dgs_grid_search(
        configs,
        intensity,
        r_values,
        alpha_values,
        options,
        erode=True,
        pin_grid=True,
        fixed_erosion_radius=None,
        n_jobs=1,
        prefer="processes",
        verbose=False
):
    """
    Profile conditional log-pseudolikelihood over (r, alpha).
    Returns dict with matrices and argmax
    
    :param configs: point configurations
    :param intensity: intensity specification
    :param r_values: interaction radii
    :param alpha_values: interaction strengths
    :param options: DGS process options
    :param erode: whether to erode the window by r to avoid edge effects (default: True)
    :param pin_grid: whether to pin the quadrature grid across different r values. If True, the grid from the first r value will be reused for all subsequent r values.
    :param fixed_erosion_radius: if provided, use this fixed erosion radius instead of r for all caches.
    :param n_jobs: number of parallel jobs
    :param prefer: parallelization preference ("processes" or "threads")
    :param verbose: verbosity flag
    :return: dictionary with results matrices and argmax
    """
    if n_jobs != 1:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    R = np.asarray(r_values, dtype=float)
    A = np.asarray(alpha_values, dtype=float)
    Z = np.empty((len(R), len(A)), dtype=float)

    pinned = None
    if pin_grid:
        window_full = _as_window(options.window, configs=configs, pad=0.)
        U_all, a_cell_full, _, _ = grid_points(window_full, Grid(options.grid_nx, options.grid_ny))
        pinned = (U_all, float(a_cell_full), window_full)

    for i, Ri in enumerate(R):
        if verbose:
            print(f"Preparing cache for r={Ri:.4f} ({i+1}/{len(R)})", flush=True)
        cacheR = _prepare_cpl_dgs_for_r(configs, intensity, Ri, options, erode=erode, pinned_grid=pinned, fixed_erosion_radius=fixed_erosion_radius)

        if n_jobs == 1:
            Z[i, :] = np.array([_cpl_value_from_cache_for_alpha(cacheR, alpha) for alpha in A], dtype=float)
        else:
            rows = Parallel(n_jobs=n_jobs, prefer=prefer)(
                delayed(_cpl_value_from_cache_for_alpha)(cacheR, alpha) for alpha in A
            )
            Z[i, :] = np.array(rows, dtype=float)

        if verbose:
            jmax = int(np.argmax(Z[i, :]))
            print(f"  Max at alpha={A[jmax]:.4f} with CPL={Z[i, jmax]:.4f}", flush=True)

    ij = np.unravel_index(int(np.argmax(Z)), Z.shape)
    out = dict(
        R = R,
        alpha = A,
        logcpl = Z,
        best_idx = ij,
        best_R = float(R[ij[0]]),
        best_alpha = float(A[ij[1]]),
        best_logcpl = float(Z[ij])
    )
    if verbose:
        print(f"Best overall at r={out['best_R']:.4f}, alpha={out['best_alpha']:.4f} with CPL={out['best_logcpl']:.4f}")

    return out


# ----------------------------------
# Strauss interaction
# ----------------------------------
class StraussOptions:
    def __init__(self, window=None, grid_nx=50, grid_ny=50, exclude_self=True):
        self.window = window
        self.grid_nx = grid_nx
        self.grid_ny = grid_ny
        self.exclude_self = exclude_self  # whether to exclude the point itself when u coincides with a point in x in the sufficient statistic


def strauss_s(u_points, x_points, r, exclude_self=True):
    """
    Sufficient statistic for a Gibbs point process with Strauss interaction:
    S(u;r) = sum_{||u - x_i|| <= r} 1
    Independent of interaction strength parameter alpha, which is then the
    coefficient of S in the linear predictor.
    
    :param u_points: quadrature points, shape (m, 2)
    :param x_points: data points from configuration, shape (n, 2)
    :param r: interaction radius
    :param exclude_self: whether to exclude the point itself when u coincides with a point in x
    :return: interaction term values at each u_point, shape (m,)
    """
    if r <= 0 or x_points is None or len(x_points) == 0:
        return np.zeros(len(u_points), dtype=float)

    tree = cKDTree(x_points)
    neigh = tree.query_ball_point(u_points, r=r)
    out = np.zeros(len(u_points), dtype=float)
    for i, idxs in enumerate(neigh):
        if not idxs:
            continue
        if exclude_self:
            idxs = [j for j in idxs if not np.allclose(u_points[i], x_points[j])]
        out[i] = float(len(idxs))
    return out


# ----------------------------------
# Conditional log-pseudolikelihood for Strauss
# Note: alpha is log(gamma)
# ----------------------------------
def strauss_conditional_log_pseudolikelihood(
    config,
    u_points,
    intensity,
    r,
    alpha,
    options,
    erode=True,
):
    """
    Calculate conditional log-pseudolikelihood for Strauss interaction Gibbs point
    process model for a single configuration.
    
    :param config: point configuration
    :param u_points: quadrature points
    :param intensity: intensity specification
    :param r: interaction radius 
    :param alpha: log interaction strength (log(gamma))
    :param options: Strauss options, should include window, grid size and optionally erosion radius and whether to exclude self-interaction
    :param erode: whether to erode the window by r to avoid edge effects in the cache (default: True)
    :return: conditional log-pseudolikelihood value
    """
    # Stat on grid including all points
    s_full = strauss_s(u_points, config, r, exclude_self=False)
    # Stat at data points with self excluded
    s_num = strauss_s(config, config, r, exclude_self=True)

    tree_grid = cKDTree(u_points)
    idx_list, delta_list = [], []
    for xi in config:
        idx = np.array(tree_grid.query_ball_point(xi, r=r), dtype=int)
        # for strauss, each nearby grid point contributes 1 when x_i is present
        delta = np.ones(idx.size, dtype=float) if idx.size else np.zeros(0, dtype=float)
        idx_list.append(idx)
        delta_list.append(delta)

    # intensity terms
    u_intensity = intensity_terms(u_points, intensity)[0]
    x_intensity = intensity_terms(config, intensity)[0]

    # window / grid cell area
    W_full = _as_window(options.window, configs=[config], pad=0.)
    W_eroded = W_full.erode(r) if erode else W_full
    dx = (W_eroded.xmax - W_eroded.xmin) / options.grid_nx
    dy = (W_eroded.ymax - W_eroded.ymin) / options.grid_ny
    a_cell = dx * dy

    # denominator base (with all points present)
    A0 = u_intensity + alpha * s_full
    A0_max = np.max(A0)
    expA0 = np.exp(A0 - A0_max)
    denom = a_cell * np.sum(expA0)
    Z0 = np.log(denom) + A0_max
    p = (a_cell * expA0) / denom

    # per point conditional normalizers and CPL
    log_cpl = 0.0
    for i in range(len(config)):
        idx = idx_list[i]
        delta = delta_list[i]
        # E[exp(-alpha * delta)] under p
        # Numerically safe computation of exp(-alpha * delta) - 1
        exp_arg = -alpha * delta

        # Prevent overflow in exp for float64 (exp(>~709) overflows)
        exp_arg = np.clip(exp_arg, -745.0, 700.0)

        # expm1 is more accurate than exp(x)-1 for small |x|
        term = np.expm1(exp_arg)

        # Optional extra guard: avoid inf propagation in multiply/sum if term gets enormous
        term = np.clip(term, -1.0, 1e300)

        s_corr = np.sum(p[idx] * term) if idx.size else 0.0
        s = 1. + s_corr
        if s <= 1e-300:
            s = 1e-300 
        Zi = Z0 + np.log(s)
        log_cpl += (x_intensity[i] + alpha * s_num[i] - Zi)
    return float(log_cpl)


def strauss_total_conditional_log_pseudolikelihood(
    configs,
    intensity,
    r,
    alpha,
    options,
    erode=True,
):
    """
    Calculate total conditional log-pseudolikelihood for Strauss model.
    
    :param configs: list of point configurations
    :param intensity: intensity specification
    :param r: interaction radius 
    :param alpha: log interaction strength (log(gamma))
    :param options: Strauss options, should include window, grid size and optionally erosion radius and whether to exclude self-interaction
    :param erode: whether to erode the window by r to avoid edge effects in the cache (default: True)
    :return: total conditional log-pseudolikelihood value
    """
    W_full = _as_window(options.window, configs=configs, pad=0.)
    W_eroded = W_full.erode(r) if erode else W_full
    u_points, _, _, _ = grid_points(W_eroded, Grid(options.grid_nx, options.grid_ny))
    total = 0.0
    for config in configs:
        if len(config) == 0:
            continue
        total += strauss_conditional_log_pseudolikelihood(config, u_points, intensity, r, alpha, options, erode=erode)
    return float(total)


# ----------------------------------
# Cached CPL for fast grid search (Strauss)
# ----------------------------------
class StraussConfigCache:
    def __init__(self, config, u_intensity, x_intensity, s_full, s_num, idx_list, delta_list):
        self.config = config
        self.u_intensity = u_intensity
        self.x_intensity = x_intensity
        self.s_full = s_full
        self.s_num = s_num
        self.idx_list = idx_list
        self.delta_list = delta_list


class StraussCacheForR:
    def __init__(self, r, window_full, window_eroded, grid, u_points, a_cell, configs):
        self.r = r
        self.window_full = window_full
        self.window_eroded = window_eroded
        self.grid = grid
        self.u_points = u_points
        self.a_cell = a_cell
        self.configs = configs


def _prepare_cpl_strauss_for_r(
    configs,
    intensity,
    r,
    options,
    erode=True,
    pinned_grid=None,
    fixed_erosion_radius=None,
):
    """
    Build all caches needed to evaluate the conditional log-pseudolikelihood 
    quickly for any alpha at a fixed r
    
    :param configs: list of point configurations
    :param intensity: intensity specification
    :param r: interaction radius
    :param options: Strauss options, should include window, grid size and optionally erosion radius and whether to exclude self-interaction
    :param erode: whether to erode the window by r to avoid edge effects in the cache (default: True)
    :param pinned_grid: optional precomputed grid (u_points, a_cell, window_eroded) to use instead of computing it from the options and r.
    :param fixed_erosion_radius: if provided, use this fixed erosion radius instead of r for all caches.
    :return: cache object containing all precomputed values for given r
    """
    window_full = _as_window(options.window, configs=configs, pad=0.)
    if pinned_grid is not None:
        U_all, a_cell_full, window_ref = pinned_grid
        rr = (fixed_erosion_radius if (fixed_erosion_radius is not None) else r)
        if erode and rr > 0:
            mask = (
                (U_all[:, 0] >= window_ref.xmin + rr) &
                (U_all[:, 0] <= window_ref.xmax - rr) &
                (U_all[:, 1] >= window_ref.ymin + rr) &
                (U_all[:, 1] <= window_ref.ymax - rr)
            )
            u_points = U_all[mask]
            a_cell = float(a_cell_full)
            window_eroded = window_ref.erode(rr)
        else:
            u_points = U_all
            a_cell = float(a_cell_full)
            window_eroded = window_ref
        grid = Grid(options.grid_nx, options.grid_ny)
    else:
        grid = Grid(options.grid_nx, options.grid_ny)
        window_eroded = window_full.erode(r) if erode else window_full
        u_points, a_cell, _, _ = grid_points(window_eroded, grid)

    u_intensity = intensity_terms(u_points, intensity)[0]
    if u_intensity is None:
        u_intensity = np.zeros(len(u_points), dtype=float)

    cached_configs = []
    tree_grid = cKDTree(u_points)

    for config in configs:
        config_in = config[window_eroded.contains(config)] if len(config) else config
        n = len(config_in)
        x_intensity = intensity_terms(config_in, intensity)[0] if n > 0 else np.zeros(0)
        if x_intensity is None:
            x_intensity = np.zeros(n, dtype=float)
        if n == 0:
            cached_configs.append(StraussConfigCache(
                config=config_in,
                u_intensity=u_intensity,
                x_intensity=x_intensity,
                s_full=np.zeros(len(u_points), dtype=float),
                s_num=np.zeros(0, dtype=float),
                idx_list=[],
                delta_list=[],
            ))
            continue
        s_full = strauss_s(u_points, config_in, r, exclude_self=False)
        s_num = strauss_s(config_in, config_in, r, exclude_self=True)

        idx_list, delta_list = [], []
        for xi in config_in:
            idx = np.array(tree_grid.query_ball_point(xi, r=r), dtype=int)
            delta = np.ones(idx.size, dtype=float) if idx.size else np.zeros(0, dtype=float)
            idx_list.append(idx)
            delta_list.append(delta)

        cached_configs.append(StraussConfigCache(
            config=config_in,
            u_intensity=u_intensity,
            x_intensity=x_intensity,
            s_full=s_full.astype(float),
            s_num=s_num.astype(float),
            idx_list=idx_list,
            delta_list=delta_list,
        ))

    return StraussCacheForR(
        r=r,
        window_full=window_full,
        window_eroded=window_eroded,
        grid=grid,
        u_points=u_points,
        a_cell=float(a_cell),
        configs=cached_configs,
    )


def _cpl_value_from_cache_for_alpha_strauss(cacheR, alpha):
    """
    Evaluate the conditional log-pseudolikelihood for all patterns at a fixed r and
    given alpha, using the precomputed cache.
    
    :param cacheR: cached values for given r
    :param alpha: log interaction strength (log(gamma))
    :return: conditional log-pseudolikelihood value for given alpha and cached r
    """
    total = 0.0
    a_cell = cacheR.a_cell
    for X in cacheR.configs:
        n = len(X.config)
        if n == 0:
            continue
        A0 = X.u_intensity + alpha * X.s_full
        A0_max = np.max(A0)
        expA0 = np.exp(A0 - A0_max)
        denom = a_cell * np.sum(expA0)
        Z0 = np.log(denom) + A0_max
        p = (a_cell * expA0) / denom
        for i in range(n):
            idx = X.idx_list[i]
            delta = X.delta_list[i]
            # Numerically safe computation of exp(-alpha * delta) - 1
            exp_arg = -alpha * delta

            # Prevent overflow in exp for float64 (exp(>~709) overflows)
            exp_arg = np.clip(exp_arg, -745.0, 700.0)

            # expm1 is more accurate than exp(x)-1 for small |x|
            term = np.expm1(exp_arg)

            # Optional extra guard: avoid inf propagation in multiply/sum if term gets enormous
            term = np.clip(term, -1.0, 1e300)

            s_corr = np.sum(p[idx] * term) if idx.size else 0.0
            s = 1.0 + s_corr
            if s <= 1e-300:
                s = 1e-300
            Zi = Z0 + np.log(s)
            total += (X.x_intensity[i] + alpha * X.s_num[i] - Zi)
    return float(total)


# ----------------------------------
# Grid search over (r, alpha) for Strauss
# ----------------------------------
def strauss_grid_search(
    configs,
    intensity,
    r_values,
    alpha_values,
    options,
    erode=True,
    pin_grid=True,
    fixed_erosion_radius=None,
    n_jobs=1,
    prefer="processes",
    verbose=False,
):
    """
    Profile conditional log-pseudolikelihood over (r, alpha).
    Returns dict with matrices and argmax
    
    :param configs: point configurations
    :param intensity: intensity specification
    :param r_values: interaction radii
    :param alpha_values: log interaction strengths
    :param options: Strauss process options
    :param erode: whether to erode the window by r to avoid edge effects (default: True)
    :param pin_grid: whether to pin the quadrature grid across different r values. If True, the grid from the first r value will be reused for all subsequent r values.
    :param fixed_erosion_radius: if provided, use this fixed erosion radius instead of r for all caches.
    :param n_jobs: number of parallel jobs
    :param prefer: parallelization preference ("processes" or "threads")
    :param verbose: verbosity flag
    :return: dictionary with results matrices and argmax
    """
    if n_jobs != 1:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    R = np.asarray(r_values, dtype=float)
    A = np.asarray(alpha_values, dtype=float)
    Z = np.empty((len(R), len(A)), dtype=float)

    pinned = None
    if pin_grid:
        window_full = _as_window(options.window, configs=configs, pad=0.)
        U_all, a_cell_full, _, _ = grid_points(window_full, Grid(options.grid_nx, options.grid_ny))
        pinned = (U_all, float(a_cell_full), window_full)

    for i, Ri in enumerate(R):
        if verbose:
            print(f"Preparing Strauss cache for r={Ri:.4f} ({i+1}/{len(R)})", flush=True)
        cacheR = _prepare_cpl_strauss_for_r(
            configs, intensity, Ri, options, erode=erode,
            pinned_grid=pinned, fixed_erosion_radius=fixed_erosion_radius
        )
        if n_jobs == 1:
            Z[i, :] = np.array([
                _cpl_value_from_cache_for_alpha_strauss(cacheR, alpha) for alpha in A
            ], dtype=float)
        else:
            rows = Parallel(n_jobs=n_jobs, prefer=prefer)(
                delayed(_cpl_value_from_cache_for_alpha_strauss)(cacheR, alpha) for alpha in A
            )
            Z[i, :] = np.array(rows, dtype=float)
        if verbose:
            jmax = int(np.argmax(Z[i, :]))
            print(f"  Max at alpha={A[jmax]:.4f} with CPL={Z[i, jmax]:.4f}", flush=True)

    ij = np.unravel_index(int(np.argmax(Z)), Z.shape)
    out = dict(
        R=R,
        alpha=A,
        logcpl=Z,
        best_idx=ij,
        best_R=float(R[ij[0]]),
        best_alpha=float(A[ij[1]]),
        best_logcpl=float(Z[ij]),
    )
    if verbose:
        print(
            f"Best Strauss overall at r={out['best_R']:.4f}, alpha={out['best_alpha']:.4f}, "
            f"gamma={np.exp(out['best_alpha']):.4f}, "
            f"with CPL={out['best_logcpl']:.4f}")
    return out


# ----------------------------------
# Helpers for data loading
# ----------------------------------
def df_to_configs(
    df,
    time_col="time_ms",
    x_col="x_position_mm",
    y_col="y_position_mm",
    id_col="object_identifier", # used only to deduplicate within a time if present
    keep="last", # which duplicate (per time,id) to keep: 'first'/'last'
    dropna=True,
    sort_time=True,
    min_points=0, # drop frames with < min_points (set 0 to keep all)
):
    """
    Build:
        - configs: [ np.array([[x,y], ...]), ... ] one per unique value of time_col
        - timestamps: [t1, t2, ...] raw time_col values, same order as configs
        - window: (xmin, ymin, xmax, ymax) inferred from all points (with small padding)    

    :param df: dataframe containing the data
    :param time_col: column for timestamps
    :param x_col: column for x coordinates
    :param y_col: column for y coordinates
    :param id_col: column for object IDs (used for deduplication within a time if present)
    :param keep: which duplicate (per time,id) to keep: 'first'/'last'
    :param dropna: whether to drop rows with missing coords/time
    :param sort_time: whether to sort by time
    :param min_points: drop frames with fewer than this many points (0 to keep all)
    :return: tuple of (configs, timestamps, window)
    """
    # time_col is only used as a frame key (and for ordering), so its raw values are used as-is
    d = df

    # Drop rows with missing coords/time if requested
    if dropna:
        d = d.dropna(subset=[time_col, x_col, y_col])

    # (Optional) de-duplicate within each frame by object_identifier
    # If the same object_identifier appears multiple times at the same datetime,
    # we keep the 'last' (or 'first') occurrence.
    if id_col in d.columns:
        d = d.sort_values([time_col, id_col]) # stable sort
        d = d.drop_duplicates(subset=[time_col, id_col], keep=keep)

    # Group by frame/time
    g = d.groupby(time_col, sort=sort_time)
    timestamps = []
    configs = []
    for ts, sub in g:
        pts = sub[[x_col, y_col]].to_numpy(dtype=float)
        if len(pts) < min_points:
            continue
        configs.append(pts)
        timestamps.append(ts)

    # Compute a reasonable window over all frames
    if any(len(c) > 0 for c in configs):
        all_pts = np.vstack([c for c in configs if len(c) > 0])
        xmin, ymin = np.min(all_pts, axis=0)
        xmax, ymax = np.max(all_pts, axis=0)
        # small padding so the edges don’t touch the border
        pad_x = 0.01 * (xmax - xmin if xmax > xmin else 1.0)
        pad_y = 0.01 * (ymax - ymin if ymax > ymin else 1.0)
        window = (float(xmin - pad_x), float(ymin - pad_y),
                  float(xmax + pad_x), float(ymax + pad_y))
    else:
        # No points at all (rare); fall back to unit square
        window = (0.0, 0.0, 1.0, 1.0)

    return configs, timestamps, window


# ----------------------------------
# Diagnostic plots
# ----------------------------------

def plot_intensity_heatmap(
    img,
    extent, # (xmin, ymin, xmax, ymax)
    points=None,
    title=None,
    cmap="viridis",
    log=False,
    vmin=None,
    vmax=None,
    save=None
):
    """
    Plot intensity heatmap with optional point overlay.

    :param img: intensity image
    :param extent: extent of intensity image in (xmin, ymin, xmax, ymax)
    :param points: point configuration in array of shape (n, 2)
    :param title: optional title
    :param cmap: color map for heatmap
    :param log: whether to use logarithmic scaling
    :param vmin: minimum value for color scaling
    :param vmax: maximum value for color scaling
    :param save: optional file path to save the plot
    """
    xmin, ymin, xmax, ymax = extent
    # Build Matplotlib extent in the order (left, right, bottom, top)
    mpl_extent = (xmin, xmax, ymin, ymax)
    width = 8.0
    aspect_ratio = (xmax - xmin) / max((ymax - ymin), 1e-12)
    height = width / aspect_ratio
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    # Optional log scaling done correctly (pass norm to imshow)
    norm = None
    if log:
        from matplotlib.colors import LogNorm
        pos = img[np.isfinite(img) & (img > 0)]
        if pos.size:
            vmin = float(np.percentile(pos, 1)) if vmin is None else vmin
            vmax = float(np.percentile(pos, 99)) if vmax is None else vmax
            norm = LogNorm(vmin=max(vmin, 1e-12), vmax=max(vmax, vmin + 1e-12))
        else:
            norm = LogNorm(vmin=1e-6, vmax=1.0)
    im = ax.imshow(
        img,
        extent=mpl_extent,
        origin='lower', # Cartesian: y increases upward
        cmap=cmap,
        norm=norm,
        vmin=vmin, vmax=vmax,
        aspect='equal'
    )
    if points is not None and len(points):
        ax.scatter(points[:,0], points[:,1], s=8, c="white", edgecolor="k", linewidths=0.3, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="Intensity")
    if save:
        plt.savefig(save, dpi=150)
    plt.show()


def plot_cpl_surface_with_ridge(
        R, A, Z,
        normalize='max',
        cmap='viridis',
        title='Conditional log-pseudolikelihood over (R, α)',
        annotate_best=True,
        ridge_style=None,
        contour_levels=12,
        figsize=(8, 5)
):
    """
    Plot conditional log-pseudolikelihood surface over (R, α) with ridgeline α*(R)
    and contour lines.

    :param R: vector of R values
    :param A: vector of α values
    :param Z: matrix of log-pseudolikelihood values, shape (len(R), len(A))
    :param normalize: 'none' | 'max' | 'rowmax'
    :param cmap: heatmap colormap
    :param title: optional title
    :param annotate_best: whether to mark global max
    :param ridge_style: dict for ridge plot styling
    :param contour_levels: number of contour levels or list of level values
    :param figsize: figure size tuple
    """

    R = np.asarray(R, float)
    A = np.asarray(A, float)
    Z = np.asarray(Z, float)

    # Handle normalization
    Zplot = Z.copy()
    if normalize == 'max':
        Zplot = Z - np.nanmax(Z)
    elif normalize == 'rowmax':
        row_max = np.nanmax(Z, axis=1, keepdims=True)
        Zplot = Z - row_max
    elif normalize == 'none':
        pass
    else:
        raise ValueError("normalize must be one of: 'none', 'max', 'rowmax'.")

    # Compute ridge α*(R)
    i_valid = np.where(np.any(np.isfinite(Z), axis=1))[0]
    i_star = np.full(len(R), -1, dtype=int)
    alpha_star = np.full(len(R), np.nan)
    z_star = np.full(len(R), np.nan)

    for i in i_valid:
        j = int(np.nanargmax(Z[i, :]))
        i_star[i] = j
        alpha_star[i] = A[j]
        z_star[i] = Z[i, j]

    # Global max
    if np.any(np.isfinite(Z)):
        iglob = int(np.nanargmax(Z))
        ir, ia = np.unravel_index(iglob, Z.shape)
        R_best, A_best, Z_best = R[ir], A[ia], Z[ir, ia]
    else:
        ir = ia = -1
        R_best = A_best = Z_best = np.nan

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    extent = (A.min(), A.max(), R.min(), R.max())
    im = ax.imshow(Zplot, origin='lower', aspect='auto',
                   extent=extent, cmap=cmap)

    # Contour lines
    AA, RR = np.meshgrid(A, R)
    # Focus contour levels around the ridge values
    z_min = np.nanpercentile(z_star, 10)
    z_max = np.nanpercentile(z_star, 100)

    contours = np.linspace(z_min, z_max, contour_levels)
    cs = ax.contour(
        AA, RR, Zplot,
        levels=contour_levels,
        colors='k',
        linewidths=0.8,
        alpha=0.8
    )
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2g")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    if normalize == 'none':
        cbar.set_label("log pseudolikelihood")
    elif normalize == 'max':
        cbar.set_label("log pseudolikelihood (minus global max)")
    else:
        cbar.set_label("log pseudolikelihood (minus row max)")

    # Ridge plot
    default_ridge_style = dict(color='white', lw=2.2, marker='o', ms=3.5,
                               mec='black', mew=0.6, alpha=0.95)
    if ridge_style:
        default_ridge_style.update(ridge_style)

    ridge_line, = ax.plot(alpha_star, R, **default_ridge_style)
    ridge_line.set_path_effects([
        pe.Stroke(linewidth=default_ridge_style.get('lw', 2.2) + 1.8,
                  foreground='black', alpha=0.5),
        pe.Normal()
    ])

    # Mark the global maximum
    if annotate_best and ir >= 0 and ia >= 0 and np.isfinite(Z_best):
        ax.scatter([A_best], [R_best], s=90, c='red',
                   edgecolor='white', linewidths=0.8, zorder=4)
        ax.text(A_best, R_best,
                f" max at R≈{R_best:.3g}, α≈{A_best:.3g}\nZ≈{Z_best:.4g}",
                color='red', fontsize=9, va='bottom', ha='left',
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    if "gamma" in title.lower():
        ax.set_xlabel("Gamma (exp(alpha))")
    else:
        ax.set_xlabel("Alpha")
    ax.set_ylabel("R")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()


def plot_cpl_surface_3d(R, A, Z, title='CPL surface (3D)'):
    """
    Plotting function that generates a 3D surface plot of the conditional log-pseudolikelihood over (R, α).
    
    :param R: vector of R values
    :param A: vector of α values
    :param Z: matrix of log-pseudolikelihood values
    :param title: optional title
    """
    Rg, Ag = np.meshgrid(R, A, indexing='ij')  # match Z[i,j] ↔ (R[i], A[j])
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Ag, Rg, Z, cmap='viridis', linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel("α")
    ax.set_ylabel("R")
    ax.set_zlabel("log pseudolikelihood")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


####################################################################################

def main():
    data = pd.read_parquet("../data/snapshots.parquet")
    window = (12000, 4000, 32000, 12000)
    data_cropped = data[(data.x_position_mm >= window[0]) & (data.x_position_mm <= window[2]) &
                        (data.y_position_mm >= window[1]) & (data.y_position_mm <= window[3])]
    configs, timestamps, _ = df_to_configs(data_cropped)

    # for testing (optional)
    no_configs = 10
    configs = configs[:no_configs]
    print(f"Using first {no_configs} frames for testing.")

    dgs_opts = DGSOptions(
        window=window,
        grid_nx=160,
        grid_ny=64
    )

    strauss_opts = StraussOptions(
        window=window,
        grid_nx=160,
        grid_ny=64
    )

    img, extent, h = estimate_poisson_kde_intensity(configs, window=window, grid=Grid(160, 64), bandwidth=None, hs=np.arange(100, 3000, 100))
    intensity = make_baseline_intensity_spec(img, extent, mode="offset", takes_log=True)
    plot_intensity_heatmap(img, extent, title=f"Estimated baseline intensity via KDE with bandwidth {h:.2f}", log=False)

    # -------------------------
    # Strauss
    # -------------------------
    res = strauss_grid_search(
        configs=configs,
        intensity=intensity,
        r_values=np.linspace(200, 2500, 47),
        alpha_values=np.log(np.arange(0.02, 1., 0.02)),
        options=strauss_opts,
        erode=False,
        pin_grid=True,
        fixed_erosion_radius=None,
        n_jobs=-4,
        prefer="processes",
        verbose=True
    )

    plot_cpl_surface_with_ridge(
        R=res['R'],
        A=np.exp(res['alpha'])
        ,
        Z=res['logcpl'],
        normalize='max',
        title='Strauss Conditional log-pseudolikelihood over (R, gamma)',
        annotate_best=True)
    
    plot_cpl_surface_3d(
        R=res['R'],
        A=np.exp(res['alpha']),
        Z=res['logcpl'],
        title='Strauss Conditional log-pseudolikelihood surface over (R, gamma) (3D)'
    )

    # -------------------------
    # DGS
    # -------------------------
    res = dgs_grid_search(
        configs=configs,
        intensity=intensity,
        r_values=np.linspace(200, 2500, 47),
        alpha_values=np.linspace(0.1, 5.0, 50),
        options=dgs_opts,
        erode=False,
        pin_grid=True,
        fixed_erosion_radius=None,
        n_jobs=-4,
        prefer="processes",
        verbose=True
    )

    plot_cpl_surface_with_ridge(
        R=res['R'],
        A=res['alpha'],
        Z=res['logcpl'],
        normalize='max',
        title='DGS Conditional log-pseudolikelihood over (R, α)',
        annotate_best=True)
    
    plot_cpl_surface_3d(
        R=res['R'],
        A=res['alpha'],
        Z=res['logcpl'],
        title='DGS Conditional log-pseudolikelihood surface (3D)'
    )


if __name__ == "__main__":
    main()

