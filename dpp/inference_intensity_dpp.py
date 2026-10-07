import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.model_selection import KFold
from scipy.special import ndtr
from joblib import Parallel, delayed

try:
    from .discrete_kde import discrete_kde_2d
    from .grid_helpers import assign_grid, grid_indices_to_physical_df, create_mask
except ImportError:
    from discrete_kde import discrete_kde_2d
    from grid_helpers import assign_grid, grid_indices_to_physical_df, create_mask

#######################################################################################

# Plot at 300dpi
mpl.rcParams['figure.dpi'] = 300


# Compute elementary symmetric polynomials E_k(lam) for k=0..Kmax
# CPU version (effectively the same as GPU version in inference_interaction_dpp.py)
def elementary_symmetric_all(lam, Kmax):
    E = np.zeros(Kmax + 1, dtype=np.float64)
    E[0] = 1.0
    for l in lam:
        for k in range(Kmax, 0, -1):
            E[k] += l * E[k-1]
    return E


# Precompute once per test set:
# counts[i] = how many times grid index i appears across all test samples
# N must match rho size
def _counts_of_indices(samples, N):
    flat = np.fromiter((i for s in samples for i in s), dtype=np.int64)
    return np.bincount(flat, minlength=N)


def log_likelihood_k(rho, samples, eps):
    loglam = np.log(rho + eps)
    k_list = [len(s) for s in samples]
    if len(k_list) == 0:
        return 0.0
    Kmax = int(max(k_list))

    E = elementary_symmetric_all(rho, Kmax)

    counts = _counts_of_indices(samples, len(rho))
    ll_det = float(np.dot(counts, np.log(rho + eps)))
    
    k_list = np.fromiter((len(s) for s in samples), dtype=np.int64)
    if k_list.size == 0:
        return 0.0
    Kmax = int(k_list.max())
    E = elementary_symmetric_all(rho, Kmax)

    hist_k = np.bincount(k_list, minlength=Kmax+1)  # hist_k[k] = #samples of size k
    ll_norm = float(np.sum(hist_k[1:] * np.log(np.maximum(E[1:], eps))))
    return ll_det - ll_norm


def log_likelihood(rho, samples, eps=1e-6):
    counts = _counts_of_indices(samples, len(rho))
    ll = float(np.dot(counts, np.log(rho + eps))) - len(samples) * float(np.sum(np.log1p(rho)))
    return ll


def convert_intensities(intens_2d, full_grid, clean_grid, fill_value=0.0):
    """
    Convert a (ny, nx) intensity evaluated on the *physical* grid centers
    to (out_2d, out_1d) aligned with the grid, regardless of any
    'x_rounded'/'y_rounded' columns that may be integer indices.

    Parameters
    ----------
    intens_2d : (ny, nx) array
        Output of `estimate_intensity`, which is ordered by sorted physical y, then x.
    full_grid : pd.DataFrame
        Must have columns ['x','y'] = physical centers (one row per cell).
    clean_grid : pd.DataFrame
        Target grid (can be same as full_grid). Must have ['x','y'].

    Returns
    -------
    out_2d : (ny*, nx*) array
        Square-rooted and mean-normalized intensities reshaped on clean_grid's unique y/x.
    out_1d : (ny* * nx*,) array
        Row-major flattening of out_2d.
    """
    # Physical axes that label `intens_2d`
    xs_full = np.sort(full_grid['x'].unique())
    ys_full = np.sort(full_grid['y'].unique())
    ny, nx = intens_2d.shape

    # Long form with physical coordinates
    I = (pd.DataFrame(intens_2d, index=ys_full, columns=xs_full)
           .rename_axis('y')
           .reset_index()
           .melt(id_vars='y', var_name='x', value_name='val'))

    # Join to target grid (by physical centers)
    if not {'x','y'}.issubset(clean_grid.columns):
        raise ValueError("clean_grid must have physical columns ['x','y'].")

    out = (clean_grid[['x', 'y']]
             .merge(I, on=['x', 'y'], how='left')
             .fillna({'val': fill_value}))

    # Reshape on clean_grid's sorted physical axes
    xs_clean = np.sort(clean_grid['x'].unique())
    ys_clean = np.sort(clean_grid['y'].unique())
    q_rect = (out.pivot(index='y', columns='x', values='val')
                .loc[ys_clean, xs_clean]
                .to_numpy())

    q_rect = np.sqrt(np.maximum(q_rect, 1e-16))
    out_2d = q_rect
    out_1d = out_2d.ravel(order='C')
    return out_2d, out_1d


# class to hold results of pilot bandwidth selection
class PilotResult:
    def __init__(self, best_h, cand_hs, ll_tests, intensity_2d, intensity_1d):
        self.best_h = best_h
        self.cand_hs = cand_hs
        self.ll_tests = ll_tests
        self.intensity_2d = intensity_2d
        self.intensity_1d = intensity_1d


def _eval_candidate(h, train_df, grid, test_samples, mode):
    img = estimate_intensity(
        data=train_df,
        grid=grid,
        bandwidth=h,
        mask=None
    )
    _, rho_1d = convert_intensities(img, grid, grid)
    if mode.lower() == 'k-dpp':
        ll = log_likelihood_k(rho_1d, test_samples, eps=1e-12)
    elif mode.lower() == 'dpp':
        ll = log_likelihood(rho_1d, test_samples, eps=1e-12)
    return ll


def _run_one_fold(
    train_idx,
    test_idx,
    samples,
    grid,
    cand_hs,
    mode,
    n_jobs_cands=1,
    df_phys=None,   # optional physical points
):
    """
    Run one CV fold: build training DataFrame, evaluate candidate bandwidths on test samples.

    - If df_phys is provided: training points are taken directly from df_phys.iloc[train_idx]
    - Otherwise: we assume samples are grid indices and map them to physical coordinates using grid_index->(x,y).
    """
    train_samples = [samples[i] for i in train_idx]
    test_samples = [samples[i] for i in test_idx]

    # Build train_df (physical coordinates)
    if df_phys is not None:
        # df_phys rows correspond to samples (one point per row for physical mode)
        train_df = df_phys.iloc[train_idx].reset_index(drop=True)
    else:
        # flatten for KDE training coordinates
        flat_train = np.fromiter((j for s in train_samples for j in s), dtype=np.int64)
        train_df = grid_indices_to_physical_df(flat_train, grid)

    # Evaluate candidates
    if n_jobs_cands != 1:
        ll_vec = np.array(
            Parallel(n_jobs=n_jobs_cands)(
                delayed(_eval_candidate)(h, train_df, grid, test_samples, mode.lower())
                for h in cand_hs
            ),
            dtype=np.float64,
        )
    else:
        ll_vec = np.zeros(len(cand_hs), dtype=np.float64)
        for j, h in enumerate(cand_hs):
            ll_vec[j] = _eval_candidate(h, train_df, grid, test_samples, mode.lower())

    return ll_vec


def select_bandwidth_discrete(
    samples,
    grid,
    candidate_hs,
    k_folds=5,
    mode="k-dpp",
    rng=None,
    verbose=False,
    n_jobs_folds=1,
    n_jobs_cands=1,
    samples_coords="grid",
):
    """
    Cross-validated bandwidth selection directly on the grid with diagonal DPP.

    Fixes:
    - Supports non-square (rectangular) grids
    - samples_coords="physical" uses physical points directly per fold.
    - Negative n_jobs_folds works (joblib convention).
    """
    samples = list(samples)
    cand_hs = [float(h) for h in candidate_hs]

    # Build representation depending on coordinate mode
    df_phys = None
    if samples_coords == "grid":
        # Expect samples as list of list of grid_index
        # We'll map indices to physical coordinates when needed inside each fold
        if verbose:
            print("samples_coords='grid': using grid_index samples.")
    elif samples_coords == "physical":
        # samples is an array-like of shape (n,2) of physical points
        arr = np.asarray(samples, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("samples_coords='physical' expects samples shape (n,2).")
        df_phys = pd.DataFrame({"x_position_mm": arr[:, 0], "y_position_mm": arr[:, 1]})

        # assign each point to nearest grid cell and store as singleton samples of grid_index
        df_gridpts = assign_grid(df_phys, grid)
        flat_idx = df_gridpts["grid_index"].astype(int).to_numpy()
        samples = [[i] for i in flat_idx]
    else:
        raise ValueError("samples_coords must be 'grid' or 'physical'.")

    if verbose:
        print(f"Evaluating {len(cand_hs)} candidates via {k_folds}-fold CV ({mode})...")

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=rng)

    # Decide whether to parallelize folds
    use_parallel_folds = (n_jobs_folds != 1)

    if use_parallel_folds:
        ll_tests_per_fold = Parallel(n_jobs=n_jobs_folds)(
            delayed(_run_one_fold)(
                train_idx,
                test_idx,
                samples=samples,
                grid=grid,
                cand_hs=cand_hs,
                mode=mode,
                n_jobs_cands=n_jobs_cands,
                df_phys=df_phys,  # only used in physical mode
            )
            for train_idx, test_idx in kf.split(samples)
        )
    else:
        ll_tests_per_fold = []
        for fold, (train_idx, test_idx) in enumerate(kf.split(samples), start=1):
            ll_vec = _run_one_fold(
                train_idx,
                test_idx,
                samples=samples,
                grid=grid,
                cand_hs=cand_hs,
                mode=mode,
                n_jobs_cands=n_jobs_cands,
                df_phys=df_phys,
            )
            ll_tests_per_fold.append(ll_vec)
            if verbose:
                best_j = int(np.argmax(ll_vec))
                print(f"Fold {fold}: max LL={ll_vec[best_j]:.4f} at h={cand_hs[best_j]:.4f}")

    # Aggregate across folds
    ll_mat = np.vstack(ll_tests_per_fold)  # (k_folds, n_cand)
    ll_sum = ll_mat.sum(axis=0)
    best_idx = int(np.argmax(ll_sum))
    best_h = cand_hs[best_idx]

    if verbose:
        print("\nSummed log-likelihood across folds:")
        for h, v in zip(cand_hs, ll_sum):
            print(f"h={h:.6g}  LL={v:.6f}")
        print(f"\nSelected bandwidth h* = {best_h:.6g}")

    # Final intensity on all data at best_h
    if df_phys is not None:
        all_df = df_phys
    else:
        # samples are grid indices; use all indices to build physical point set
        flat_all = np.fromiter((j for s in samples for j in s), dtype=np.int64)
        all_df = grid_indices_to_physical_df(flat_all, grid)

    img_best = estimate_intensity(data=all_df, grid=grid, bandwidth=best_h, mask=None)

    rho_best_2d, rho_best_1d = convert_intensities(img_best, grid, grid)

    return PilotResult(
        best_h=best_h,
        cand_hs=list(cand_hs),
        ll_tests=ll_tests_per_fold,
        intensity_2d=rho_best_2d,
        intensity_1d=rho_best_1d,
    )


def _edge_factor_gauss_rect(u, window, h):
    """
    Diggle's edge correction factor e(u) = ∫_W Kσ(u - v) dv on a rectangular window
    using a Gaussian kernel with bandwidth h. For rectangles, e(u) factorizes into
    the product of 1D CDF differences.
    
    Parameters
    ----------
    u: point at which to evaluate the edge factor
    window: rectangular window (xmin, ymin, xmax, ymax)
    h: bandwidth of the Gaussian kernel

    Returns
    -------
    sx * sy: edge correction factor
    """
    x, y = u[:, 0], u[:, 1]
    sx = (ndtr((window[2] - x) / h) - ndtr((window[0] - x) / h))
    sy = (ndtr((window[3] - y) / h) - ndtr((window[1] - y) / h))
    return sx * sy  # already accounts for the kernel normalisation


def _infer_rect_from_grid(xs, ys):
        """
        Infer (xmin, xmax, ymin, ymax) from near-regular grid centers.
        Uses half a median step as margin to approximate cell edges.
        """
        if len(xs) == 0 or len(ys) == 0:
            return (0.0, 1.0, 0.0, 1.0)
        dx = np.median(np.diff(xs)) if len(xs) > 1 else 1.0
        dy = np.median(np.diff(ys)) if len(ys) > 1 else 1.0
        return (xs[0] - 0.5 * dx, xs[-1] + 0.5 * dx,
                ys[0] - 0.5 * dy, ys[-1] + 0.5 * dy)


def estimate_intensity(
    data,
    grid,
    bandwidth="auto",
    mask=None,
    _eps=1e-10,
    window="rect",
    rect=None,
    normalize=False,
    edge_correction=False,
    verbose=False,
    candidate_hs=None
):
    """
    Estimate a 2D intensity on a rectilinear grid via Gaussian KDE.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns ['x_position_mm', 'y_position_mm'] with point coordinates.
    grid : pd.DataFrame
        Must contain columns ['x', 'y'] representing centers of grid cells.
        The grid should form an (ny * nx)-length DataFrame, one row per cell center.
    bandwidth : {"auto", float}, default "auto"
        KDE bandwidth. If "auto", uses cross-validated selection via `select_bandwidth_discrete`.
    mask : None, (ny, nx) bool array, or (ny*nx,) bool array, optional
        Cells flagged True will be set to zero after KDE evaluation.
    _eps : float, default 1e-10
        Small epsilon added after normalization to avoid exact zeros.
    window : {"rect", "poly"} 
        if poly, also requires poly argument, if rect specify rect
    rect : tuple (x_min, x_max, y_min, y_max), optional
        If window="rect", this specifies the rectangular window to consider for KDE.
         Only points within this rectangle will be used for KDE fitting and evaluation. Also needs to be specified if window="poly".
    normalize : bool, default False
        If True, normalize the intensity to sum to 1. If False, the intensity is unnormalized (but still non-negative).
    edge_correction : bool, default False
        If True, apply Diggle's edge correction factor to the KDE estimates. Only implemented for window="rect" so far.
    verbose : bool, default False
        If True, print progress and bandwidth selection information.
    candidate_hs : optional sequence of floats
        If provided, overrides the default bandwidth selection with a grid search over these candidates. Only used if bandwidth="auto".

    Returns
    -------
    img : (ny, nx) ndarray
        Normalized intensity array (sums to 1).
    """
    # Extract data points (N x 2)
    X = data[["x_position_mm", "y_position_mm"]].to_numpy(dtype=float)

    # Sort grid by (y, x) so reshape is consistent: rows = y, cols = x
    grid_sorted = grid.sort_values(["y", "x"]).reset_index(drop=True)
    grid_coords = grid_sorted[["x", "y"]].to_numpy(dtype=float)

    xs = np.sort(grid_sorted["x"].unique())
    ys = np.sort(grid_sorted["y"].unique())
    nx, ny = len(xs), len(ys)

    # Bandwidth selection
    if bandwidth == "auto":
        res = select_bandwidth_discrete(
            samples=X,
            grid=grid,
            candidate_hs = candidate_hs if candidate_hs is not None else np.logspace(-1, 1, 10),
            k_folds=5,
            n_jobs_folds=5,
            n_jobs_cands=1,
            samples_coords="physical",
            verbose=verbose,
            mode="k-DPP"
        )
        bandwidth_val = res.best_h

    else:
        bandwidth_val = float(bandwidth)
        if bandwidth_val <= 0:
            raise ValueError("`bandwidth` must be positive.")

    # Fit KDE and evaluate at grid centers
    XY_df = data[["x_position_mm", "y_position_mm"]].copy()
    XY_grid = assign_grid(XY_df, grid_sorted)
    vals = discrete_kde_2d(
        X=XY_grid['x_rounded'].to_numpy(dtype=int), 
        Y=XY_grid['y_rounded'].to_numpy(dtype=int), 
        nx=nx, ny=ny, 
        h=bandwidth_val
    ).flatten()

    if edge_correction:
        if window != "rect":
            raise NotImplementedError("Edge correction only implemented for window='rect'.")
        if rect is not None:
            rxmin, rxmax, rymin, rymax = rect
        else:
            rxmin, rxmax, rymin, rymax = _infer_rect_from_grid(
                np.sort(grid_sorted["x"].unique()),
                np.sort(grid_sorted["y"].unique())
            )
        rect_tuple = (float(rxmin), float(rxmax), float(rymin), float(rymax))
        e = _edge_factor_gauss_rect(grid_coords, rect_tuple, float(bandwidth_val))
        vals = vals / np.maximum(e, _eps)  # avoid division by zero

    # Reshape to (ny, nx)
    img = vals.reshape(ny, nx)

    # Apply mask if provided
    if mask is not None:
        # Build the same permutation used to sort the grid
        # (Use np.lexsort to get indices, then apply to both grid and mask.)
        sort_idx = np.lexsort((grid["x"].to_numpy(), grid["y"].to_numpy()))

        m = np.asarray(mask)

        if m.ndim == 1:
            if m.size != nx * ny:
                raise ValueError(f"1-D mask length {m.size} != nx*ny ({nx*ny}).")
            # Reorder the 1-D mask to match grid_sorted
            m_sorted = m[sort_idx]
            m2 = m_sorted.reshape(ny, nx)

        elif m.ndim == 2:
            # If a 2‑D mask is supplied, assume it is already in (ny, nx) sorted order.
            if m.shape != (ny, nx):
                raise ValueError(f"2-D mask shape {m.shape} != (ny, nx)=({ny}, {nx}).")
            m2 = m
        else:
            raise ValueError("`mask` must be 1-D (length nx*ny) or 2-D (ny, nx).")

        img = np.where(m2, 0.0, img)

    # Normalize to sum 1 and add epsilon
    if normalize:
        s = img.sum()
        if s <= 0 or not np.isfinite(s):
            # In extremely sparse paths, provide a uniform fallback over unmasked cells
            if mask is None:
                img = np.ones_like(img) / (nx * ny)
            else:
                valid = (~m2).astype(float)
                z = valid.sum()
                if z > 0:
                    img = valid / z
                else:
                    img = np.ones_like(img) / (nx * ny)
        else:
            img = img / s

    img = img + _eps  # avoid exact zeros downstream

    return img


def plot_intensities(intensities, grid=None, grid_dim=None, cmap='inferno', title=None):
    """
    Parameters
    ----------
    intensities : 1-D (nx*ny,) or 2-D (ny, nx) array
    grid       : DataFrame with columns ['x','y'] giving *centers* of cells (optional, recommended)
    grid_dim   : (nx, ny) if grid is not provided (fallback)
    """
    intens = np.asarray(intensities)

    # Infer nx, ny and reshape if needed
    if intens.ndim == 1:
        if grid is not None:
            xs = np.sort(grid['x'].unique())
            ys = np.sort(grid['y'].unique())
            nx, ny = len(xs), len(ys)
        elif grid_dim is not None:
            nx, ny = int(grid_dim[0]), int(grid_dim[1])
        else:
            raise ValueError("Provide either `grid` or `grid_dim` when intensities is 1-D.")

        if intens.size != nx * ny:
            raise ValueError(f"Cannot reshape: len(intensities)={intens.size} but nx*ny={nx*ny}.")
        img = intens.reshape(ny, nx)          # rows=y, cols=x
    elif intens.ndim == 2:
        img = intens
        ny, nx = img.shape
        xs = np.arange(nx) if grid is None else np.sort(grid['x'].unique())
        ys = np.arange(ny) if grid is None else np.sort(grid['y'].unique())
    else:
        raise ValueError("`intensities` must be 1-D or 2-D.")

    # Compute physical extents from grid centers if available, else index extents
    if grid is not None:
        xs = np.sort(grid['x'].unique())
        ys = np.sort(grid['y'].unique())

        # Convert centers to edges (assumes regular spacing)
        def centers_to_edges(c):
            c = np.asarray(c, dtype=float)
            if len(c) == 1:
                step = 1.0
                return np.array([c[0] - 0.5*step, c[0] + 0.5*step])
            steps = np.diff(c)
            step = np.median(steps)           # robust for minor noise
            return np.concatenate(([c[0] - step/2], (c[:-1] + c[1:]) / 2, [c[-1] + step/2]))

        x_edges = centers_to_edges(xs)
        y_edges = centers_to_edges(ys)
        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    else:
        extent = [0, nx, 0, ny]

    # Choose figure size that respects aspect (optional)
    aspect_ratio = (extent[3] - extent[2]) / (extent[1] - extent[0])
    width = 10.0
    height = width * aspect_ratio

    fig, ax = plt.subplots(figsize=(width, height), dpi=200, constrained_layout=True)
    im = ax.imshow(img, cmap=cmap, origin='lower', extent=extent, aspect='equal')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Density", fontsize=12)
    ax.tick_params(labelsize=10)
    if title:
        ax.set_title(title, fontsize=13)
    plt.show()

#######################################################################################

def main():
    data = pd.read_parquet("../data/snapshots.parquet")
    full_grid = pd.read_parquet("../data/grid_500mm.parquet")
    irregular_grid = pd.read_parquet("../data/clean_grid_500mm.parquet")
    mask = create_mask(full_grid, irregular_grid)
    print("data and grids loaded.")

    intensity = estimate_intensity(data, full_grid, bandwidth="auto", mask=mask, verbose=False, normalize=True, edge_correction=False)

    np.save(f"../data/intensity.npy", intensity)

if __name__ == "__main__":
    main()