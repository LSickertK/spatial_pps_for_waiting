import sys
import numpy as np
import pandas as pd
import cupy as cp
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

try: # for notebook
    from .kernels import generalized_gaussian
    from .inference_intensity_dpp import estimate_intensity, convert_intensities, select_bandwidth_discrete, plot_intensities
    from .inference_interaction_dpp import gaussian_grid_search, elementary_symmetric_all
    from .grid_helpers import create_mask, reindex_rect_grid, assign_grid
except ImportError:
    from kernels import generalized_gaussian
    from inference_intensity_dpp import estimate_intensity, convert_intensities, select_bandwidth_discrete, plot_intensities
    from inference_interaction_dpp import gaussian_grid_search, elementary_symmetric_all
    from grid_helpers import create_mask, reindex_rect_grid, assign_grid

################################################################

COEFFS = np.arange(0.9, 10.1, 0.1)
SIGMAS = np.arange(0.1, 5.0, 0.05)

# likelihoods
def log_det(matrix):
    try:
        return 2 * np.sum(np.log(np.diag(np.linalg.cholesky(matrix + 1e-6 * np.eye(matrix.shape[0], dtype=matrix.dtype)))))
    except np.linalg.LinAlgError:
            np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)
            print("Matrix not positive definite (on GPU)")
            print(matrix.get())
            return np.asarray(0.0, dtype=np.float64)


def log_likelihood(model, samples):
    T = len(samples)
    L_sigma_matrix = model.L().get_matrix()
    I = np.eye(model.N)
    log_det_I_plus_L = 2 * np.sum(
        np.log(np.diag(np.linalg.cholesky(L_sigma_matrix + I)))
    )
    log_det_L_part = -T * log_det_I_plus_L

    # cache submatrices once before loop
    sample_submatrices = [model.L().submatrix(samples[t]) for t in range(T)]

    log_likelihood_sum_values = []
    for t in range(T):
        log_likelihood_sum_values.append(log_det(sample_submatrices[t]))

    # sum results
    ll = log_det_L_part + np.sum(np.stack(log_likelihood_sum_values))
    return ll


# conditional version (on GPU)
def to_gpu(x, dtype=cp.float64):
    """Convert x to a CuPy array if it isn't already."""
    return x if isinstance(x, cp.ndarray) else cp.asarray(x, dtype=dtype)


def log_det_cp(matrix, jitter=1e-6):
    """
    Computes log(det(matrix)) via Cholesky on the GPU using CuPy.
    Adds a small jitter to the diagonal for numerical stability.
    Returns a CuPy scalar (0-d array).
    """
    m = to_gpu(matrix)
    n = m.shape[0]
    try:
        m_j = m.copy()
        idx = cp.arange(n)
        m_j[idx, idx] += jitter
        chol = cp.linalg.cholesky(m_j)
        return 2 * cp.sum(cp.log(cp.diagonal(chol)))
    except cp.linalg.LinAlgError:
        # Bring to CPU for readable printing
        np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)
        print("Matrix not positive definite (on GPU)")
        try:
            print(cp.asnumpy(m))
        except Exception:
            # Fallback if for some reason conversion fails
            print(m)
        return cp.asarray(0.0, dtype=cp.float64)

    
def log_likelihood_k_cp(model, samples):
    k_list = [len(s) for s in samples]
    unique_k = sorted(set(k_list))
    count_by_k = {k: k_list.count(k) for k in unique_k}

    # base matrix and eigendecomposition on GPU
    L_obj = model.L()
    L_full = to_gpu(L_obj.get_matrix())  # (N,N) CuPy array
    terms = []

    # Submatrix determinants: sum_t log det(L_{A_t})
    for s in samples:
        idx = cp.asarray(s, dtype=cp.int32)
        sub = L_full[idx[:, None], idx]
        terms.append(log_det_cp(sub))
    logdet_sum = cp.sum(cp.stack(terms)) if terms else cp.asarray(0.0, dtype=cp.float64)

    # Spectral normalizer: - sum_t log e_{k_t}(λ)
    lam = cp.linalg.eigvalsh(L_full)
    Kmax = int(max(unique_k))
    E, log_scale = elementary_symmetric_all(lam, Kmax, rescale=True)

    neg_logZ_terms = []
    for k in unique_k:
        # log e_k = log_scale + log(E[k])
        Ek = cp.clip(E[k], 1e-300, cp.inf)  # avoid log(0)
        log_ek = log_scale + cp.log(Ek)
        neg_logZ_terms.append(count_by_k[k] * log_ek)

    neg_logZ = cp.sum(cp.stack(neg_logZ_terms)) if neg_logZ_terms else cp.asarray(0.0, dtype=lam.dtype)

    return logdet_sum - neg_logZ


def evaluate_model_across_bandwidths(data_train, samples_test, cand_hs, sigma0, full_grid, irregular_grid, mode='dpp'):
    """
    Evaluate different candidate models, returning the log-likelihoods on the test set for each candidate bandwidth.
    """
    ll_tests = []
    mask = create_mask(full_grid, irregular_grid)
    for h in cand_hs:
        intensity_h = estimate_intensity(data_train, full_grid, bandwidth=h, mask=mask)
        _, intensity_h_1d = convert_intensities(intensity_h, full_grid, irregular_grid)
        model = generalized_gaussian(N=len(irregular_grid), d=2, rho=intensity_h_1d, sigma=sigma0, beta=2, window=irregular_grid)
        if mode == 'dpp':
            ll_test = log_likelihood(model, samples_test)
        elif mode == 'k-dpp':
            ll_test = log_likelihood_k_cp(model, samples_test)
        ll_tests.append(ll_test)
    return ll_tests


def evaluate_uniform(samples_test, sigma0, irregular_grid, mode='dpp'):
    """
    Evaluate the uniform model on the test set.
    """
    model = generalized_gaussian(N=len(irregular_grid), d=2, rho=1, sigma=sigma0, beta=2, window=irregular_grid)
    if mode == 'dpp':
        ll_test = log_likelihood(model, samples_test)
    elif mode == 'k-dpp':
        ll_test = log_likelihood_k_cp(model, samples_test)
    return ll_test


def reestimate_sigma_conditional(
    data_train,
    samples_train,
    full_grid,
    irregular_grid,
    h,
    sigmas
):
    """
    Reestimate the interaction parameter sigma using conditional likelihood on the training data.
    """
    intensity = estimate_intensity(data_train, full_grid, bandwidth=h, mask=create_mask(full_grid, irregular_grid))
    _, intensity_1d = convert_intensities(intensity, full_grid, irregular_grid)

    model = gaussian_grid_search(
        N=len(irregular_grid), d=2, intensity=intensity_1d, samples=samples_train, window=irregular_grid
    )
    best_sigmas, _ = model.fit(sigmas, mode="k-DPP", local_maxima=True, plot=True)
    # Identify interior candidates
    interior = [s for s in best_sigmas if s[0] != sigmas[0] and s[0] != sigmas[-1]]

    if interior:
        interior = np.array(interior)
        best_sigma = interior[np.argmax(interior[:, 1])][0] 
    else:
        print("Warning: no interior local maxima found for sigma, using boundary value.")
        best_sigma = best_sigmas[0][0]   # fallback to boundary (pick the available one)

    return best_sigma


def run_full_pipeline_for_fold(data_train, samples_train, samples_test, full_grid, irregular_grid, pilot_h=None, conditioned=False, sigma=3., check_uniform=False):
    """
    Run the full pipeline for a single fold of cross-validation.
    """
    # pilot bandwidth
    if pilot_h is None:
        print("Estimating pilot bandwidth...")
        rectangle_grid = reindex_rect_grid(full_grid)
        h = select_bandwidth_discrete(
            data_train[["x_position_mm", "y_position_mm"]].to_numpy(dtype=float), rectangle_grid, candidate_hs=np.arange(0.1, 10.1, 0.1), k_folds=10, mode='k-dpp', verbose=False, n_jobs_folds=-4, n_jobs_cands=1, samples_coords="physical"
        ).best_h
    else:
        h = pilot_h

    pilot_intensity = estimate_intensity(data_train, full_grid, bandwidth=h, mask=create_mask(full_grid, irregular_grid))
    plot_intensities(pilot_intensity, grid_dim=(full_grid['x'].nunique(), full_grid['y'].nunique()), title=f"Estimated Intensity at h={h:.4f}")
    
    # estimating interaction
    if sigma is not None:
        print(f"True sigma assumed to be {sigma}, no estimation necessary.")
    else:
        print("Estimating sigma via conditional likelihood on rectangle...")
        sigma = reestimate_sigma_conditional(
            data_train, samples_train, full_grid, irregular_grid, h, sigmas=SIGMAS
        )

    # candidate bandwidths
    cand_hs = [c * h for c in COEFFS]

    # choose h that maximizes test log-likelihood
    print("Evaluating candidate bandwidths...")
    if conditioned:
        ll_tests = evaluate_model_across_bandwidths(
        data_train, samples_test, cand_hs, sigma, full_grid, irregular_grid, mode='k-dpp'
        )
        if check_uniform:
            ll_uniform = evaluate_uniform(samples_test, sigma, irregular_grid, mode='k-dpp')
        else:
            ll_uniform = None
    else:
        ll_tests = evaluate_model_across_bandwidths(
        data_train, samples_test, cand_hs, sigma, full_grid, irregular_grid, mode='dpp'
        )
        if check_uniform:
            ll_uniform = evaluate_uniform(samples_test, sigma, irregular_grid, mode='dpp')
        else:
            ll_uniform = None

    return dict(
        pilot_h = h,
        ll_tests = ll_tests,
        sigma0 = sigma,
        cand_hs = cand_hs,
        ll_uniform = ll_uniform
    )


def cross_validate_k_folds(data, samples, full_grid, irregular_grid, k=5, pilot_h=None, conditioned=False, true_sigma=None, check_uniform=False, rng=1):
    """
    Perform k-fold cross-validation to select the best bandwidth and interaction parameter.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=rng)
    
    results_keys = ['pilot_h', 'll_tests', 'sigma0', 'cand_hs']
    if check_uniform:
        results_keys.append('ll_uniform')
    results = {key: [] for key in results_keys}

    fold = 1
    for train_index, test_index in kf.split(samples):
        print(f"\n========== Fold {fold}/{k} ==========")
        snapshot_times = data['time_ms'].unique()
        data_train = data.query('time_ms in @snapshot_times[@train_index]')
        samples_train = samples[train_index]
        samples_test = samples[test_index]

        r = run_full_pipeline_for_fold(
            data_train, samples_train, samples_test, full_grid, irregular_grid, pilot_h=pilot_h, conditioned=conditioned, sigma=true_sigma, check_uniform=check_uniform
        )
        for key in results:
            results[key].append(r[key])
        fold += 1

        print("\n--- Fold Results ---")
        print(f"Pilot bandwidth h     = {r['pilot_h']:.4f}")
        print(f"Sigma                 = {r['sigma0']:.4f}")
        print("Candidate bandwidths h: ", end="")
        print(", ".join([f"{h:.4f}" for h in r['cand_hs']]))
        print("Log-likelihoods on test set: ", end="")
        print(", ".join([f"{ll:.4f}" for ll in r['ll_tests']]))
        if check_uniform:
            print(f"Log-likelihood for uniform intensity: {r['ll_uniform']:.4f}")

    # aggregation
    print("\n=========== Cross‑validated Summary ===========")
    for key in ['pilot_h', 'sigma0']:
        mean = np.mean(results[key])
        std  = np.std(results[key])
        print(f"{key} : mean={mean:.4f}, sd={std:.4f}")

    # Final chosen hyperparameters:
    cand_hs = np.array(results['cand_hs']).mean(axis=0)
    print("\nCandidate bandwidths h (averaged over folds): ", end="")
    print(", ".join([f"{h:.4f}" for h in cand_hs]))
    # Convert the nested list of log-likelihoods into a pure NumPy 2D array
    ll_tests_cpu = np.array([
        [float(x.get()) if hasattr(x, "get") else float(x) for x in fold_list]
        for fold_list in results['ll_tests']
    ])

    ll_tests_sum = ll_tests_cpu.sum(axis=0)
    print("\nSummed log-likelihoods across folds: ", end="")
    print(", ".join([f"{ll:.4f}" for ll in ll_tests_sum]))
    if check_uniform:
        ll_uniforms = np.array([
            float(x.get()) if hasattr(x, "get") else float(x) for x in results['ll_uniform']
        ])

    best_idx = np.argmax(ll_tests_sum)
    best_coeff = COEFFS[best_idx]

    # Decide the base pilot h to multiply the coefficient with:
    if pilot_h is not None:
        # Iteration >= 2: use the pilot that was passed in (from previous iteration)
        base_h = pilot_h
        pilot_source = "previous iteration pilot"
    else:
        # Iteration 1: compute a global pilot on (rectangle/full) data as you did before
        rectangle_grid = reindex_rect_grid(full_grid)
        base_h = select_bandwidth_discrete(
            data[['x_position_mm','y_position_mm']].to_numpy(dtype=float), rectangle_grid, candidate_hs=np.arange(0.1, 10.1, 0.1), k_folds=10, mode='k-dpp', verbose=True, n_jobs_folds=-4, n_jobs_cands=1, samples_coords="physical"
        ).best_h
        pilot_source = "global pilot from data"

    best_h = best_coeff * base_h
    results["best_h"] = best_h
    # (optional but handy for debugging/plots)
    results["best_coeff"] = best_coeff
    results["base_h"] = base_h

    print("\n=========== Final Model Selection ===========")
    print(f"Best coefficient = {best_coeff:.4f}")
    print(f"Base pilot h     = {base_h:.4f} ({pilot_source})")
    print(f"Chosen bandwidth h = {best_h:.4f}")
    print(f"Sum of log-likelihoods at chosen coefficient = {ll_tests_sum[best_idx]:.4f}")
    intensity_best = estimate_intensity(
        data_train, full_grid, bandwidth=best_h, mask=create_mask(full_grid, irregular_grid)
    ) 

    if check_uniform:
        print("For comparison, log-likelihood for uniform intensity on folds:")
        print(ll_uniforms)
        print(f"Mean log-likelihood for uniform intensity: {np.mean(ll_uniforms)}")
        print(f"Sum of log-likelihoods for uniform intensity across folds: {ll_uniforms.sum()}")

    if true_sigma is not None:
        print("True sigma is known, so no reestimation necessary.")
        results["sigma1"] = true_sigma
    else:
        if best_coeff != 1.0:
            print("Reestimating interaction parameter at selected bandwdith...")
            sigma1 = reestimate_sigma_conditional(
                data,
                samples,
                full_grid,
                irregular_grid,
                h=best_h,
                sigmas=SIGMAS
            )
            results["sigma1"] = sigma1
        else:
            print("Selected bandwidth is the same as global pilot, skipping reestimation of sigma.")
            results["sigma1"] = np.mean(results["sigma0"])

    return results


def iterative_refinement(
    data,
    samples,
    full_grid,
    irregular_grid,
    conditioned=True,
    n_iter=1,
    true_sigma=None,
    k_folds=5,
    check_uniform=False
):
    """
    Iteratively refine the bandwidth and interaction parameter.
    """
    hs, sigmas = [], []
    ll_tests = []
    if check_uniform:
        ll_uniforms = []
    h_next = None
    for i in range(n_iter):
        print(f"\n========== Iteration {i+1}/{n_iter} ==========")
        results = cross_validate_k_folds(
            data, samples, full_grid, irregular_grid, k=k_folds, pilot_h=h_next, conditioned=conditioned, true_sigma=true_sigma, check_uniform=check_uniform
        )
        h_next = results["best_h"]
        sigma_next = results.get("sigma1", np.mean(results["sigma0"]))
        if i == 0:
            hs.append(results["base_h"])
            if true_sigma is not None:
                sigmas.append(true_sigma)
            else:
                sigmas.append(np.mean(results["sigma0"]))

        hs.append(h_next)
        sigmas.append(sigma_next)
        print(f"Selected bandwidth h: {h_next:.4f}")
        print(f"Selected sigma: {sigma_next:.4f}")

        ll_tests_iteration = results['ll_tests']
        ll_tests.append(ll_tests_iteration)

        if check_uniform:
            ll_uniform = results['ll_uniform']
            ll_uniforms.append(ll_uniform)

    res_dict = {
        "h_path": hs,
        "sigma_path": sigmas,
        "ll_tests": ll_tests,
    }

    if check_uniform:
        res_dict["ll_uniform"] = ll_uniforms

    return res_dict


def plot_parameter_paths(results):
    """
    Plot the paths of bandwidth h and interaction parameter sigma over iterations.
    """
    h_path = results["h_path"]
    sigma_path = results["sigma_path"]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(h_path, marker='o')
    plt.title("Bandwidth h over iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Bandwidth h")

    plt.subplot(1, 2, 2)
    plt.plot(sigma_path, marker='o')
    plt.title("Sigma over iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Sigma")

    plt.tight_layout()
    plt.show()


################################################################

def main():
    # import grid and define rectangle for estimation
    full_grid = pd.read_parquet("../data/grid_500mm.parquet")
    grid_rect = (24, 64, 8, 24)
    grid_xmin, grid_xmax, grid_ymin, grid_ymax = grid_rect
    rectangle_grid = full_grid[
        (full_grid['x_rounded'] >= grid_xmin) & (full_grid['x_rounded'] <= grid_xmax) &
        (full_grid['y_rounded'] >= grid_ymin) & (full_grid['y_rounded'] <= grid_ymax)
        ]
    rectangle_grid = reindex_rect_grid(rectangle_grid)

    # import data
    data = pd.read_parquet("../data/snapshots.parquet")

    # restrict data to rectangle for estimation
    rect = (12000, 32000, 4000+4136, 12000+4136)
    xmin, xmax, ymin, ymax = rect
    rectangle_data = data[
        (data['x_position_mm'] >= xmin) &
        (data['x_position_mm'] < xmax) &
        (data['y_position_mm'] + 4136 >= ymin) &
        (data['y_position_mm'] + 4136 < ymax)
    ]

    # assign points to grid cells
    assigned = assign_grid(rectangle_data, rectangle_grid)
    times = rectangle_data["time_ms"].unique()
    samples_rect = []
    for t in times:
        idx = assigned.loc[assigned["time_ms"] == t, "grid_index"].to_numpy(dtype=np.int32)
        idx = np.unique(idx)  # remove duplicates within snapshot (multiple points in same cell)
        samples_rect.append(idx)
    samples_rect = np.array(samples_rect, dtype=object)

    # run estimation algorithm
    results = iterative_refinement(
        rectangle_data,
        samples_rect,
        rectangle_grid,
        rectangle_grid,
        conditioned=True,
        n_iter=5,
        k_folds=5
    )

    plot_parameter_paths(results)


if __name__ == "__main__":
    main()