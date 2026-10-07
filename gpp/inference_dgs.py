import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import gc

try:
    from gpp.inference_gpp_backend import (
        DGSOptions,
        Grid,
        df_to_configs,
        estimate_poisson_kde_intensity,
        make_baseline_intensity_spec,
        plot_intensity_heatmap,
        plot_cpl_surface_with_ridge,
        dgs_grid_search,
        dgs_total_conditional_log_pseudolikelihood,
    )
except ImportError:
    from inference_gpp_backend import (
        DGSOptions,
        Grid,
        df_to_configs,
        estimate_poisson_kde_intensity,
        make_baseline_intensity_spec,
        plot_intensity_heatmap,
        plot_cpl_surface_with_ridge,
        dgs_grid_search,
        dgs_total_conditional_log_pseudolikelihood,
    )

################################################################

COEFFS = np.arange(0.5, 1.5, 0.05)
PILOTS = np.arange(50, 2001, 50)

ALPHA_VALS = np.arange(0, 10.05, 0.05)
R_VALS = np.arange(100, 1001, 50)


def evaluate_model_across_bandwidths_conditional(
    data_train,
    data_test,
    cand_hs,
    R0,
    alpha0,
    window,
    grid=(160, 64),
    erode=False,
    n_jobs=1,
    prefer="threads"
):
    """
    For each candidate bandwidth h:
      - Estimate baseline intensity from data_train with bandwidth h.
      - Compute conditional log-pseudolikelihood on data_test set with fixed (R0, alpha0).
    Return (best_h, logcpl_tests).
    """
    opts = DGSOptions(window=window, grid_nx=grid[0], grid_ny=grid[1])
    
    train_list = list(data_train)
    test_list = list(data_test)

    def one(h):
        img, extent, _ = estimate_poisson_kde_intensity(
            configs=train_list,
            window=window,
            grid=Grid(grid[0], grid[1]),
            bandwidth=h,
            hs=PILOTS,
        )
        intensity = make_baseline_intensity_spec(img, extent, mode="offset", takes_log=True)
        return dgs_total_conditional_log_pseudolikelihood(
            configs=test_list,
            intensity=intensity,
            r=R0,
            alpha=alpha0,
            options=opts,
            erode=erode,
        )

    if n_jobs == 1:
        logcpl_tests = [one(h) for h in cand_hs]
    else:
        logcpl_tests = Parallel(n_jobs=n_jobs, prefer=prefer)(
            delayed(one)(h) for h in cand_hs
        )

    del train_list, test_list
    gc.collect()

    best_idx = int(np.argmax(logcpl_tests))
    return cand_hs[best_idx], logcpl_tests


def reestimate_R_alpha_conditional(
    data_train, 
    h, 
    window, 
    grid, 
    erode=False,
    r_values=R_VALS,
    alpha_values=ALPHA_VALS,
    pin_grid=True,
    fixed_erosion_radius=None,
    n_jobs=1,
    prefer="threads"
):
    """Re-profile (R, alpha) on data_train given a fixed bandwidth h for the baseline."""
    img, extent, _ = estimate_poisson_kde_intensity(
        configs=data_train,
        window=window,
        grid=Grid(grid[0], grid[1]),
        bandwidth=h
    )
    intensity = make_baseline_intensity_spec(img, extent, mode="offset", takes_log=True)
    dgs_opts = DGSOptions(window=window, grid_nx=grid[0], grid_ny=grid[1])

    prof = dgs_grid_search(
        configs=list(data_train),
        intensity=intensity,
        r_values=r_values,
        alpha_values=alpha_values,
        options=dgs_opts,
        erode=erode,
        pin_grid=pin_grid,
        fixed_erosion_radius=fixed_erosion_radius,
        n_jobs=n_jobs,
        prefer=prefer
    )
    return prof["best_R"], prof["best_alpha"], prof['logcpl']


def run_full_pipeline_for_fold(
    data_train,
    data_test,
    pilot_h=None,
    window=None,
    grid=(160, 64),
    erode=False,
    n_jobs=1,
    prefer="threads"
):
    # Pilot bandwidth via Poisson-likelihood CV (if not provided)
    if pilot_h is None:
        print("Estimating pilot bandwidth via Poisson-likelihood CV…")
        img, extent, h0 = estimate_poisson_kde_intensity(
            configs=list(data_train),
            window=window,
            grid=Grid(grid[0], grid[1]),
            bandwidth=None,
            hs = PILOTS
        )
        plot_intensity_heatmap(img, extent, title=f"Pilot baseline (h = {h0:.2f})")
    else:
        img, extent, _ = estimate_poisson_kde_intensity(
            configs=list(data_train),
            window=window,
            grid=Grid(grid[0], grid[1]),
            bandwidth=pilot_h,
            hs=PILOTS
        )
        h0 = pilot_h

    # Build intensity for parameter profiling (offset, log)
    intensity = make_baseline_intensity_spec(img, extent, mode="offset", takes_log=True)
    dgs_opts = DGSOptions(window=window, grid_nx=grid[0], grid_ny=grid[1])

    # Conditional MPLE profile over (R, alpha)
    print("Estimating interaction parameters (conditional MPLE)…")
    prof = dgs_grid_search(
        configs=list(data_train),
        intensity=intensity,
        r_values=R_VALS,
        alpha_values=ALPHA_VALS,
        options=dgs_opts,
        erode=erode,
        pin_grid=True,
        fixed_erosion_radius=None,
        n_jobs=n_jobs,
        prefer=prefer
    )
    R0, alpha0 = prof["best_R"], prof["best_alpha"]

    # Candidate bandwidths: c * h0
    cand_hs = [h0 * c for c in COEFFS]

    # Choose h that maximizes test-set conditional pseudolikelihood
    print("Evaluating bandwidths on the test split (conditional CPL)…")
    best_h, logcpl_tests = evaluate_model_across_bandwidths_conditional(
        data_train=data_train,
        data_test=data_test,
        cand_hs=cand_hs,
        R0=R0,
        alpha0=alpha0,
        window=window,
        grid=grid,
        erode=erode,
        n_jobs=n_jobs,
        prefer=prefer
    )

    return dict(R0=R0, alpha0=alpha0, h0=h0, best_h=best_h, logcpl_tests=logcpl_tests, cand_hs=cand_hs)


def cross_validate_k_folds(data, k=5, pilot_h=None, window=None, grid=(160, 64), erode=False, n_jobs=1, prefer="threads", rng=1):
    kf = KFold(n_splits=k, shuffle=True, random_state=rng)
    results = {key: [] for key in ["R0", "alpha0", "h0", "best_h", "logcpl_tests", "cand_hs"]}

    for fold, (train_idx, test_idx) in enumerate(kf.split(data), start=1):
        print(f"\n========== Fold {fold}/{k} ==========")
        data_train = data[train_idx]
        data_test = data[test_idx]
        r = run_full_pipeline_for_fold(
            data_train=data_train,
            data_test=data_test,
            pilot_h=pilot_h,
            window=window,
            grid=grid,
            erode=erode,
            n_jobs=n_jobs,
            prefer=prefer
        )
        for key in results:
            results[key].append(r[key])
        
        print("\n--- Fold Results ---")
        print(f"Pilot bandwidth h     = {r['h0']:.4f}")
        print(f"R                 = {r['R0']:.4f}")
        print(f"Alpha             = {r['alpha0']:.4f}")
        print("Candidate bandwidths h: ", end="")
        print(", ".join([f"{h:.4f}" for h in r['cand_hs']]))
        print("Log-likelihoods on test set: ", end="")
        print(", ".join([f"{ll:.4f}" for ll in r['logcpl_tests']]))

        del data_train, data_test
        gc.collect()

    # ---- aggregation ----
    print("\n=========== Cross‑validated Summary ===========")
    for key in ["R0", "alpha0", "h0"]:
        mean = np.mean(results[key])
        std = np.std(results[key])
        print(f"{key} : mean={mean:.4f}, sd={std:.4f}")

    # Candidate bandwidths (averaged over folds; all folds use the same COEFFS length)
    cand_hs = np.array(results['cand_hs']).mean(axis=0)
    print("\nCandidate bandwidths h (averaged over folds): ", end="")
    print(", ".join([f"{h:.4f}" for h in cand_hs]))
    logcpl_tests_sum = np.array(results['logcpl_tests']).sum(axis=0)
    print("\nSummed conditional log-pseudolikelihood across folds: ", end="")
    print(", ".join([f"{ll:.4f}" for ll in logcpl_tests_sum]))
    best_idx = int(np.argmax(logcpl_tests_sum))

    # pick best coefficient by likelihood, then multiply by a global pilot bandwidth
    img1, extent1, global_pilot_h = estimate_poisson_kde_intensity(
        configs=list(data),
        window=window,
        grid=Grid(grid[0], grid[1]),
        bandwidth=None,
        hs=PILOTS
    )
    best_coeff = COEFFS[best_idx]
    if pilot_h is None:
        # No prior iteration to warm-start from: h0^all = Poisson-CV pilot over all data, per spec.
        best_h = best_coeff * global_pilot_h
        anchor_desc = "global pilot"
    else:
        # Warm-started loop: anchor to the same pilot the per-fold candidates were built around
        # (the previous iteration's h_hat), not the iteration-independent global_pilot_h
        best_h = best_coeff * pilot_h
        anchor_desc = "previous iteration's h_hat"
    results["best_h"] = best_h
    results["global_pilot_h"] = global_pilot_h

    print("\n=========== Final Model Selection ===========")
    print(f"Global pilot bandwidth (reference only) = {global_pilot_h:.4f}")
    print(f"Best coefficient = {best_coeff:.4f}")
    print(f"Chosen bandwidth h = {best_h:.4f} (using {anchor_desc})")

    # Visual checks
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    cand_hs_plot = results["cand_hs"][0]  # same for all folds
    if pilot_h is None:
        # first iteration → coefficients undefined
        x = cand_hs_plot
        ax.plot(x, logcpl_tests_sum, marker='o', lw=1.8)
        ax.set_xlabel("Bandwidth h (pilot grid)")
    else:
        x = cand_hs_plot / pilot_h
        ax.plot(x, logcpl_tests_sum, marker='o', lw=1.8)
        ax.set_xlabel("Bandwidth coefficient c")
        ax.axvline(COEFFS[best_idx], color='red', ls='--', lw=1.5, label=f'best c = {COEFFS[best_idx]:.2f}')
        ax.legend()
    ax.set_ylabel('Summed conditional log-pseudolikelihood across folds')
    ax.set_title('CV: Summed CPL vs bandwidth coefficient')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()

    plot_intensity_heatmap(img1, extent1, title=f"Baseline (global pilot h = {global_pilot_h:.2f})")
    img3, extent3, _ = estimate_poisson_kde_intensity(
        configs=list(data), 
        window=window, 
        bandwidth=best_h, 
        grid=Grid(grid[0], grid[1]), 
        hs = PILOTS
    )
    plot_intensity_heatmap(img3, extent3, title="Estimated baseline chosen h")

    # Re‑estimate (R, alpha) at best_h
    if best_coeff != 1.0:
        print("Estimating interaction parameters at pilot bandwidth…")
        R0, alpha0, logcpl0 = reestimate_R_alpha_conditional(
            data_train=data,
            h=global_pilot_h,
            window=window,
            grid=grid,
            r_values=R_VALS,
            alpha_values=ALPHA_VALS,
            erode=erode,
            n_jobs=n_jobs,
            prefer=prefer
        )
        print(f"At pilot bandwidth: R = {R0:.4f}, alpha = {alpha0:.4f}")
        print("Re‑estimating interaction parameters at the selected bandwidth…")
        R1, alpha1, logcpl1 = reestimate_R_alpha_conditional(
            data_train=data,
            h=best_h,
            window=window,
            grid=grid,
            r_values=R_VALS,
            alpha_values=ALPHA_VALS,
            erode=erode,
            n_jobs=n_jobs,
            prefer=prefer
        )
        results["R0"] = R0
        results["alpha0"] = alpha0
        results["logcpl0"] = logcpl0
        results["R1"] = R1
        results["alpha1"] = alpha1
        results["logcpl1"] = logcpl1
        print(f"At selected bandwidth: R = {R1:.4f}, alpha = {alpha1:.4f}")
    else:
        print("Selected bandwidth is the same as global pilot. Estimating at original bandwidth...")
        R0, alpha0, logcpl0 = reestimate_R_alpha_conditional(
            data_train=data,
            h=global_pilot_h,
            window=window,
            r_values=R_VALS,
            alpha_values=ALPHA_VALS,
            grid=grid,
            erode=erode,
            n_jobs=n_jobs,
            prefer=prefer,
        )
        results["R0"] = R0
        results["alpha0"] = alpha0
        results["logcpl0"] = logcpl0
        results["R1"] = R0
        results["alpha1"] = alpha0
        results["logcpl1"] = logcpl0
        print(f"At pilot bandwidth: R = {R0:.4f}, alpha = {alpha0:.4f}")

    if logcpl0 is not None:
        plot_cpl_surface_with_ridge(
            R=R_VALS,
            A=ALPHA_VALS,
            Z=logcpl0,
            title=f"Conditional log-pseudolikelihood surface at global pilot bandwidth h={global_pilot_h:.4f}",
            contour_levels=15
        )

    if logcpl1 is not None:
        plot_cpl_surface_with_ridge(
            R=R_VALS,
            A=ALPHA_VALS,
            Z=logcpl1,
            title=f"Conditional log-pseudolikelihood surface at selected bandwidth h={best_h:.4f}",
            contour_levels=15
        )

    return results


def iterative_refinement(
    data,
    n_iter=10,
    k_folds=5,
    window=None,
    grid=(160, 64),
    erode=False,
    n_jobs=1,
    prefer="threads",
    rng=1
):
    hs, Rs, alphas, logcpls = [], [], [], []
    pilot_h = None  # compute pilot in first iteration

    for i in range(n_iter):
        print(f"\n========== Iteration {i+1}/{n_iter} ==========")
        cv_results = cross_validate_k_folds(
            data,
            k=k_folds,
            pilot_h=pilot_h,
            window=window,
            grid=grid,
            erode=erode,
            n_jobs=n_jobs,
            prefer=prefer,
            rng=rng
        )
        h_next = cv_results["best_h"]
        R_next = cv_results["R1"]
        alpha_next = cv_results["alpha1"]
        if i == 0:
            hs.append(cv_results["global_pilot_h"])
            Rs.append(cv_results["R0"])
            alphas.append(cv_results["alpha0"])
            logcpls.append(cv_results.get("logcpl0"))
        hs.append(h_next)
        Rs.append(R_next)
        alphas.append(alpha_next)
        logcpls.append(cv_results.get("logcpl1", None))
        pilot_h = h_next

        print(f"Selected bandwidth h = {h_next:.4f}, R = {R_next:.4f}, alpha = {alpha_next:.4f}")

        del cv_results
        gc.collect()

    return {
        "h_path": np.array(hs),
        "R_path": np.array(Rs),
        "alpha_path": np.array(alphas),
        "logcpl_surfaces": logcpls
    }


def plot_parameter_paths(results):
    h_path = results["h_path"]
    R_path = results["R_path"]
    alpha_path = results["alpha_path"]

    iters = np.arange(1, len(h_path) + 1)
    plt.figure(figsize=(14, 4))

    # bandwidth
    plt.subplot(1, 3, 1)
    plt.plot(iters, h_path, marker='o')
    plt.title("Bandwidth h over iterations")
    plt.xlabel("Iteration")
   
    # R
    plt.subplot(1, 3, 2)
    plt.plot(iters, R_path, marker='o')
    plt.title("R over iterations")
    plt.xlabel("Iteration")

    # alpha
    plt.subplot(1, 3, 3)
    plt.plot(iters, alpha_path, marker='o')
    plt.title("α over iterations")
    plt.xlabel("Iteration")

    plt.tight_layout()
    plt.show()

###############################################################################

def main():
    xmin, ymin, xmax, ymax = 12000, 4000, 32000, 12000
    window = (xmin, ymin, xmax, ymax)

    # import data
    data = pd.read_parquet("../data/snapshots.parquet")
    data_cropped = data[
        (data.x_position_mm >= window[0]) & 
        (data.x_position_mm <= window[2]) & 
        (data.y_position_mm >= window[1]) & 
        (data.y_position_mm <= window[3])
    ]
    configs, _, _ = df_to_configs(data_cropped)
    configs = np.array(configs, dtype=object)

    # for testing
    no_configs = 10
    configs = configs[:no_configs]
    print(f"Using first {no_configs} frames for testing.")

    results = iterative_refinement(
        configs,
        n_iter=1,
        k_folds=2,
        window=window,
        grid=(160, 64),
        erode=False,
    )

    plot_parameter_paths(results)


if __name__ == "__main__":
    main()