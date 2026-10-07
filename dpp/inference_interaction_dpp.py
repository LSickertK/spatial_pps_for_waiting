import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

try:
    from .kernels_gpu import gaussian_kernel_learn
except ImportError: 
    from kernels_gpu import gaussian_kernel_learn

# Elementary symmetric polynomial computation for k-DPP normalization (GPU version)
def elementary_symmetric_all(lam, Kmax, rescale=True):
    """
    Compute E[0..Kmax] where E[l] = e_l(lam), using Algorithm 7 (Kulesza & Taskar).
    - lam : (N,) CuPy array of eigenvalues of L
    - Kmax: largest k needed (max sample size across the fold)
    Returns:
      E         : CuPy array (Kmax+1,) with E[0]=1 and E[l]=e_l(lam)
      log_scale : CuPy scalar tracking multiplicative rescaling applied to E
    """
    lam = cp.asarray(lam, dtype=cp.float64)
    # Clamp tiny negative eigenvalues from numerical round-off
    lam = cp.maximum(lam, 0.0)

    E = cp.zeros(Kmax + 1, dtype=lam.dtype)
    E[0] = 1.0
    log_scale = cp.asarray(0.0, dtype=lam.dtype)

    for i in range(lam.size):
        lmb = lam[i]
        # We can only populate up to min(i+1, Kmax)
        upper = min(i + 1, Kmax)
        # Descending update: E[l] <- E[l] + λ_i * E[l-1]
        # Use a RHS copy to avoid aliasing the freshly-updated E
        rhs = E[:upper].copy()        # E[0:upper]
        E[1:upper+1] += lmb * rhs     # E[1:upper+1] <-- E[1:upper+1] + λ * E[0:upper]

        if rescale:
            max_abs = cp.max(cp.abs(E))
            if max_abs != 0:
                # Keep magnitudes in a comfortable range to avoid under/overflow
                if (max_abs > 1e100) or (max_abs < 1e-100):
                    E /= max_abs
                    log_scale += cp.log(max_abs)

    return E, log_scale


# Inference class for Gaussian kernel DPPs on a grid, with CuPy support
class gaussian_grid_search:
    def __init__(self, N, d, samples, intensity, window="grid"):
        self.N = N
        self.d = d
        self.samples_gpu = [cp.asarray(s, dtype=cp.int32) for s in samples]
        self.intensity = intensity
        self.window = window

        gauss = gaussian_kernel_learn(
            N=self.N, d=self.d, rho=self.intensity, window=self.window
        )
        self.L = gauss.L  # kernel builder

        # k-DPP bookkeeping
        self.k_list = [int(s.size) for s in self.samples_gpu]
        self._unique_k = sorted(set(self.k_list))
        self._count_by_k = {k: self.k_list.count(k) for k in self._unique_k}

        # caches
        self._cache_sigma = None
        self._cache_L = None
        self._cache_lam = None
        self._cache_U = None

    # Internal cache update
    def _update_cache(self, sigma):
        if self._cache_sigma is None or float(sigma) != float(self._cache_sigma):
            self._cache_sigma = float(sigma)
            Lmat = self.L(sigma).get_matrix()  # build L(sigma)
            self._cache_L = Lmat
            self._cache_lam, self._cache_U = cp.linalg.eigh(Lmat)

    # Determinant utilities
    def log_det(self, submatrix):
        """Stable GPU log(det()) via Cholesky."""
        try:
            chol = cp.linalg.cholesky(
                submatrix + 1e-8 * cp.eye(submatrix.shape[0], dtype=submatrix.dtype)
            )
            return 2 * cp.sum(cp.log(cp.diag(chol)))
        except cp.linalg.LinAlgError:
            print("Matrix not PD in log_det()")
            print(cp.asnumpy(submatrix))
            return cp.asarray(0.0, dtype=cp.float64)

    # DPP log-likelihood
    def log_likelihood(self, sigma):
        self._update_cache(sigma)
        L = self._cache_L
        T = len(self.samples_gpu)

        I = cp.eye(self.N, dtype=cp.float64)
        chol = cp.linalg.cholesky(L + I + 1e-10 * I)
        logdet_I_plus_L = 2 * cp.sum(cp.log(cp.diag(chol)))
        global_term = -T * logdet_I_plus_L

        logdet_sum = cp.asarray(0.0)
        for s in self.samples_gpu:
            idx = s
            sub = L[idx[:, None], idx]
            logdet_sum += self.log_det(sub)

        return global_term + logdet_sum

    # k‑DPP log-likelihood (corrected)
    def log_likelihood_k(self, sigma):
        """Compute k-DPP log-likelihood for a given sigma."""
        self._update_cache(sigma)
        L_full = self._cache_L
        lam = self._cache_lam

        # Submatrix determinant term
        terms = []
        for s in self.samples_gpu:
            idx = s
            sub = L_full[idx[:, None], idx]
            terms.append(self.log_det(sub))
        logdet_sum = cp.sum(cp.stack(terms)) if terms else cp.asarray(0.0)

        # Spectral normalizer: - ∑ log e_{k_t}(λ)
        Kmax = int(max(self._unique_k))
        E, log_scale = elementary_symmetric_all(lam, Kmax, rescale=True)

        logZ_terms = []
        for k in self._unique_k:
            Ek = cp.clip(E[k], 1e-300, cp.inf)
            log_ek = log_scale + cp.log(Ek)
            logZ_terms.append(self._count_by_k[k] * log_ek)

        logZ = cp.sum(cp.stack(logZ_terms))

        return logdet_sum - logZ, logdet_sum, logZ

    # Sigma grid search
    def fit(self, sigmas, mode="k-DPP", local_maxima=False, plot=False, plot_components=False):
        """
        Sweep over sigma values and return (sigma_best, all_values).
        mode: "DPP", "k-DPP", or "kDPP"
        """
        mode_norm = mode.replace("-", "").lower()

        values = []
        log_det_values = []
        logZ_values = []
        for sigma in sigmas:
            if mode_norm == "dpp":
                ll, sample_contributions, normalizer = self.log_likelihood(sigma)
            elif mode_norm == "kdpp":
                ll, sample_contributions, normalizer = self.log_likelihood_k(sigma)
            else:
                raise ValueError("mode must be 'DPP', 'k-DPP', or 'kDPP'")

            values.append(float(cp.asnumpy(ll)))
            log_det_values.append(float(cp.asnumpy(sample_contributions)))
            logZ_values.append(float(cp.asnumpy(normalizer)))

        best_idx = int(np.argmax(values))

        if local_maxima:
            print("Local maxima (sigma, log-likelihood):")
            maxima = []

            n = len(values)

            # Left boundary
            if n > 1 and values[0] > values[1]:
                print(f"  Sigma: {sigmas[0]:.4f}, Log-Likelihood: {values[0]:.4f}")
                maxima.append((sigmas[0], values[0]))

            # Interior points
            for i in range(1, n - 1):
                if values[i] > values[i - 1] and values[i] > values[i + 1]:
                    print(f"  Sigma: {sigmas[i]:.4f}, Log-Likelihood: {values[i]:.4f}")
                    maxima.append((sigmas[i], values[i]))

            # Right boundary
            if n > 1 and values[-1] > values[-2]:
                print(f"  Sigma: {sigmas[-1]:.4f}, Log-Likelihood: {values[-1]:.4f}")
                maxima.append((sigmas[-1], values[-1]))

            if plot:
                plt.plot(sigmas, values)

                for sigma, ll in maxima:
                    plt.scatter(sigma, ll, label=f'{sigma:.2f}', color='red', marker='x')

                plt.xlabel("Sigma")
                plt.ylabel("k-DPP Log-Likelihood")
                plt.title("k-DPP Log-Likelihood vs Sigma")
                plt.legend()
                plt.show()

                if plot_components:
                    log_det_values = [float(cp.asnumpy(v)) for v in log_det_values]
                    neglogZ_values = [-float(cp.asnumpy(v)) for v in logZ_values]

                    plt.plot(sigmas, log_det_values, label="Sum log det(L_{A_t})")
                    plt.plot(sigmas, neglogZ_values, label="- Sum log e_{k_t}(λ)")
                    plt.xlabel("Sigma")
                    plt.ylabel("Component Values")
                    plt.title("k-DPP Log-Likelihood Components")
                    plt.legend()
                    plt.show()

            return maxima, values
        
        else:
            if plot:
                plt.plot(sigmas, values)
                plt.scatter([sigmas[best_idx]], [max(values)], color='red', label=f'Best sigma: {sigmas[best_idx]:.2f}')
                plt.xlabel("Sigma")
                plt.ylabel("k-DPP Log-Likelihood")
                plt.title("k-DPP Log-Likelihood vs Sigma")
                plt.legend()
                plt.show()

                if plot_components:
                    log_det_values = [float(cp.asnumpy(v)) for v in log_det_values]
                    neglogZ_values = [-float(cp.asnumpy(v)) for v in logZ_values]

                    plt.plot(sigmas, log_det_values, label="Sum log det(L_{A_t})")
                    plt.plot(sigmas, neglogZ_values, label="- Sum log e_{k_t}(λ)")
                    plt.xlabel("Sigma")
                    plt.ylabel("Component Values")
                    plt.title("k-DPP Log-Likelihood Components")
                    plt.legend()
                    plt.show()
            return sigmas[best_idx], values

