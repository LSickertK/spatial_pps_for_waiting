import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from dppy.finite_dpps import FiniteDPP
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve
from joblib import Parallel, delayed


#######################################################################################


# Wrapper class for kernel matrices (mainly useful for calculating submatrices easily)
class MatrixWrapper:
    def __init__(self, matrix):
        self.matrix = matrix

    def __repr__(self):
        return f"{self.matrix}"
    
    def get_matrix(self):
        return np.array(self.matrix, dtype=np.float64)
    
    # Method to get the submatrix of the current matrix
    def submatrix(self, sample):
        return np.array(self.matrix[np.ix_(sample, sample)], dtype=np.float64)


# Implement complete spatial randomness as a DPP with a diagonal kernel (i.e. no interaction, just intensity)
class CSR_DPP:
    def __init__(self, N, rho, rng=None):
        self.N = N
        self.rho = np.asarray(rho, dtype=float)
        self.rng = np.random.default_rng(rng) if rng is not None else np.random.default_rng()

    def build(self):
        L = np.diag(self.rho**2)
        return L
    
    def L(self):
        return MatrixWrapper(self.build())
    
    def K(self):
        L = self.L().get_matrix()
        identity = np.identity(L.shape[0])
        c, lower = cho_factor(L + identity)
        K = L @ cho_solve((c, lower), np.eye(L.shape[0]))
        return MatrixWrapper(K)
    
    # Simulate finite DPP using L as likelihood kernel
    def sample_dpp_once(self, points, L_matrix, mode, k=None):
        """ Helper function to sample once from the DPP """
        if mode == 'likelihood':
            dpp = FiniteDPP('likelihood', L=L_matrix)
        elif mode == 'correlation':
            dpp = FiniteDPP('correlation', K=L_matrix)
        
        if k:
            dpp.sample_exact_k_dpp(size=k)
        else:
            dpp.sample_exact()
        
        sampled_indices = dpp.list_of_samples[-1]
        return points[sampled_indices]

    def sim(self, points, mode='likelihood', n=None, k=None):
        # Precompute the L matrix once
        L_matrix = self.L().get_matrix()
        
        # If n > 1, parallelize the sampling using joblib
        if n:
            samples = Parallel(n_jobs=-1)(
                delayed(self.sample_dpp_once)(points, L_matrix, mode, k) 
                for _ in range(n)
            )
        else:  # Sample once
            samples = self.sample_dpp_once(points, L_matrix, mode, k)

        return samples


# Generalized Gaussian kernel - beta=2 corresponds to Gaussian kernel. 
class generalized_gaussian:
    def __init__(self, N, d, rho, sigma, beta=2, window="grid"):
        self.N = N # cardinality of ground set
        self.d = d # dimension
        self.rho = rho # intensity
        self.sigma = sigma # std. dev., variance is sigma**2
        self.beta = beta # exponent, large beta -> faster decay
        self.window = window # "grid" for regular gridded window. 
        # If window is irregular grid, pass pandas dataframe to window keyword.
        # This df should contain coordinates of grid square centers assigned to each index (up to N)

    # build kernel matrix
    def build(self):
        L = np.zeros((self.N, self.N))
        if self.d == 1:
            if isinstance(self.window, pd.DataFrame):
                    # Extract the 1D coordinate
                    if "x_rounded" in self.window.columns:
                        coords = self.window["x_rounded"].to_numpy(dtype=float)
                    elif "x" in self.window.columns:
                        coords = self.window["x"].to_numpy(dtype=float)
                    else:
                        raise ValueError("1D kernel requires 'x_rounded' or 'x' column in the window DataFrame.")

                    # Pairwise 1D distances
                    dist_matrix = np.abs(coords[:, None] - coords[None, :])

                    # Inhomogeneous or homogeneous rho
                    if isinstance(self.rho, np.ndarray):
                        rho_outer = np.outer(self.rho, self.rho)
                    else:
                        rho_outer = self.rho**2

                    # Build the kernel
                    L = rho_outer * np.exp(-(dist_matrix**self.beta) / (2 * self.sigma**self.beta))
                    return L

            if isinstance(self.window, str) and self.window == "grid":
                if isinstance(self.rho, np.ndarray):
                    raise ValueError("1D 'grid' mode does not support inhomogeneous rho. Pass a DataFrame window.")

                L = np.zeros((self.N, self.N))
                for i in range(self.N):
                    for j in range(i+1):
                        dist = abs(i - j)
                        value = self.rho**2 * np.exp(-(dist**self.beta) / (2 * self.sigma**self.beta))
                        L[i, j] = value
                        L[j, i] = value
                return L


        elif self.d == 2:
            if isinstance(self.window, str) and self.window == "grid": # if grid, assume regular sqrt(N)xsqrt(N) grid in d dimensions
                coords = np.array([(i // np.sqrt(self.N), i % np.sqrt(self.N)) for i in range(self.N)])
                dist_matrix = cdist(coords, coords)
                # Precompute values only once
                rho_sq = self.rho ** 2 if not isinstance(self.rho, np.ndarray) else None
                L = rho_sq * np.exp(-(dist_matrix**self.beta) / (2 * self.sigma**self.beta)) if rho_sq else np.outer(self.rho, self.rho) * np.exp(-(dist_matrix**self.beta) / (2 * self.sigma**self.beta))
            elif isinstance(self.window, pd.DataFrame): # grid given by dataframe
                self.window = self.window.drop_duplicates(subset=["x_rounded", "y_rounded"]).reset_index(drop=True)

                # Extract coordinates from DataFrame
                x_coords = self.window["x_rounded"].values
                y_coords = self.window["y_rounded"].values
                coords = np.stack((x_coords, y_coords), axis=1)

                # Calculate distance matrix using cdist
                dist_matrix = cdist(coords, coords)

                # Apply Gaussian kernel to distance matrix
                if isinstance(self.rho, np.ndarray):
                    # If rho is an array, calculate outer product for weights
                    rho_matrix = np.outer(self.rho, self.rho)
                    L = rho_matrix * np.exp(-(dist_matrix**self.beta) / (2 * self.sigma**self.beta))
                else:
                    # If rho is a scalar, use it directly in the Gaussian function
                    L = (self.rho ** 2) * np.exp(-(dist_matrix**self.beta) / (2 * self.sigma**self.beta))
            else:
                raise ValueError("Object passed to \"window\" keyword not supported.")

        return L
    
    # wrap matrix
    def L(self):
        return MatrixWrapper(self.build())
    
    # calculate correlation kernel K from L
    def K(self):
        #K = self.L().get_matrix() @ np.linalg.inv(self.L().get_matrix() + np.identity(self.L().get_matrix().shape[0]))
        L = self.L().get_matrix()
        identity = np.identity(L.shape[0])
        c, lower = cho_factor(L + identity)
        K = L @ cho_solve((c, lower), np.eye(L.shape[0]))
        return MatrixWrapper(K)
    
    # Simulate finite DPP using L as likelihood kernel
    def sample_dpp_once(self, points, L_matrix, mode, k=None):
        """ Helper function to sample once from the DPP """
        if mode == 'likelihood':
            dpp = FiniteDPP('likelihood', L=L_matrix)
        elif mode == 'correlation':
            dpp = FiniteDPP('correlation', K=L_matrix)
        
        if k:
            dpp.sample_exact_k_dpp(size=k)
        else:
            dpp.sample_exact()
        
        sampled_indices = dpp.list_of_samples[-1]
        return points[sampled_indices]

    def sim(self, points, mode='likelihood', n=None, k=None):
        # Precompute the L matrix once
        L_matrix = self.L().get_matrix()
        
        # If n > 1, parallelize the sampling using joblib
        if n:
            samples = Parallel(n_jobs=-1)(
                delayed(self.sample_dpp_once)(points, L_matrix, mode, k) 
                for _ in range(n)
            )
        else:  # Sample once
            samples = self.sample_dpp_once(points, L_matrix, mode, k)
            
        return samples