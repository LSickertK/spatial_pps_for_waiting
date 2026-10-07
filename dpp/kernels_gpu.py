import numpy as np
import pandas as pd
import cupy as cp


# Wrapper class for kernel matrices (mainly useful for calculating submatrices easily)
class MatrixWrapper:
    def __init__(self, matrix):
        self.matrix = matrix

    def __repr__(self):
        return f"{self.matrix}"
    
    def get_matrix(self):
        # If matrix is a CuPy array, keep it
        # If numpy array, ensure it's float64
        if isinstance(self.matrix, cp.ndarray):
            return self.matrix
        else:
            return cp.asarray(self.matrix, dtype=cp.float64)
        
    def submatrix(self, sample):
        sample_gpu = cp.asarray(sample) if not isinstance(sample, cp.ndarray) else sample
        return self.matrix[sample_gpu[:, None], sample_gpu]


# Custom implementation of cdist for CuPy arrays
def cp_cdist(XA, XB, **kwargs):
    """
    Compute distance between each pair of the two collections of inputs.
    This custom implementation currently only supports 'sqeuclidean' metric.

    Parameters
    ----------
    XA : cupy.ndarray
        An (M, K) array of original observations in an K-dimensional space.
    XB : cupy.ndarray
        An (N, K) array of original observations in an K-dimensional space.
    **kwargs : dict, optional
        Extra arguments to metric (not used for 'sqeuclidean').

    Returns
    -------
    cupy.ndarray
        A (M, N) distance matrix.
    
    """
    
    if XA.shape[1] != XB.shape[1]:
        raise ValueError("XA and XB must have the same number of columns (dimensions).")

    # Ensure inputs are float type, as distances are typically float.
    # CuPy will handle type promotion if needed, but explicit conversion can prevent issues.
    XA = XA.astype(cp.float64, copy=False) # Use float64 for precision by default
    XB = XB.astype(cp.float64, copy=False)

    # Calculate ||a||^2 for each row in XA
    sum_sq_XA = cp.sum(XA**2, axis=1, keepdims=True) # Shape (M, 1)

    # Calculate ||b||^2 for each row in XB
    sum_sq_XB = cp.sum(XB**2, axis=1, keepdims=True) # Shape (N, 1)

    # Calculate the dot product matrix (XA @ XB.T)
    # Resulting shape (M, N)
    dot_product_matrix = XA @ XB.T

    # Calculate the squared Euclidean distance matrix
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * (a . b)
    # Using broadcasting: sum_sq_XA is (M,1), sum_sq_XB.T is (1,N)
    dist_matrix = sum_sq_XA + sum_sq_XB.T - 2 * dot_product_matrix

    return dist_matrix


# Gaussian kernel class for learning with CuPy support
class gaussian_kernel_learn:
    def __init__(self, N, d, rho, window="grid"):
        self.N = N
        self.d = d
        
        if isinstance(rho, np.ndarray):
            self.rho = cp.asarray(rho, dtype=cp.float64)
        else:
            self.rho = cp.array(rho, dtype=cp.float64)
        
        self.window = window

        # Pre-process window if it's a DataFrame, convert to CuPy array
        if isinstance(self.window, pd.DataFrame):
            x_coords = self.window["x_rounded"].values
            y_coords = self.window["y_rounded"].values
            self.coords_gpu = cp.stack((cp.asarray(x_coords, dtype=cp.float64), cp.asarray(y_coords, dtype=cp.float64)), axis=1)
            self.distance_matrix = cp_cdist(self.coords_gpu, self.coords_gpu)
        elif self.window == "grid":
            side = int(np.sqrt(N))
            x, y = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
            coords = np.column_stack([x.ravel(), y.ravel()])
            self.coords_gpu = cp.asarray(coords, dtype=cp.float64)
            self.distance_matrix = cp_cdist(self.coords_gpu, self.coords_gpu)   
        else:
            raise ValueError("Object passed to \"window\" not supported.")
        
    def build(self, sigma):
        sigma_gpu = cp.asarray(sigma, dtype=cp.float64)

        # grid case
        if isinstance(self.rho, cp.ndarray) and self.rho.ndim > 0:
            rho_matrix_gpu = cp.outer(self.rho, self.rho)
            L_gpu = rho_matrix_gpu * cp.exp(- self.distance_matrix / (2 * sigma_gpu**2))
        else:
            rho_sq_gpu = self.rho ** 2
            L_gpu = rho_sq_gpu * cp.exp(- self.distance_matrix / (2 * sigma_gpu**2))

        # DataFrame case (identical change)
        if isinstance(self.rho, cp.ndarray) and self.rho.ndim > 0:
            rho_matrix_gpu = cp.outer(self.rho, self.rho)
            L_gpu = rho_matrix_gpu * cp.exp(- self.distance_matrix / (2 * sigma_gpu**2))
        else:
            rho_sq_gpu = self.rho ** 2
            L_gpu = rho_sq_gpu * cp.exp(- self.distance_matrix / (2 * sigma_gpu**2))
        
        return L_gpu

    def L(self, sigma):
        return MatrixWrapper(self.build(sigma))
    
    def K(self, sigma):
        L_matrix_gpu = self.L(sigma).get_matrix()
        identitiy_matrix_gpu = cp.identity(L_matrix_gpu.shape[0], dtype=L_matrix_gpu.dtype)
        K_gpu = L_matrix_gpu @ cp.linalg.inv(identitiy_matrix_gpu + L_matrix_gpu)
        return MatrixWrapper(K_gpu)