import numpy as np


def gaussian_pdf(u):
    return np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)


def build_weight_matrix(N, h):
    """
    Build the N x N weight matrix W_h with entries
      W[i,j] ∝ K( (i-j)/h ), then row-normalize over j ∈ {0, 1, ..., N-1}.
    Normalize by s_i(h) = sum_j K((i-j)/h).
    """
    lattice = np.arange(N)
    I = lattice[:, None]  # column vector
    J = lattice[None, :]  # row vector
    # raw wegihts from Gaussian kernel (infinite support but finite window "lattice")
    U = (I - J) / h  # shape (N, N)
    W = gaussian_pdf(U)  # shape (N, N)
    # row-normalize
    row_sums = W.sum(axis=1, keepdims=True)  # shape (N, 1)
    W = W / row_sums  # shape (N, N)
    return W


def discrete_kde_2d(X, Y, nx, ny, h):
    # Build 1D kernels
    Wx = build_weight_matrix(nx, h) # (nx × nx)
    Wy = build_weight_matrix(ny, h) # (ny × ny)

    # Build empirical distribution
    p_emp = np.zeros((ny, nx)) # rows = y, cols = x
    for x, y in zip(X, Y):
        p_emp[y, x] += 1
    p_emp = p_emp / len(X)

    # Apply 2D kernel smoothing
    # First smooth along x-axis (columns)
    tmp = p_emp @ Wx.T # (ny × nx)

    # Then smooth along y-axis (rows)
    p_hat = Wy @ tmp # (ny × nx)

    return p_hat