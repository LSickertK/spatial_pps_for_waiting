import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dppy.finite_dpps import FiniteDPP

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from dpp.kernels import generalized_gaussian
from dpp.inference_intensity_dpp import estimate_intensity
from dpp.grid_helpers import create_mask

"""
Simulate "twins" of the observed snapshots from the fitted DPP model. For each sample
in data/samples.npy, a k-DPP with k = len(sample) is drawn on the full 500 mm grid,
using a Gaussian kernel with the estimated intensity and interaction parameter.
All simulated patterns are collected in a numpy array of dtype=object and exported
as a .csv file for validation in R.
"""

################################################################

# fitted model parameters (in grid units, i.e. 500 mm cells) - set to estimated values
BANDWIDTH = 0.42  # KDE bandwidth h for the intensity (in grid units)
SIGMA = 0.714       # interaction parameter sigma of the Gaussian kernel

# k-DPPs are invariant to scaling L, so scale the intensity to prevent the elementary
# symmetric polynomials in dppy from underflowing to 0 for large k
SCALE = 500

SEED = None # optional seeding
OUT_FILE = ROOT / "simulation" / "data_simulation" / "dpp_simulations.csv"


def sample_k_dpp(eig_vals, eig_vecs, k, rng):
    """ Draw one k-DPP sample from the eigendecomposition of L """
    dpp = FiniteDPP("likelihood", L_eig_dec=(eig_vals, eig_vecs))
    dpp.sample_exact_k_dpp(size=k, random_state=rng)
    return dpp.list_of_samples[-1]


################################################################

def main():
    samples = np.load(ROOT / "data" / "samples.npy", allow_pickle=True)
    data = pd.read_parquet(ROOT / "data" / "snapshots.parquet")
    full_grid = pd.read_parquet(ROOT / "data" / "grid_500mm.parquet")
    clean_grid = pd.read_parquet(ROOT / "data" / "clean_grid_500mm.parquet")
    points = np.arange(len(full_grid))  # ground set: grid indices of the full grid
    print("data and grids loaded.")

    # intensity on the full grid (zero outside the platform), flattened row-major to match grid_index
    mask = create_mask(full_grid, clean_grid)
    intensities = estimate_intensity(data, full_grid, bandwidth=BANDWIDTH, mask=mask, normalize=True)
    intensities = SCALE * np.array(intensities, dtype=np.float64).flatten()

    model = generalized_gaussian(N=len(full_grid), d=2, rho=np.sqrt(intensities), sigma=SIGMA, beta=2, window=full_grid)

    # eigendecomposition of L only needs to be computed once for all samples
    eig_vals, eig_vecs = np.linalg.eigh(model.L().get_matrix())
    eig_vals = np.clip(eig_vals, 0.0, None)
    print("kernel built.")

    # simulate one k-DPP per observed snapshot, with k = number of points in the snapshot
    rng = np.random.RandomState(SEED)
    simulations = np.empty(len(samples), dtype=object)
    for i, sample in enumerate(samples):
        simulations[i] = points[sample_k_dpp(eig_vals, eig_vecs, len(sample), rng)]
    print(f"{len(simulations)} samples simulated.")

    # export as csv, one row per point (same format as data/samples.csv, plus grid cell ids)
    # x, y are the physical coordinates of the grid cell centers
    lut = full_grid.set_index("grid_index")[["x", "y"]]
    df = pd.concat(
        [
            pd.DataFrame({
                "pattern_id": i + 1,
                "cell_id": s,
                "x": lut.loc[s, "x"].to_numpy(),
                "y": lut.loc[s, "y"].to_numpy(),
            })
            for i, s in enumerate(simulations)
        ],
        ignore_index=True,
    )
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)
    print(f"saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
