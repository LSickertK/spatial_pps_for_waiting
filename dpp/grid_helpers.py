import numpy as np
import pandas as pd

from sklearn.neighbors import KDTree


def assign_grid(data, grid):
    """
    Assigns each point in `data` to its nearest grid cell in `grid`,
    and computes the integer grid indices (x_rounded, y_rounded) and 
    the flattened grid_index.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns ['x_position_mm', 'y_position_mm'] (for dim=2)
        or ['x_position_mm'] (for dim=1).
    grid : pd.DataFrame
        Must contain grid-center coordinates ['x', 'y']

    Returns
    -------
    df : pd.DataFrame
        Original data plus:
            - x_grid, y_grid (nearest grid coords)
            - x_rounded, y_rounded (integer grid indices)
            - grid_index (flattened index)
    """
    df = data.copy()

    tree = KDTree(grid[['x','y']].values, metric='chebyshev')
    _, idx = tree.query(df[['x_position_mm','y_position_mm']].values)
    idx = idx.flatten()

    # nearest grid coordinates
    df['x_grid'] = grid.iloc[idx]['x'].values
    df['y_grid'] = grid.iloc[idx]['y'].values

    # unique sorted axes
    xs = np.sort(grid['x'].unique())
    ys = np.sort(grid['y'].unique())

    # map physical coordinates to index positions
    df['x_rounded'] = xs.searchsorted(df['x_grid'])
    df['y_rounded'] = ys.searchsorted(df['y_grid'])

    # number of columns
    N_cols = len(xs)

    df['x_rounded'] = df['x_rounded'].astype(int)
    df['y_rounded'] = df['y_rounded'].astype(int)

    # flattened index, row-major
    df['grid_index'] = df['y_rounded'] * N_cols + df['x_rounded']

    return df


def grid_indices_to_physical_df(indices, grid: pd.DataFrame) -> pd.DataFrame:
    """
    Map a list/array of grid_index values to a DataFrame with columns x_position_mm,y_position_mm.
    Robust to non-square (rectangular) grids.
    """
    lut = grid[["grid_index", "x", "y"]].drop_duplicates("grid_index")
    # Make lookup fast
    lut = lut.set_index("grid_index")[["x", "y"]]

    idx = np.asarray(indices, dtype=int)
    # If any index is missing, you'll get KeyError: that's good — it reveals misindexing
    xy = lut.loc[idx].to_numpy()
    return pd.DataFrame({"x_position_mm": xy[:, 0], "y_position_mm": xy[:, 1]})


def create_mask(grid, selection):
    """
    Create a boolean mask from a given grid and a given selection of grid squares
    """
    # Create a set of coordinate tuples from selection for efficient lookup
    selection_coords = set(zip(selection['x'], selection['y']))
    mask = grid.apply(lambda row: (row['x'], row['y']) in selection_coords, axis=1)
    return ~mask.astype(bool) # negate: False for points to keep, True for points to remove


def reindex_rect_grid(g):
    """
    Reindex to match a new rectangular grid
    """
    g = g.copy()

    # Build local maps for rounded integer coordinates
    xs = np.sort(g["x_rounded"].unique())
    ys = np.sort(g["y_rounded"].unique())
    x_map = {x: i for i, x in enumerate(xs)}
    y_map = {y: i for i, y in enumerate(ys)}

    g["x_rounded"] = g["x_rounded"].map(x_map).astype(int)
    g["y_rounded"] = g["y_rounded"].map(y_map).astype(int)

    nx = len(xs)
    g["grid_index"] = g["y_rounded"] * nx + g["x_rounded"]

    # Optional: sort for consistency
    g = g.sort_values(["y_rounded", "x_rounded"]).reset_index(drop=True)
    return g