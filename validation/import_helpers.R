library(spatstat)

# the full 500 mm grid in grid units (one unit = one cell of ~0.5 m)
window_grid <- owin(xrange = c(0, 160), yrange = c(0, 26))

unflatten_coords <- function(id, ncols = 160) {
  x <- (id %% ncols) + 0.5
  y <- (id %/% ncols) + 0.5
  data.frame(x = x, y = y)
}

make_ppp <- function(config, ncols = 160, win = window_grid) {
  coords <- unflatten_coords(as.numeric(config), ncols = ncols)
  ppp(x = coords$x, y = coords$y, window = win)
}

# assign physical coordinates (mm) to the nearest cell of the 500 mm grid (data/grid_500mm.parquet)
physical_to_cell_id <- function(x, y, ncols = 160, nrows = 26,
                                x0 = 0, y0 = -47.111084, dx = 502.958588, dy = 511.111083) {
  ix <- pmin(pmax(round((x - x0) / dx), 0), ncols - 1)
  iy <- pmin(pmax(round((y - y0) / dy), 0), nrows - 1)
  iy * ncols + ix
}

load_sim_csv <- function(path) {
  df <- read.csv(path)
  if (is.null(df$cell_id)) df$cell_id <- physical_to_cell_id(df$x, df$y)
  configs <- split(df$cell_id, df$pattern_id)   # list of index vectors
  configs <- lapply(configs, unique)            # remove duplicates (multiple points in same cell)
  as.solist(lapply(configs, make_ppp))
}

# Intensity of the aggregated patterns on the grid cells, estimated with the
# discrete Gaussian KDE used for the DPP (estimate_intensity() in
# dpp/inference_intensity_dpp.py): a separable kernel with bandwidth h (in
# cells) whose weights are normalized over the grid. No platform mask is
# applied, since some data points lie in masked cells.
# Returns an image with one pixel per cell.
grid_kde <- function(pp_list, h, ncols = 160, nrows = 26) {
  weights <- function(n) {
    W <- dnorm(outer(seq_len(n), seq_len(n), "-") / h)
    W / rowSums(W)
  }
  agg    <- do.call(superimpose, pp_list)
  counts <- table(factor(floor(agg$y), 0:(nrows - 1)), factor(floor(agg$x), 0:(ncols - 1)))
  p      <- weights(nrows) %*% (unclass(counts) / npoints(agg)) %*% t(weights(ncols))
  im(p / sum(p) + 1e-10, xcol = seq_len(ncols) - 0.5, yrow = seq_len(nrows) - 0.5)
}