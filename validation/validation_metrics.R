library(spatstat)

# Summary statistics used to compare simulated point patterns with the data.
#
# Every metric takes a named list of point pattern sets, e.g.
#   list(Data = data_ppp, Simulations = simulations)
# and returns a tidy data frame with a `source` column (a factor whose levels
# follow the order of the list). Distances are returned in meters; use
# `units_per_m` to state the units of the point coordinates (1000 for mm).

# occupancy class of a pattern with n points: low (<= 30), med (31-60), high (> 60)
occupancy_class <- function(n, breaks = c(30, 60)) {
  cut(n, c(-Inf, breaks, Inf), labels = c("low", "med", "high"))
}

as_source <- function(src, pp_sets) factor(src, levels = names(pp_sets))

# Mean k-NN distance per pattern, averaged over patterns with the same number of points.
# Only occupancy levels with at least `min_reps` patterns in every source are kept.
nn_by_occupancy <- function(pp_sets, k = 1, min_reps = 5, units_per_m = 1000) {
  df <- do.call(rbind, lapply(names(pp_sets), function(src) {
    pps <- Filter(function(pp) npoints(pp) > k, pp_sets[[src]])
    data.frame(
      npoints = sapply(pps, npoints),
      nn      = sapply(pps, function(pp) mean(nndist(pp, k = k))) / units_per_m,
      source  = src
    )
  }))

  counts <- table(df$npoints, df$source)
  valid  <- as.numeric(rownames(counts))[apply(counts >= min_reps, 1, all)]
  df <- df[df$npoints %in% valid, ]
  if (nrow(df) == 0) {
    warning("nn_by_occupancy: no occupancy level has >= ", min_reps, " patterns in every source")
    return(data.frame(npoints = numeric(0), mean = numeric(0), se = numeric(0),
                      source = as_source(character(0), pp_sets)))
  }

  groups <- split(df$nn, list(df$npoints, df$source), drop = TRUE, sep = "|")
  keys   <- do.call(rbind, strsplit(names(groups), "|", fixed = TRUE))
  data.frame(
    npoints = as.numeric(keys[, 1]),
    mean    = sapply(groups, mean),
    se      = sapply(groups, function(x) sd(x) / sqrt(length(x))),
    source  = as_source(keys[, 2], pp_sets),
    row.names = NULL
  )
}

# Empirical density (histogram) of all k-NN distances, pooled per occupancy class.
# `breaks` are the histogram bin edges in meters; distances outside the
# bin range are discarded before normalizing.
nn_distance_density <- function(pp_sets, k = 1, breaks = seq(0, 5, length.out = 100),
                                units_per_m = 1000, occupancy_breaks = c(30, 60)) {
  centers <- 0.5 * (breaks[-1] + breaks[-length(breaks)])

  do.call(rbind, lapply(names(pp_sets), function(src) {
    pps   <- Filter(function(pp) npoints(pp) > k, pp_sets[[src]])
    dists <- lapply(pps, function(pp) nndist(pp, k = k) / units_per_m)
    occ   <- occupancy_class(sapply(pps, npoints), occupancy_breaks)

    do.call(rbind, lapply(levels(occ), function(cls) {
      d <- unlist(dists[occ == cls])
      d <- d[d >= min(breaks) & d <= max(breaks)]
      if (length(d) == 0) return(NULL)
      data.frame(
        r         = centers,
        density   = hist(d, breaks = breaks, plot = FALSE)$density,
        occupancy = factor(cls, levels = levels(occ)),
        source    = as_source(src, pp_sets)
      )
    }))
  }))
}

# Kernel density estimate of all k-NN distances, pooled per occupancy class,
# evaluated at `n` points on `range` (in meters). Replaces the histogram of
# nn_distance_density() for patterns on a lattice, whose k-NN distances only
# take a few discrete values. `adjust` scales the "nrd" bandwidth, either by
# one value or per occupancy class, e.g. c(low = 1, med = 1, high = 1.5).
nn_distance_kde <- function(pp_sets, k = 1, range = c(0, 5), adjust = 1, n = 512,
                            units_per_m = 1000, occupancy_breaks = c(30, 60)) {
  do.call(rbind, lapply(names(pp_sets), function(src) {
    pps   <- Filter(function(pp) npoints(pp) > k, pp_sets[[src]])
    dists <- lapply(pps, function(pp) nndist(pp, k = k) / units_per_m)
    occ   <- occupancy_class(sapply(pps, npoints), occupancy_breaks)

    do.call(rbind, lapply(levels(occ), function(cls) {
      d <- unlist(dists[occ == cls])
      d <- d[d >= range[1] & d <= range[2]]
      if (length(unique(d)) < 2) return(NULL)
      kde <- density(d, bw = "nrd", adjust = if (length(adjust) == 1) adjust else adjust[[cls]],
                     n = n, from = range[1], to = range[2])
      data.frame(
        r         = kde$x,
        density   = kde$y,
        occupancy = factor(cls, levels = levels(occ)),
        source    = as_source(src, pp_sets)
      )
    }))
  }))
}

# ball count of each point for each radius: matrix (points x radii).
neighbour_counts <- function(pp, r) {
  cp <- closepairs(pp, rmax = max(r), what = "ijd")
  matrix(vapply(r, function(ri) {
    close <- cp$d <= ri
    tabulate(c(cp$i[close], cp$j[close]), nbins = npoints(pp))
  }, numeric(npoints(pp))), nrow = npoints(pp))
}

# Average ball count C(r) around a point, averaged over all points of all
# patterns (optionally per occupancy class). `r` is given in meters.
ball_count <- function(pp_sets, r = 10^seq(log10(0.5), log10(50), length.out = 50),
                       by_occupancy = FALSE, units_per_m = 1000,
                       occupancy_breaks = c(30, 60)) {
  do.call(rbind, lapply(names(pp_sets), function(src) {
    pps    <- Filter(function(pp) npoints(pp) > 0, pp_sets[[src]])
    n      <- sapply(pps, npoints)
    counts <- do.call(rbind, lapply(pps, neighbour_counts, r = r * units_per_m))
    group  <- if (by_occupancy) rep(occupancy_class(n, occupancy_breaks), n)
              else factor(rep("all", nrow(counts)))

    do.call(rbind, lapply(levels(droplevels(group)), function(g) {
      m <- counts[group == g, , drop = FALSE]
      data.frame(
        r         = r,
        mean      = colMeans(m),
        se        = apply(m, 2, sd) / sqrt(nrow(m)),
        occupancy = factor(g, levels = levels(group)),
        source    = as_source(src, pp_sets)
      )
    }))
  }))
}

# L-function (no edge correction) averaged over patterns. With `lambda` (an
# intensity image, or a list of images named by source) the inhomogeneous
# version is computed; lambda is renormalized per pattern so that it integrates
# to that pattern's number of points. `r` is given in meters.
average_L <- function(pp_sets, r = seq(0, 3, by = 0.01), lambda = NULL,
                      units_per_m = 1000) {
  stopifnot(is.null(lambda) || is.im(lambda) || all(names(pp_sets) %in% names(lambda)))
  r_units <- r * units_per_m
  L_one <- function(pp, lambda) {
    L <- if (is.null(lambda)) Lest(pp, r = r_units, correction = "none")
         else Linhom(pp, lambda = lambda, r = r_units, correction = "none", renormalise = TRUE)
    L$un
  }

  do.call(rbind, lapply(names(pp_sets), function(src) {
    lambda_src <- if (is.im(lambda)) lambda else lambda[[src]]
    data.frame(
      r          = r,
      L          = rowMeans(sapply(pp_sets[[src]], L_one, lambda = lambda_src)) / units_per_m,
      correction = if (is.null(lambda)) "hom." else "inhom.",
      source     = as_source(src, pp_sets)
    )
  }))
}

# All metrics used for validation, bundled in a list that can be saved with
# saveRDS() and passed to the plotting functions. `L_sets` may contain further
# reference models (e.g. Poisson) for the L-function comparison; `lambda` is
# the intensity estimate used for the inhomogeneous L-function (one image for
# all sources, or a list of images named by source). For a comparison of
# several models, pass all of them in `pp_sets`.
compute_validation_metrics <- function(pp_sets, L_sets = pp_sets, lambda,
                                       units_per_m = 1000) {
  message("Computing k-NN distances...")
  nn1_by_occupancy <- nn_by_occupancy(pp_sets, k = 1, units_per_m = units_per_m)
  nn2_by_occupancy <- nn_by_occupancy(pp_sets, k = 2, units_per_m = units_per_m)
  nn1_density <- nn_distance_density(pp_sets, k = 1, breaks = seq(0, 5, length.out = 100),
                                     units_per_m = units_per_m)
  nn2_density <- nn_distance_density(pp_sets, k = 2, breaks = seq(0, 6, length.out = 100),
                                     units_per_m = units_per_m)

  message("Computing ball counts...")
  ball_count_all <- ball_count(pp_sets, units_per_m = units_per_m)
  ball_count_occ <- ball_count(pp_sets, by_occupancy = TRUE, units_per_m = units_per_m)

  message("Computing L-functions...")
  L <- rbind(average_L(L_sets, units_per_m = units_per_m),
             average_L(L_sets, lambda = lambda, units_per_m = units_per_m))
  L$correction <- factor(L$correction, levels = c("hom.", "inhom."))

  list(
    nn1_by_occupancy = nn1_by_occupancy,
    nn2_by_occupancy = nn2_by_occupancy,
    nn1_density      = nn1_density,
    nn2_density      = nn2_density,
    ball_count       = ball_count_all,
    ball_count_occ   = ball_count_occ,
    L                = L
  )
}

# The same metrics for patterns on the 500 mm DPP grid, whose coordinates are
# in grid cells (~0.5 m each, hence `units_per_m` = 2). The k-NN distance
# densities are kernel smoothed instead of binned (see nn_distance_kde()), on
# 0 to `nn_max` meters (1-NN and 2-NN) with bandwidth adjustment `kde_adjust`;
# the high occupancy class is smoothed more.
compute_validation_metrics_dpp <- function(pp_sets, L_sets = pp_sets, lambda, nn_max = c(5, 6),
                                           kde_adjust = c(low = 1.5, med = 1.5, high = 2.25),
                                           units_per_m = 2) {
  message("Computing k-NN distances...")
  nn1_by_occupancy <- nn_by_occupancy(pp_sets, k = 1, units_per_m = units_per_m)
  nn2_by_occupancy <- nn_by_occupancy(pp_sets, k = 2, units_per_m = units_per_m)
  nn1_density <- nn_distance_kde(pp_sets, k = 1, range = c(0, nn_max[1]), adjust = kde_adjust,
                                 units_per_m = units_per_m)
  nn2_density <- nn_distance_kde(pp_sets, k = 2, range = c(0, nn_max[2]), adjust = kde_adjust,
                                 units_per_m = units_per_m)

  message("Computing ball counts...")
  ball_count_all <- ball_count(pp_sets, units_per_m = units_per_m)
  ball_count_occ <- ball_count(pp_sets, by_occupancy = TRUE, units_per_m = units_per_m)

  message("Computing L-functions...")
  r <- seq(0, 2.5, by = 0.05)
  L <- rbind(average_L(L_sets, r = r, units_per_m = units_per_m),
             average_L(L_sets, r = r, lambda = lambda, units_per_m = units_per_m))
  L$correction <- factor(L$correction, levels = c("hom.", "inhom."))

  list(
    nn1_by_occupancy = nn1_by_occupancy,
    nn2_by_occupancy = nn2_by_occupancy,
    nn1_density      = nn1_density,
    nn2_density      = nn2_density,
    ball_count       = ball_count_all,
    ball_count_occ   = ball_count_occ,
    L                = L
  )
}
