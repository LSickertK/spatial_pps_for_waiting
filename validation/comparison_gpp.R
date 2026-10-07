library(spatstat)
source("validation/validation_metrics.R")
source("plotting/validation_plots_gpp.R")

# bandwidths of the intensity estimates for the inhomogeneous L-function, one
# per source (Strauss and DGS as in simulation/strauss_simulation.R and
# simulation/dgs_simulation.R, their mean for the data and Poisson)
bandwidths <- c(Data = 296.75, Poisson = 296.75, Strauss = 296, DGS = 297.5)

metrics_file <- "validation/metrics/comparison_gpp_metrics.rds"
plot_dir     <- "validation/validation_plots"

# Metrics are cached in `metrics_file`; set to TRUE to recompute them (e.g.
# after rerunning the simulations). Plots can be changed without recomputing.
recompute_metrics <- FALSE

if (recompute_metrics || !file.exists(metrics_file)) {
  # load window
  window <- readRDS("data/clean_window.rds")

  # load data
  data <- read.csv("data/samples.csv")
  data_ppp <- lapply(split(data, data$pattern_id), function(d) {
    ppp(d$x, d$y + 4136, window = window)
  })

  # load simulations
  pp_sets <- list(
    Data    = data_ppp,
    Poisson = readRDS("simulation/data_simulation/poisson_simulations.rds"),
    Strauss = readRDS("simulation/data_simulation/strauss_simulations.rds"),
    DGS     = readRDS("simulation/data_simulation/dgs_simulations.rds")
  )

  # intensity estimates for the inhomogeneous L-function, each from the
  # patterns of its own source
  intensities <- lapply(setNames(names(pp_sets), names(pp_sets)), function(src) {
    intensity <- density(do.call(superimpose, pp_sets[[src]]), sigma = bandwidths[[src]],
                         diggle = TRUE)
    intensity[intensity <= 0] <- 1e-6
    intensity
  })

  metrics <- compute_validation_metrics(pp_sets = pp_sets, lambda = intensities)
  dir.create(dirname(metrics_file), showWarnings = FALSE, recursive = TRUE)
  saveRDS(metrics, metrics_file)
}

# plot
metrics <- readRDS(metrics_file)
plots <- make_comparison_plots(metrics)
save_validation_plots(plots, plot_dir, prefix = "comparison_gpp")
