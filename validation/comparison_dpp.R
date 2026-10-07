library(spatstat)
source("validation/validation_metrics.R")
source("validation/import_helpers.R")
source("plotting/validation_plots_dpp.R")

h <- 0.42 # bandwidth for density estimation, in grid cells

metrics_file <- "validation/metrics/comparison_dpp_metrics.rds"
plot_dir     <- "validation/validation_plots"

# Metrics are cached in `metrics_file`; set to TRUE to recompute them (e.g.
# after rerunning the simulations). Plots can be changed without recomputing.
recompute_metrics <- FALSE

if (recompute_metrics || !file.exists(metrics_file)) {
  # load data, assigned to the cells of 500 mm grid
  # (coordinates in grid units)
  data_ppp <- load_sim_csv("data/samples.csv")
 
  # intensity estimate for the inhomogeneous L-function of all sources, from
  # all data (should match intensity used in simulation)
  intensity <- grid_kde(data_ppp, h = h)

  # load simulations
  pp_sets <- list(
    Data               = data_ppp,
    `Discrete Poisson` = load_sim_csv("simulation/data_simulation/csr_simulations.csv"),
    DPP                = load_sim_csv("simulation/data_simulation/dpp_simulations.csv")
  )

  # 2-NN densities up to 7.5 m
  metrics <- compute_validation_metrics_dpp(pp_sets = pp_sets, lambda = intensity,
                                            nn_max = c(5, 7.5))
  dir.create(dirname(metrics_file), showWarnings = FALSE, recursive = TRUE)
  saveRDS(metrics, metrics_file)
}

# plot
metrics <- readRDS(metrics_file)
plots <- make_comparison_plots_dpp(metrics)
save_validation_plots(plots, plot_dir, prefix = "comparison_dpp")
