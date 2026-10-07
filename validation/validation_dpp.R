library(spatstat)
source("validation/validation_metrics.R")
source("validation/import_helpers.R")
source("plotting/validation_plots_dpp.R")

h <- 0.42 # bandwidth for density estimation, in grid cells

model_name   <- "DPP"
metrics_file <- "validation/metrics/dpp_metrics.rds"
plot_dir     <- "validation/validation_plots"
csr_file     <- "simulation/data_simulation/csr_simulations.csv" # optional, for L-function plot

# Metrics are cached in `metrics_file`; set to TRUE to recompute
# Plots can be changed without recomputing.
recompute_metrics <- FALSE

if (recompute_metrics || !file.exists(metrics_file)) {
  # load data, assigned to the cells of 500 mm grid
  # (coordinates in grid units)
  data_ppp <- load_sim_csv("data/samples.csv")

  # intensity estimate for the inhomogeneous L-function, from all data
  # (should match intensity used in simulation)
  intensity <- grid_kde(data_ppp, h = h)

  # load simulations
  simulations <- load_sim_csv("simulation/data_simulation/dpp_simulations.csv")

  # models to compare in the L-function plot
  L_sets <- setNames(list(data_ppp, simulations), c("Data", model_name))
  if (file.exists(csr_file)) L_sets[["Discrete Poisson"]] <- load_sim_csv(csr_file)

  metrics <- compute_validation_metrics_dpp(
    pp_sets = list(Data = data_ppp, Simulations = simulations),
    L_sets  = L_sets,
    lambda  = intensity
  )
  dir.create(dirname(metrics_file), showWarnings = FALSE, recursive = TRUE)
  saveRDS(metrics, metrics_file)
}

# plot
metrics <- readRDS(metrics_file)
plots <- make_validation_plots_dpp(metrics)
save_validation_plots(plots, plot_dir, prefix = tolower(model_name))
