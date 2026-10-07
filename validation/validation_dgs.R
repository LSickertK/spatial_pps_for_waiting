library(spatstat)
source("validation/validation_metrics.R")
source("plotting/validation_plots_gpp.R")

b <- 297.5 # bandwidth for density estimation (as in simulation/dgs_simulation.R)

model_name   <- "DGS"
metrics_file <- "validation/metrics/dgs_metrics.rds"
plot_dir     <- "validation/validation_plots"
poisson_file <- "simulation/data_simulation/poisson_simulations.rds" # optional, for the L-function plot

# Metrics are cached in `metrics_file`; set to TRUE to recompute them (e.g.
# after rerunning the simulation). Plots can be changed without recomputing.
recompute_metrics <- FALSE

if (recompute_metrics || !file.exists(metrics_file)) {
  # load window
  window   <- readRDS("data/clean_window.rds")

  # load data
  data <- read.csv("data/samples.csv")
  data_ppp <- lapply(split(data, data$pattern_id), function(d) {
    ppp(d$x, d$y + 4136, window = window)
  })

  # load simulations
  simulations <- readRDS("simulation/data_simulation/dgs_simulations.rds")

  # intensity estimate for the inhomogeneous L-function
  agg <- do.call(superimpose, data_ppp)
  intensity <- density(agg, sigma = b, diggle = TRUE)
  intensity[intensity <= 0] <- 1e-6

  # models to compare in the L-function plot
  L_sets <- setNames(list(data_ppp, simulations), c("Data", model_name))
  if (file.exists(poisson_file)) L_sets$Poisson <- readRDS(poisson_file)

  metrics <- compute_validation_metrics(
    pp_sets = list(Data = data_ppp, Simulations = simulations),
    L_sets  = L_sets,
    lambda  = intensity
  )
  dir.create(dirname(metrics_file), showWarnings = FALSE, recursive = TRUE)
  saveRDS(metrics, metrics_file)
}

# plot
metrics <- readRDS(metrics_file)
plots <- make_validation_plots(metrics)
save_validation_plots(plots, plot_dir, prefix = tolower(model_name))
