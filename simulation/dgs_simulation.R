library(spatstat)

alpha <- 2.7 # shape parameter for DGS interaction
R <- 600 # interaction radius
b <- 297.5 # bandwidth for density estimation

out_file <- "simulation/data_simulation/dgs_simulations.rds"

# load window
window   <- readRDS("data/clean_window.rds")

# load data
data <- read.csv("data/samples.csv")
data_ppp <- lapply(split(data, data$pattern_id), function(d) {
  ppp(d$x, d$y + 4136, window = window)
})

agg <- do.call(superimpose, data_ppp)

intensity <- density(agg, sigma = b, diggle = TRUE)
intensity <- eval.im(pmax(intensity, 0))

# Build lookup table for custom DGS interaction
n_steps <- 1000 # number of steps in step function
r <- seq(R / n_steps, R, length.out = n_steps) # distances
h <- rep(1, n_steps)
h[r > 0] <- (sin(pi * r[r > 0] / (2 * R)))^(2 * alpha)
h[length(h)] <- 1  # ensure the last value = 1

# Define the model with lookup
model_lookup <- rmhmodel(
  cif = "lookup",
  par = list(beta = 1, r = r, h = h),
  w = window,
  trend = intensity # spatial trend
)

# MCMC Simulation function (fixed number of points)
simulate_lookup <- function(data_ppp) {
  n_points <- data_ppp$n
  start <- rmhstart(n.start = n_points) # random starting configuration
  control <- rmhcontrol(p = 1, nrep = 1e5, nverb = 1e4) # fixed number of points
  rmh(model_lookup, start = start, control = control)
}

# Simulate for all test patterns
simulations <- lapply(data_ppp, simulate_lookup)

# Save simulated set
dir.create(dirname(out_file), showWarnings = FALSE, recursive = TRUE)
saveRDS(simulations, out_file)
