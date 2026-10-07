library(spatstat)

gamma <- 0.15 # interaction parameter
R <- 400 # interaction radius
b <- 296 # bandwidth for density estimation

out_file <- "simulation/data_simulation/strauss_simulations.rds"

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

model <- rmhmodel(
  cif = "strauss",
  par = list(beta = 1, gamma = gamma, r = R), # beta is scaling factor for trend
  w = window,
  trend = intensity # exp due to log scaling in definition
)

# MCMC simulation function
simulate_strauss <- function(data_ppp) {
  n_points <- data_ppp$n
  start <- rmhstart(n.start = n_points) # generate a random configuration of n points
  control <- rmhcontrol(p = 1, nrep = 1e5, nverb = 1e4) # fixed number of points
  sim_ppp <- rmh(model, start = start, control = control) # simulate
  sim_ppp
}

simulations <- lapply(data_ppp, simulate_strauss)
dir.create(dirname(out_file), showWarnings = FALSE, recursive = TRUE)
saveRDS(simulations, out_file)