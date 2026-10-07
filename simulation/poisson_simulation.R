library(spatstat)

out_file <- "simulation/data_simulation/poisson_simulations.rds"

# load window
window   <- readRDS("data/clean_window.rds")

# load data
data <- read.csv("data/samples.csv")
data_ppp <- lapply(split(data, data$pattern_id), function(d) {
  ppp(d$x, d$y + 4136, window = window)
})

# estimate an intensity function from the data
agg <- do.call(superimpose, data_ppp)
b_lcv <- bw.ppl(agg)
est_intensity <- density(agg, sigma = b_lcv, diggle = TRUE)

est_intensity <- eval.im(est_intensity) # ensures values inside the image
est_intensity <- as.im(est_intensity, W = window)
# Truncate negative values to 0
est_intensity[est_intensity < 0] <- 0

n <- length(data_ppp)
sim_set <- lapply(data_ppp, function(pp) {
  n_points <- pp$n
  rpoint(n_points, win = window, f = est_intensity)
})

dir.create(dirname(out_file), showWarnings = FALSE, recursive = TRUE)
saveRDS(sim_set, out_file)