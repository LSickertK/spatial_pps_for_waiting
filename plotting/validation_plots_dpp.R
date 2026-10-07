source("plotting/validation_plots_gpp.R")

# Plots comparing DPP simulations with the data, built from the list returned by
# compute_validation_metrics_dpp() (validation/validation_metrics.R)
# plot functions are shared with the GPP validation
# reference slopes of the ball count and the panel tags differ, 
# and NN pdfs use a KDE smoothing instead of a histogram

# power-law reference lines coef * r^power for the ball count plot
ball_count_refs_dpp <- data.frame(
  label = c("r^2", "r"), power = c(2, 1), coef = c(3.5, 10),
  from = c(0.5, 3), to = c(3, 15),
  label_x = c(1.1, 6), label_mult = c(1.8, 1.5)
)

# Generate all validation plots. Returns a named list of ggplot objects; with
# `tags = TRUE` the panels are enumerated a), b), ...
make_validation_plots_dpp <- function(metrics, tags = FALSE) {
  list(
    nn1_by_occupancy = plot_nn_by_occupancy(metrics$nn1_by_occupancy, k = 1,
                                            tag = panel_tag("a)", tags)),
    nn2_by_occupancy = plot_nn_by_occupancy(metrics$nn2_by_occupancy, k = 2,
                                            tag = panel_tag("b)", tags)),
    nn1_density      = plot_nn_density(metrics$nn1_density, k = 1, tag = panel_tag("c)", tags)),
    nn2_density      = plot_nn_density(metrics$nn2_density, k = 2, tag = panel_tag("d)", tags)),
    L                = plot_L(metrics$L, tag = panel_tag("e)", tags)),
    ball_count       = plot_ball_count(metrics$ball_count, tag = panel_tag("f)", tags),
                                       refs = ball_count_refs_dpp),
    ball_count_occ   = plot_ball_count_by_occupancy(metrics$ball_count_occ,
                                                    tag = panel_tag("g)", tags))
  )
}

# Build all comparison plots from the list returned by
# compute_validation_metrics_dpp() with several models. Returns a named list of
# ggplot objects; `tags = TRUE` enumerates the panels.
make_comparison_plots_dpp <- function(metrics, tags = FALSE) {
  c(
    make_occupancy_comparison_plots(metrics, tags),
    list(
      L_hom   = plot_L_comparison(metrics$L, "hom.", tag = panel_tag("j)", tags),
                                  legend_alpha = 0.95, legend_linewidth = 0.4, base_size = 16),
      L_inhom = plot_L_comparison(metrics$L, "inhom.", tag = panel_tag("k)", tags),
                                  legend_alpha = 0.95, legend_linewidth = 0.4, base_size = 16)
    )
  )
}
