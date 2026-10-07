library(ggplot2)

# Plots comparing simulations with the data. Each function takes one of the
# data frames produced by validation/validation_metrics.R (see
# compute_validation_metrics()), so plots can be restyled and regenerated
# without recomputing the metrics. Series are identified by the `source`
# column, so any model (Strauss, DGS, Poisson, ...) can be plotted.

occupancy_labels_short <- expression(
  low  = "Low occ. (" * phantom() <= 30 * ")",
  med  = "Med. occ. (31-60)",
  high = "High occ. (" * phantom() > 60 * ")"
)

occupancy_labels_long <- expression(
  low  = "Low occupancy (" <= 30 * ")",
  med  = "Medium occupancy (31–60)",
  high = "High occupancy (> 60)"
)

occupancy_axis_label <- expression("Platform occupancy [N(" * bold(x) * ")]")

# common look: legend inside the panel, large text, tag in the top-left corner
theme_validation <- function(legend_inside, legend_just = c(1, 0), legend_alpha = 0.6,
                             legend_linewidth = 0.5, base_size = 11, text_size = 16,
                             tag_size = 18) {
  theme_minimal(base_size = base_size) +
    theme(
      legend.position = "inside",
      legend.position.inside = legend_inside,
      legend.justification = legend_just,
      legend.background = element_rect(fill = alpha("white", legend_alpha),
                                       color = "black", linewidth = legend_linewidth),
      legend.title = element_blank(),
      legend.text = element_text(size = text_size),
      axis.title  = element_text(size = text_size),
      axis.text   = element_text(size = text_size),
      plot.tag = element_text(size = tag_size),
      plot.tag.position = c(0.02, 0.98)
    )
}

# log10 axes with minor breaks at 2..9 x 10^k and log ticks on bottom/left
log_minor_breaks <- function(limits) {
  decades <- seq(floor(log10(limits[1])), ceiling(log10(limits[2])))
  sort(unique(as.numeric(outer(2:9, 10^decades))))
}

log_log_axes <- function() {
  list(
    scale_x_log10(minor_breaks = log_minor_breaks),
    scale_y_log10(minor_breaks = log_minor_breaks),
    annotation_logticks(sides = "bl")
  )
}

# (a, b) mean k-NN distance vs. platform occupancy, with 95% CI band
plot_nn_by_occupancy <- function(df, k = 1, tag = NULL) {
  ggplot(df, aes(x = npoints, y = mean, color = source, fill = source)) +
    geom_line(linewidth = 1) +
    geom_ribbon(aes(ymin = mean - 1.96 * se, ymax = mean + 1.96 * se),
                alpha = 0.2, color = NA) +
    scale_color_brewer(palette = "Dark2") +
    scale_fill_brewer(palette = "Dark2") +
    labs(x = occupancy_axis_label, y = paste0(k, "-NN distance [m]"), tag = tag) +
    theme_validation(legend_inside = c(0.95, 0.8))
}

# (c, d) density of k-NN distances per occupancy class
plot_nn_density <- function(df, k = 1, tag = NULL) {
  ggplot(df, aes(x = r, y = density, color = occupancy, linetype = source)) +
    geom_line(linewidth = 1) +
    scale_color_brewer(palette = "Dark2", labels = function(x) occupancy_labels_short[x]) +
    guides(color = guide_legend(order = 1), linetype = guide_legend(order = 2)) +
    labs(x = paste0(k, "-NN distance [m]"), y = "Density", tag = tag) +
    theme_validation(legend_inside = c(0.98, 0.98), legend_just = c("right", "top"),
                     legend_alpha = 0.95, legend_linewidth = 0.4, base_size = 16) +
    theme(
      legend.box = "vertical",
      legend.box.just = "right",
      legend.spacing.y = unit(2, "mm"),
      legend.key.width = unit(1.8, "cm")
    )
}

# power-law reference lines coef * r^power for the ball count plot
ball_count_refs <- data.frame(
  label = c("r^2", "r"), power = c(2, 1), coef = c(2, 10),
  from = c(0.5, 6), to = c(5, 30),
  label_x = c(1.1, 12), label_mult = c(1.8, 1.5)
)

# (e) average ball count C(r) in log-log scale, with reference slopes
plot_ball_count <- function(df, tag = NULL, refs = ball_count_refs) {
  # reference lines use the next Dark2 colour after the sources
  ref_colour <- scales::brewer_pal(palette = "Dark2")(nlevels(df$source) + 1)[nlevels(df$source) + 1]
  r <- unique(df$r)
  ref_lines <- do.call(rbind, lapply(seq_len(nrow(refs)), function(i) {
    ri <- r[r >= refs$from[i] & r <= refs$to[i]]
    data.frame(r = ri, y = refs$coef[i] * ri^refs$power[i], ref = refs$label[i])
  }))

  ggplot(df, aes(x = r, y = mean, color = source)) +
    geom_line(linewidth = 1.2) +
    geom_ribbon(aes(ymin = mean - 1.96 * se, ymax = mean + 1.96 * se, fill = source),
                alpha = 0.2, show.legend = FALSE) +
    geom_line(data = ref_lines, aes(x = r, y = y, group = ref), inherit.aes = FALSE,
              color = ref_colour, linewidth = 1.2) +
    annotate("text", x = refs$label_x,
             y = refs$label_mult * refs$coef * refs$label_x^refs$power,
             label = refs$label, parse = TRUE, color = "black",
             hjust = 0, vjust = -0.5, size = 6) +
    scale_color_brewer(palette = "Dark2") +
    scale_fill_brewer(palette = "Dark2") +
    log_log_axes() +
    labs(x = "Radius [m]", y = "Average C(r)", tag = tag) +
    theme_validation(legend_inside = c(0.95, 0.1))
}

# (f) average ball count per occupancy class; CI bands for the simulations only
plot_ball_count_by_occupancy <- function(df, tag = NULL) {
  ggplot(df, aes(x = r, y = mean, color = occupancy, linetype = source)) +
    geom_line(linewidth = 1.2) +
    geom_ribbon(data = df[df$source != "Data", ],
                aes(ymin = mean - 1.96 * se, ymax = mean + 1.96 * se, fill = occupancy),
                alpha = 0.2, color = NA, show.legend = FALSE) +
    scale_color_brewer(palette = "Dark2", labels = function(x) occupancy_labels_long[x]) +
    scale_fill_brewer(palette = "Dark2") +
    log_log_axes() +
    guides(linetype = guide_legend(order = 1), color = guide_legend(order = 2)) +
    labs(x = "Radius [m]", y = "Average C(r)", tag = tag) +
    theme_validation(legend_inside = c(0.97, 0.05), legend_just = c("right", "bottom"),
                     legend_alpha = 1, base_size = 16) +
    theme(legend.box = "vertical", legend.box.just = "right")
}

# homogeneous and inhomogeneous average L-functions on a common scale
plot_L <- function(df, tag = NULL, legend_inside = c(0.35, 0.5)) {
  ggplot(df, aes(x = r, y = L, color = source, linetype = correction)) +
    geom_line(linewidth = 1.2) +
    scale_color_brewer(palette = "Dark2") +
    scale_linetype_manual(values = c("hom." = "solid", "inhom." = "11")) +
    guides(color = guide_legend(order = 1), linetype = guide_legend(order = 2)) +
    labs(x = "r [m]", y = "L(r)", tag = tag) +
    theme_validation(legend_inside = legend_inside, text_size = 14, tag_size = 16)
}

# panel label `label` (e.g. "a)") if `tags` is TRUE, otherwise no label
panel_tag <- function(label, tags) if (tags) label

# Build all validation plots from the list returned by
# compute_validation_metrics(). Returns a named list of ggplot objects; with
# `tags = TRUE` the panels are enumerated a), b), ...
make_validation_plots <- function(metrics, tags = FALSE) {
  list(
    nn1_by_occupancy = plot_nn_by_occupancy(metrics$nn1_by_occupancy, k = 1,
                                            tag = panel_tag("a)", tags)),
    nn2_by_occupancy = plot_nn_by_occupancy(metrics$nn2_by_occupancy, k = 2,
                                            tag = panel_tag("b)", tags)),
    nn1_density      = plot_nn_density(metrics$nn1_density, k = 1, tag = panel_tag("c)", tags)),
    nn2_density      = plot_nn_density(metrics$nn2_density, k = 2, tag = panel_tag("d)", tags)),
    ball_count       = plot_ball_count(metrics$ball_count, tag = panel_tag("e)", tags)),
    ball_count_occ   = plot_ball_count_by_occupancy(metrics$ball_count_occ,
                                                    tag = panel_tag("f)", tags)),
    L                = plot_L(metrics$L, tag = panel_tag("a)", tags))
  )
}

# ---- Comparison of several models ----
# Built from compute_validation_metrics() run with all models in `pp_sets`.
# Metrics split by occupancy class get one plot per class, one line per model.

# y-axis title naming the occupancy class, e.g. "Density (low occ.: <= 30)"
occupancy_axis_title <- function(prefix, occupancy) {
  switch(occupancy,
    low  = bquote(.(prefix) * " (low occ.: " * phantom() <= 30 * ")"),
    med  = paste0(prefix, " (med. occ.: 31–60)"),
    high = bquote(.(prefix) * " (high occ.: " * phantom() > 60 * ")")
  )
}

# density of k-NN distances in one occupancy class, optionally on a log10
# distance axis (legend then moves to the top-left corner)
plot_nn_density_comparison <- function(df, occupancy, k = 1, xlog = FALSE, tag = NULL) {
  df <- df[df$occupancy == occupancy & (!xlog | df$r > 0), ]
  p <- ggplot(df, aes(x = r, y = density, color = source)) +
    geom_line(linewidth = 1) +
    scale_color_brewer(palette = "Dark2") +
    labs(x = paste0(k, "-NN distance [m]"), y = occupancy_axis_title("Density", occupancy),
         tag = tag)
  if (xlog) {
    p <- p + scale_x_log10(breaks = scales::log_breaks(), minor_breaks = log_minor_breaks) +
      annotation_logticks(sides = "b")
  }
  p + theme_validation(legend_inside = c(if (xlog) 0.02 else 0.98, 0.98),
                       legend_just = c(if (xlog) "left" else "right", "top"),
                       legend_alpha = 0.95, legend_linewidth = 0.4, base_size = 16) +
    theme(legend.key.width = unit(1.8, "cm"))
}

# average ball count C(r) in one occupancy class, with 95% CI bands for the
# simulations only
plot_ball_count_comparison <- function(df, occupancy, tag = NULL) {
  df <- df[df$occupancy == occupancy, ]
  ggplot(df, aes(x = r, y = mean, color = source)) +
    geom_line(linewidth = 1.2) +
    geom_ribbon(data = df[df$source != "Data", ],
                aes(ymin = mean - 1.96 * se, ymax = mean + 1.96 * se, fill = source),
                alpha = 0.18, color = NA, show.legend = FALSE) +
    scale_color_brewer(palette = "Dark2") +
    scale_fill_brewer(palette = "Dark2", drop = FALSE) + # keep the colours of the lines
    log_log_axes() +
    labs(x = "Radius [m]", y = occupancy_axis_title("Avg. C(r)", occupancy), tag = tag) +
    theme_validation(legend_inside = c(0.97, 0.05), legend_just = c("right", "bottom"),
                     legend_alpha = 1, base_size = 16)
}

# add an inset zooming into r in `bounds` (in meters), placed at `position`
# (left, bottom, right, top) relative to the panel
add_inset <- function(p, bounds, position = c(0.05, 0.6, 0.4, 0.97)) {
  inset <- (p %+% p$data[p$data$r >= bounds[1] & p$data$r <= bounds[2], ]) +
    labs(x = NULL, y = NULL, tag = NULL) +
    theme_minimal(base_size = 10) +
    theme(
      legend.position  = "none",
      axis.text        = element_text(size = 9),
      panel.grid.minor = element_blank(),
      plot.background  = element_rect(fill = "white", color = "black", linewidth = 0.4),
      plot.margin      = margin(2, 2, 2, 2)
    )
  rng <- ggplot_build(p)$layout$panel_params[[1]]
  p + annotation_custom(
    ggplotGrob(inset),
    xmin = rng$x.range[1] + position[1] * diff(rng$x.range),
    xmax = rng$x.range[1] + position[3] * diff(rng$x.range),
    ymin = rng$y.range[1] + position[2] * diff(rng$y.range),
    ymax = rng$y.range[1] + position[4] * diff(rng$y.range)
  )
}

# average L-function for one correction ("hom." or "inhom."), optionally with
# an inset zooming into r in `inset_bounds` (in meters). Further arguments go
# to theme_validation().
plot_L_comparison <- function(df, correction = "hom.", tag = NULL, inset_bounds = NULL,
                              inset_position = c(0.05, 0.6, 0.4, 0.97), ...) {
  p <- ggplot(df[df$correction == correction, ], aes(x = r, y = L, color = source)) +
    geom_line(linewidth = 1.2) +
    scale_color_brewer(palette = "Dark2") +
    labs(x = "r [m]", y = paste0("L(r) (", correction, ")"), tag = tag) +
    theme_validation(legend_inside = c(0.98, 0.4), legend_just = c("right", "top"), ...)
  if (is.null(inset_bounds)) p else add_inset(p, inset_bounds, inset_position)
}

# k-NN densities (linear and log distance axis) and ball counts, one plot per
# occupancy class; with `tags = TRUE` enumerated a)-c) for 1-NN, d)-f) for
# 2-NN and g)-i) for C(r)
make_occupancy_comparison_plots <- function(metrics, tags = FALSE) {
  plots <- list()
  for (i in 1:3) {
    occ <- c("low", "med", "high")[i]
    for (k in 1:2) {
      df   <- metrics[[paste0("nn", k, "_density")]]
      name <- paste0("nn", k, "_density_", occ)
      tag  <- panel_tag(paste0(letters[3 * (k - 1) + i], ")"), tags)
      plots[[name]] <- plot_nn_density_comparison(df, occ, k = k, tag = tag)
      plots[[paste0(name, "_log")]] <- plot_nn_density_comparison(df, occ, k = k, xlog = TRUE,
                                                                  tag = tag)
    }
    plots[[paste0("ball_count_", occ)]] <-
      plot_ball_count_comparison(metrics$ball_count_occ, occ,
                                 tag = panel_tag(paste0(letters[6 + i], ")"), tags))
  }
  plots
}

# Build all comparison plots from the list returned by
# compute_validation_metrics() with several models. Returns a named list of
# ggplot objects; `tags = TRUE` enumerates the panels.
make_comparison_plots <- function(metrics, tags = FALSE) {
  c(
    make_occupancy_comparison_plots(metrics, tags),
    list(
      L_hom   = plot_L_comparison(metrics$L, "hom.", tag = panel_tag("a)", tags),
                                  inset_bounds = c(0.25, 0.6), text_size = 14, tag_size = 16),
      L_inhom = plot_L_comparison(metrics$L, "inhom.", tag = panel_tag("b)", tags),
                                  inset_bounds = c(0.25, 0.6), text_size = 14, tag_size = 16)
    )
  )
}

# Save each plot as <out_dir>/<prefix>_<name>.pdf
save_validation_plots <- function(plots, out_dir, prefix, width = 6, height = 4) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  for (name in names(plots)) {
    ggsave(file.path(out_dir, paste0(prefix, "_", name, ".pdf")), plots[[name]],
           width = width, height = height)
  }
}
