import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.dates as mdates
import seaborn as sns

from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from pathlib import Path


HERE = Path(__file__).resolve().parent

def _add_enumeration_label(fig, ax, enumeration, fontsize=18):
    """Draw a panel label (e.g. 'a)') above and to the left of the y-axis label."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    ylab_bb = ax.yaxis.label.get_window_extent(renderer=renderer)
    ylab_bb_fig = fig.transFigure.inverted().transform(ylab_bb)
    y_label_x_fig = (ylab_bb_fig[0, 0] + ylab_bb_fig[1, 0]) / 2 + 0.01

    title_bb = ax.title.get_window_extent(renderer=renderer)
    title_bb_fig = fig.transFigure.inverted().transform(title_bb)
    title_y_fig = title_bb_fig[1, 1]

    fig.text(y_label_x_fig, title_y_fig, f'{enumeration}', fontsize=fontsize, ha='right', va='top')


def plot_background(labels=True, title=False, save_path=None, enumeration=None):
    """Plot the station background image with optional area labels and title."""
    fig, ax = plt.subplots(figsize=(16, 2.6), dpi=300)
    bg = mpimg.imread(HERE.parent / "data" / "background.png")
    # Display the background image on the axis
    ax.imshow(bg, extent=[-1039/500, 80091/500, (-3539 - 4136)/500, (23909 - 4136)/500], alpha=1)
    if title:
        ax.set_title("Eindhoven Central Station - Platform 2 (Tracks 3 and 4)", fontsize=20)

    if labels:
        alpha = 0.5
        box_props = dict(facecolor='white', alpha=alpha, edgecolor='none', boxstyle='round,pad=0.3')
        ax.text(10, 12, "Entrance", fontsize=14, color='black', bbox=box_props)
        ax.text(77, 12, "Café / Waiting Area", fontsize=14, color='black', bbox=box_props)
        ax.text(129, 14, "Bench", fontsize=14, color='black', bbox=box_props)
        ax.text(148, 12, "Building", fontsize=14, color='black', bbox=box_props)

    ax.set_xlabel('X [m]', fontsize=16)
    ax.set_ylabel('Y [m]', fontsize=16)
    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([f'{int(tick/2)}' for tick in x_ticks], fontsize=14)
    ax.set_yticklabels([f'{int(tick/2)}' for tick in y_ticks], fontsize=14)
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 26)

    if enumeration:
        _add_enumeration_label(fig, ax, enumeration)

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


def plot_aggregate_heatmap(
    samples,
    grid,
    background=True,
    bandwidth=None,
    cmap=sns.color_palette("rocket", as_cmap=True),
    enumeration=None,
    title=False,
    normalize=True,
    crop=False,
    colorbar=True,
    save_path=None,
):
    """
    Plot an aggregate 2D KDE heatmap from a list of samples.
    Each sample is a DataFrame with columns ["x_position_mm", "y_position_mm"].
    """
    # concatenate all pedestrian positions
    X = np.vstack([s[["x_position_mm", "y_position_mm"]].to_numpy() for s in samples])
    grid_coords = grid[['x', 'y']].to_numpy()  # grid centers

    # KDE on pooled data
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(X)
    logprob = kde.score_samples(grid_coords)
    kde_values = np.exp(logprob)

    # reshape into grid
    kde_values = kde_values.reshape(len(np.unique(grid['y'])), len(np.unique(grid['x'])))

    # normalize to get a better comparison
    if normalize:
        kde_values = np.clip(kde_values, None, np.percentile(kde_values, 99))

    # figure setup
    aspect_ratio = 26 / 160
    width = 16
    height = width * aspect_ratio
    fig, ax = plt.subplots(figsize=(width, height), dpi=300, constrained_layout=True)

    if background:
        bg = mpimg.imread(HERE.parent / "data" / "background.png")
        ax.imshow(bg, extent=[-1039, 80091, (-3539-4136), (23909-4136)], alpha=1, zorder=0)

    num_total = sum(len(s['object_identifier'].unique()) for s in samples)
    if title:
        ax.set_title(f'Aggregate heatmap of {len(samples)} samples ({num_total} pedestrians total)', fontsize=18)

    im = ax.imshow(kde_values, cmap=cmap,
              extent=[grid['x'].min(), grid['x'].max(), grid['y'].min(), grid['y'].max()],
              origin='lower', aspect='equal', zorder=1, alpha=0.75)

    ax.set_xlabel('X [m]', fontsize=16)
    ax.set_ylabel('Y [m]', fontsize=16)
    x_ticks = ax.get_xticks()
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{int(tick/1000)}' for tick in x_ticks], fontsize=14)
    ax.set_yticks([0, 5000, 10000])
    ax.set_yticklabels(['0', '5', '10'], fontsize=14)
    if crop:
        plt.xlim(0, 50000)
        plt.ylim(0, 16772-4136)
        if title is not None:
            ax.set_title(title, fontsize=16)
        else:
            ax.set_title(f'Aggregate heatmap of {len(samples)} samples ({num_total} pedestrians total)', fontsize=16)
        ax.set_xlabel('X [m]', fontsize=16)
        ax.set_ylabel('Y [m]', fontsize=16)
        ax.set_xticklabels([f'{int(tick/1000)}' for tick in x_ticks], fontsize=14)
        ax.set_yticklabels(['0', '5', '10'], fontsize=14)
    else:
        plt.xlim(0, 80091)
        plt.ylim(0, 16772-4136)

    if colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.01)
        cbar.set_ticks([np.min(kde_values), np.max(kde_values)])
        cbar.set_ticklabels(["LO", "HI"])
        cbar.ax.tick_params(labelsize=14)

    if enumeration:
        _add_enumeration_label(fig, ax, enumeration)

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", transparent=True)
    plt.show()


def plot_points(
    df,
    x_label='x_position_mm',
    y_label='y_position_mm',
    background=True,
    frac_standing=False,
    with_walking=False,
    velocity=False,
    vel_lookup=None,
    arrow_fixed_length=False,
    title=False,
    rectangle=None,
    enumeration=None,
    save_path=None,
):
    """Plot pedestrian positions for each timestamp, optionally split into standing/walking with velocity arrows."""
    for (time), grp in df.groupby("time_ms"):
        fig, ax = plt.subplots(figsize=(16, 2.6), dpi=300)
        if not with_walking:
            ax.scatter(grp[x_label]/500, grp[y_label]/500, color='red', s=15)
        else:
            standing = grp[grp["stand_or_walk"] == 0]
            ax.scatter(standing[x_label]/500, standing[y_label]/500, c="red", s=15, marker="o")

            walking = grp[grp["stand_or_walk"] == 1]
            if velocity:
                vel_cols = ["x_vel_SG", "y_vel_SG"]
                if set(vel_cols).issubset(walking.columns):
                    walking_with_vel = walking
                else:
                    if vel_lookup is None:
                        raise ValueError("Velocities not in df; provide vel_lookup")
                    vel_current = vel_lookup[vel_lookup["time_ms"] == time]
                    walking_with_vel = walking.merge(
                        vel_current[["object_identifier"] + vel_cols],
                        on="object_identifier",
                        how="left",
                    )

                # Compute direction-only arrows if requested
                if arrow_fixed_length:
                    # Compute the norm of each velocity vector
                    vx = walking_with_vel['x_vel_SG'].values
                    vy = walking_with_vel['y_vel_SG'].values
                    norm = np.sqrt(vx**2 + vy**2)
                    norm[norm == 0] = 1  # avoid division by zero
                    dx = vx / norm  # normalized to length 1
                    dy = vy / norm
                    # Optional: scale for visibility
                    dx *= 2
                    dy *= 2
                else:
                    dx = walking_with_vel['x_vel_SG'].values
                    dy = walking_with_vel['y_vel_SG'].values

                # Plot arrows
                ax.quiver(
                    walking_with_vel[x_label]/500,
                    walking_with_vel[y_label]/500,
                    dx,
                    dy,
                    angles='xy',
                    scale_units='xy',
                    scale=1,
                    color='k',
                    width=0.003
                )
            else:
                ax.scatter(walking[x_label]/500, walking[y_label]/500, c="black", s=15, marker="D")

        if background:
            bg = mpimg.imread(HERE.parent / "data" / "background.png")
            # Display the background image on the axis
            ax.imshow(bg, extent=[-1039/500, 80091/500, (-3539 - 4136)/500, (23909 - 4136)/500], alpha=0.75)
        num = len(grp['object_identifier'].unique())
        if title:
            if frac_standing:
                frac = grp['frac_standing'].iloc[0]
                ax.set_title(f'Sample with {num} pedestrians - Fraction waiting: {frac:.2f}', fontsize=18)
            else:
                ax.set_title(f'Sample with {num} pedestrians', fontsize=18)
        ax.set_xlabel('X [m]', fontsize=16)
        ax.set_ylabel('Y [m]', fontsize=16)
        x_ticks = ax.get_xticks()
        y_ticks = ax.get_yticks()
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels([f'{int(tick/2)}' for tick in x_ticks], fontsize=14)
        ax.set_yticklabels([f'{int(tick/2)}' for tick in y_ticks], fontsize=14)
        ax.set_xlim(0, 160)
        ax.set_ylim(0, 26)
        if enumeration:
            _add_enumeration_label(fig, ax, enumeration)
        if rectangle is not None:
            xmin, xmax, ymin, ymax = rectangle # mm, y includes the +4136 shift
            rect = Rectangle(
                (xmin / 500, (ymin - 4136) / 500), # lower-left corner
                (xmax - xmin) / 500, # width
                (ymax - ymin) / 500, # height
                linewidth=1, edgecolor="r", facecolor="none",
            )
            ax.add_patch(rect)
        if save_path:
            plt.savefig(save_path, format="pdf", bbox_inches="tight")
        plt.show()


def plot_count_histogram(
    df,
    inset=True,
    bins=100,
    kde=False,
    zoom_limit=None,
    enumeration=None,
    save_path=None,
    add_stats=False,
    title=None,
):
    """
    Plot histogram of number of pedestrians per snapshot.
    If inset=True and zoom_limit is provided, the main plot shows the zoomed distribution,
    and the full distribution is shown in an inset.
    """
    # Make sure each snapshot is counted only once
    snapshot_counts = df.drop_duplicates(subset=["time_ms"])["occupancy"]

    # Split data
    if inset and zoom_limit is not None:
        snapshot_counts_small = snapshot_counts[snapshot_counts <= zoom_limit]
        main_data = snapshot_counts_small
        inset_data = snapshot_counts
    else:
        main_data = snapshot_counts
        inset_data = None

    # --- Main plot ---
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    sns.histplot(main_data, bins=bins, kde=kde, ax=ax, color="#1B9E77FF", edgecolor=None)

    ax.set_xlabel(r"$N(\mathbf{x})$", fontsize=16)
    ax.set_ylabel("Count", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)

    # Title
    if inset and zoom_limit is not None and title is not None:
        ax.set_title(f"Zoomed: ≤ {zoom_limit} people", fontsize=16)
    elif title is not None:
        ax.set_title(title, fontsize=16)

    # Stats box (based on main data)
    if add_stats:
        mode_val = main_data.mode().iloc[0]
        mean_val = main_data.mean()
        median_val = main_data.median()
        std_val = main_data.std()
        n_snapshots = main_data.shape[0]
        stats_text = (
            f"Mode no. of peds: {mode_val}\n"
            f"Mean no. of peds: {mean_val:.2f}\n"
            f"Median no. of peds: {median_val:.2f}\n"
            f"Std dev: {std_val:.2f}\n"
            f"No. of snapshots: {n_snapshots}"
        )
        ax.text(
            0.62, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
        )

    # --- Inset: full distribution ---
    if inset and zoom_limit is not None:
        ax_inset = inset_axes(ax, width="50%", height="50%",
                              loc="upper right", borderpad=2)

        sns.histplot(inset_data, bins=bins, kde=kde, ax=ax_inset, color="#D95F02FF", edgecolor=None)

        ax_inset.set_title("Zoomed out: full distribution", fontsize=14)
        ax_inset.tick_params(axis='both', labelsize=12)

        # Visual indicators of zoom region
        ax_inset.axvline(zoom_limit, color='k', linestyle='--', linewidth=1)
        ax_inset.axvspan(0, zoom_limit, color='grey', alpha=0.2)

        # compress vertical scale for long tails
        ax_inset.set_yscale('log')

        ax_inset.set_xlabel("")
        ax_inset.set_ylabel("")

    # Enumeration label
    if enumeration:
        _add_enumeration_label(fig, ax, enumeration)

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")

    plt.show()


def plot_count_vs_frac(
    df,
    enumeration=None,
    save_path=None,
    title=None,
    crop_limit=None,
    style="hexbin",
):
    """Plot the relationship between platform occupancy (occupancy) and the fraction of pedestrians waiting."""
    snapshot_data = df.drop_duplicates(subset=["time_ms"])[["occupancy", "frac_standing"]]
    if crop_limit is not None:
        snapshot_data = snapshot_data[snapshot_data["occupancy"] <= crop_limit]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    if style == "scatter":
        sns.scatterplot(data=snapshot_data, x="occupancy", y="frac_standing", ax=ax)
    elif style == "hexbin":
        if crop_limit is not None:
            ax.set_aspect(crop_limit, adjustable="datalim")
            im = ax.hexbin(snapshot_data["occupancy"], snapshot_data["frac_standing"], gridsize=20, cmap=sns.color_palette("mako", as_cmap=True))
        else:
            max_count = snapshot_data["occupancy"].max()
            ax.set_aspect(max_count, adjustable="datalim")
            im = ax.hexbin(snapshot_data["occupancy"], snapshot_data["frac_standing"], gridsize=40, cmap=sns.color_palette("mako", as_cmap=True))
        cbar = plt.colorbar(im)
        cbar.set_label("No. of snapshots", fontsize=14)
        cbar.ax.tick_params(labelsize=12)
    if title is not None:
        ax.set_title(title, fontsize=16)
    ax.set_xlabel(r"No. of people on the platform", fontsize=16)
    ax.set_ylabel("Fraction of people waiting", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)

    if enumeration:
        _add_enumeration_label(fig, ax, enumeration)

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")

    plt.show()


def aggregate_discrete_counts(samples, grid, n_rows=26, n_cols=160):
    """
    Convert a list of discrete samples (lists of flattened indices)
    into a (n_rows x n_cols) matrix of total counts.
    """
    agg = np.zeros((n_rows, n_cols), dtype=int)

    # Precompute mapping index -> (row, col)
    rows = grid["y_rounded"].to_numpy()
    cols = grid["x_rounded"].to_numpy()

    for sample in samples:
        r = rows[sample]
        c = cols[sample]
        # add 1 to each visited grid cell
        np.add.at(agg, (r, c), 1)

    return agg


def plot_discrete_aggregate_heatmap(
    samples,
    grid,
    background=True,
    cmap=sns.color_palette("rocket", as_cmap=True),
    enumeration=None,
    title=None,
    normalize=True,
    colorbar=True,
    save_path=None
):
    """
    Plot an aggregate heatmap for a discrete point process on a 26x160 grid.
    """
    # Aggregate counts
    agg = aggregate_discrete_counts(samples, grid, n_rows=26, n_cols=160)
    values = agg.astype(float)

    # Optional clipping for robust visualization
    if normalize:
        vmax = np.percentile(values, 99)
        values = np.clip(values, 0, vmax)

    # Figure setup
    aspect_ratio = 26 / 160
    width = 16
    height = width * aspect_ratio
    fig, ax = plt.subplots(figsize=(width, height), dpi=300, constrained_layout=True)

    if background:
        bg = mpimg.imread(HERE.parent / "data" / "background.png")
        # Display the background image on the axis
        ax.imshow(bg, extent=[-1039/500, 80091/500, (-3539-4136)/500, (23909-4136)/500], alpha=1)

    im = ax.imshow(values,
                cmap=cmap,
                origin='lower',
                extent=[0, 160, 0, 26],  # match grid dimensions
                aspect='equal',
                zorder=1,
                alpha=0.75
                )

    if title:
        if isinstance(title, str):
            ax.set_title(title, fontsize=18)
        else:  # title == True
            total_points = sum(len(s) for s in samples)
            ax.set_title(f"Aggregate discrete heatmap ({len(samples)} samples, {total_points} points)",
                        fontsize=18)
    # If title is False, do nothing

    if colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.01)
        cbar.set_ticks([np.min(values), np.max(values)])
        cbar.set_ticklabels(["LO", "HI"])
        cbar.ax.tick_params(labelsize=14)

    ax.set_xlim(0, 160)
    ax.set_ylim(0, 26)
    ax.set_xlabel('X [m]', fontsize=18)
    ax.set_ylabel('Y [m]', fontsize=18)
    ax.set_xticks(np.arange(0, 160 + 1, 20))
    ax.set_yticks(np.arange(0, 26 + 1, 10))
    ax.set_xticklabels([
        f'{int(tick/2)}' if i % 1 == 0 else ''
        for i, tick in enumerate(ax.get_xticks())
    ], fontsize=16)
    ax.set_yticklabels([
        f'{int(tick/2)}' if i % 1 == 0 else ''
        for i, tick in enumerate(ax.get_yticks())
    ], fontsize=16)

    if enumeration:
        _add_enumeration_label(fig, ax, enumeration, fontsize=20)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_sample(grid, sample, title=None, background=True, show_grid=True, enumeration=None, rectangle=None):
    """
    Plot a single discretized sample on the grid
    """
    x_coords = grid["x_rounded"].iloc[sample] + 0.5
    y_coords = grid["y_rounded"].iloc[sample] + 0.5

    fig, ax = plt.subplots(figsize=(16, 2.6), dpi=300)
    ax.scatter(x_coords, y_coords, s=15, c="blue")

    if background:
        bg = mpimg.imread(HERE.parent / "data" / "background.png")
        # Display the background image on the axis
        ax.imshow(bg, extent=[-1039/500, 80091/500, (-3539-4136)/500, (23909-4136)/500], alpha=0.75)
    if rectangle:
        xmin, xmax, ymin, ymax = rectangle # mm, y includes the +4136 shift
        rect = Rectangle(
            (xmin, ymin), # lower-left corner
            (xmax - xmin), # width
            (ymax - ymin), # height
            linewidth=1, edgecolor="r", facecolor="none",
        )
        ax.add_patch(rect)

    num = len(sample)
    if title is not None:
        ax.set_title(f"{title} - {num} points", fontsize=20)
    else:
        ax.set_title(f"Simulated sample - {num} points", fontsize=20)

    if show_grid:
        ax.grid(True, which='both', linestyle=':', linewidth=0.5, color="k")

    ax.set_xlim(0, 160)
    ax.set_ylim(0, 26)
    ax.set_xlabel('X [m]', fontsize=18)
    ax.set_ylabel('Y [m]', fontsize=18)
    ax.set_xticks(np.arange(0, 160 + 1, 1))
    ax.set_yticks(np.arange(0, 26 + 1, 1))
    ax.set_xticklabels([
        f'{int(tick/2)}' if i % 5 == 0 else ''
        for i, tick in enumerate(ax.get_xticks())
    ], fontsize=16)
    ax.set_yticklabels([
        f'{int(tick/2)}' if i % 5 == 0 else ''
        for i, tick in enumerate(ax.get_yticks())
    ], fontsize=16)

    if enumeration:
        fig.canvas.draw()
        # Get bounding box of the y-axis label in display coordinates
        ylab = ax.yaxis.label
        ylab_bb = ylab.get_window_extent(renderer=fig.canvas.get_renderer())
        # Convert it to figure coordinates
        ylab_bb_fig = fig.transFigure.inverted().transform(ylab_bb)
        # Get the x coordinate of the center/left side of the y-axis label box
        y_label_x_fig = (ylab_bb_fig[0, 0] + ylab_bb_fig[1, 0]) / 2 + 0.01
        # Get title y position in figure coordinates
        title = ax.title
        title_bb = title.get_window_extent(renderer=fig.canvas.get_renderer())
        title_bb_fig = fig.transFigure.inverted().transform(title_bb)
        title_y_fig = title_bb_fig[1, 1]  # top of the title bbox in figure coords
        fig.text(y_label_x_fig, title_y_fig, f'{enumeration}', fontsize=20,
                ha='right', va='top')

    plt.show()