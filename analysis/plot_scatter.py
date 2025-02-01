#Maxence Faldor, Ryan Bahlous-Boldi

import pickle
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from qdax.utils.plotting import get_voronoi_finite_polygons_2d
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib

import os
import glob
import yaml

#so far: ant_omni_nobound, kheperax
to_plot = "kheperax"

def get_latest_run_for_env(env_name, base_path="./multirun"):
    """
    Get the latest single run directory for a given environment name.

    :param env_name: The name of the environment to search for.
    :param base_path: The base path where the runs are stored.
    :return: The path to the latest run directory for the specified environment.
    """
    # Get all run directories
    run_dirs = glob.glob(f"{base_path}/*/*/0/")
    run_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)  # Sort by modification time, latest first

    for run_dir in run_dirs:
        config_path = os.path.join(run_dir, ".hydra", "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                if config["env"]["name"] == env_name:
                    return run_dir

def get_latest_run_for_env_and_pop(env_name, pop_name, base_path="./multirun", num_points_desc=30):
    run_dirs = glob.glob(f"{base_path}/*/*/0/")
    run_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)  # Sort by modification time, latest first

    for run_dir in run_dirs:
        config_path = os.path.join(run_dir, ".hydra", "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                if env_name == "maze_min_energy":
                    if config["env"]["name"] == env_name and config["population"]["name"] == pop_name:
                        if config["env"]["num_points_desc"] == num_points_desc:
                            return run_dir
                else:
                    if config["env"]["name"] == env_name and config["population"]["name"] == pop_name:
                        return run_dir


def plot_kheperax_maze(population, min_bd, max_bd, ax, num_bd_points: int) -> None:
    #population.descriptors is a 2d array of shape (num_individuals, num_points_desc) where num_points_desc is (x_1, y_1, x_2, y_2, ...)
    #we want to plot the positions of the individuals on the maze after a number of timesteps (i.e. x_i, y_i)
    colors = plt.cm.viridis(jnp.linspace(0, 1, num_bd_points // 2))
    for i in range(0, num_bd_points, 2):
        ax.scatter(population.descriptors[:, i], population.descriptors[:, i+1], 
                   color=colors[i // 2], label=f'Timestep {i // 2 + 1}', s=10, alpha=0.7)

    #x, y lims are min_bd, max_bd
    ax.set_xlim(min_bd, max_bd)
    ax.set_ylim(min_bd, max_bd)
    #ax.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Adjust legend position

def plot_kheperax_maze_arrows(population, min_bd, max_bd, ax, num_bd_points: int) -> None:
    # Plot arrows showing the movement of each individual over the timesteps
    #descriptors is a 2d array of shape (num_individuals, num_points_desc)
    # For each individual, it shows the x,y positions at each timestep.
    colors = plt.cm.viridis(np.linspace(0, 1, num_bd_points // 2))
    print(population.descriptors.shape)
    for j in range(0, 100):
        for i in range(2, num_bd_points, 2):
            descriptors = population.descriptors[j]
            ax.arrow(descriptors[i-2], descriptors[i-1],
                     descriptors[i] - descriptors[i-2],
                     descriptors[i+1] - descriptors[i-1], 
                     color=colors[i // 2], 
                     label=f'Timestep {i // 2 + 1}' if j == 2 else "",
                     head_width=0.02, head_length=0.05, alpha=0.5)  # Increase arrow head size
    
    ax.set_xlabel("BD 1")
    ax.set_ylabel("BD 2")

    ax.set_xlim(min_bd, max_bd)
    ax.set_ylim(min_bd, max_bd)
    #ax.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Move legend outside the figure


def plt_scatter_population(ax, population, title, cmap):
    print(f"population fitnesses: {population.fitnesses}")
    s = ax.scatter(population.descriptors[:, 0], population.descriptors[:, 1],
                      c=population.fitnesses, cmap=cmap)
    ax.set_title(title)
    ax.set_aspect("equal")
    


def plot_2d_repertoire(ax, repertoire, minval, maxval, vmin, vmax, display_descriptors=False, cbar=False):
    """Plot a 2d map elites repertoire on the given axis."""
    assert repertoire.centroids.shape[-1] == 2, "Descriptor space must be 2d"

    repertoire_empty = repertoire.fitnesses == -jnp.inf

    # Set axes limits
    ax.set_xlim(minval[0], maxval[0])
    ax.set_ylim(minval[1], maxval[1])
    ax.set(adjustable="box", aspect="equal")

    # Create the regions and vertices from centroids
    regions, vertices = get_voronoi_finite_polygons_2d(repertoire.centroids)

    # Colors
    cmap = matplotlib.cm.viridis
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Fill the plot with contours
    for region in regions:
        polygon = vertices[region]
        ax.fill(*zip(*polygon), alpha=0.05, edgecolor="black", facecolor="white", lw=1)

    # Fill the plot with the colors
    for idx, fitness in enumerate(repertoire.fitnesses):
        if fitness > -jnp.inf:
            region = regions[idx]
            polygon = vertices[region]
            ax.fill(*zip(*polygon), alpha=0.8, color=cmap(norm(fitness)))

    # if descriptors are specified, add points location
    if display_descriptors:
        descriptors = repertoire.descriptors[~repertoire_empty]
        ax.scatter(
            descriptors[:, 0],
            descriptors[:, 1],
            c=repertoire.fitnesses[~repertoire_empty],
            cmap=cmap,
            s=10,
            zorder=0,
        )

    # Aesthetic
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        cbar.ax.tick_params()
    ax.set_aspect("equal")

    return ax

if __name__ == "__main__":
        #avoid type3 fonts in matplotlib
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rc("font", size=16)

    if to_plot == "ant_omni_nobound":

        env_name = "ant_omni_nobound"
        grid_location = get_latest_run_for_env_and_pop(env_name, "grid_population")
        adaptive_location = get_latest_run_for_env_and_pop(env_name, "adaptive_population")

        print(grid_location)
        print(adaptive_location)

        grid_file_name = os.path.join(grid_location, "population.pickle")
        adaptive_file_name = os.path.join(adaptive_location, "population.pickle")

        print(grid_file_name)
        print(adaptive_file_name)

        with open(grid_file_name, "rb") as f:
            grid_population = pickle.load(f)

        with open(adaptive_file_name, "rb") as f:
            adaptive_population = pickle.load(f)

        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

        # Colors and normalization
        cmap = matplotlib.cm.viridis
        norm = matplotlib.colors.Normalize(vmin=0, vmax=4500)  # Set normalization range to match fitness range

        # Plot populations
        plt_scatter_population(ax1, grid_population, "MAP-Elites", cmap)
        plt_scatter_population(ax2, adaptive_population, "Dominated Novelty Search", cmap)

        # Add colorbar to the right axis only
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)  # Use the defined normalization
        cbar.ax.tick_params()

        # Draw a red rectangle on ax2 from -30,30 on both axes
        ax2.add_patch(plt.Rectangle((-30, -30), 60, 60, color="red", fill=False, lw=2))

        plt.tight_layout()
        plt.savefig(f"figs/{env_name}_comparison_scatter_plots.png")
        plt.close()
    
    elif to_plot == "kheperax":
        # get the latest run for kheperax
        kheperax_grid_location = get_latest_run_for_env_and_pop("maze_min_energy", "grid_population", num_points_desc=30)
        kheperax_file_name = os.path.join(kheperax_grid_location, "population.pickle")

        kheperax_adaptive_location = get_latest_run_for_env_and_pop("maze_min_energy", "adaptive_population", num_points_desc=30)
        kheperax_adaptive_file_name = os.path.join(kheperax_adaptive_location, "population.pickle")

        with open(kheperax_file_name, "rb") as f:
            kheperax_grid_population = pickle.load(f)

        with open(kheperax_adaptive_file_name, "rb") as f:
            kheperax_adaptive_population = pickle.load(f)

        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

        # Plot populations
        plot_kheperax_maze(kheperax_grid_population, 0, 1, ax1, 30)
        ax1.set_title("MAP-Elites")
        plot_kheperax_maze(kheperax_adaptive_population, 0, 1, ax2, 30)
        ax2.set_title("Dominated Novelty Search")

        # Add colorbar to the right axis only
        norm = matplotlib.colors.Normalize(vmin=0, vmax=250)  # Adjust normalization range to match new timestep range
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis), cax=cax)
        cbar.set_label('Timestep in Simulation', rotation=270)
        cbar.ax.get_yaxis().labelpad = 15  # Adjust the label padding if necessary

        fig.tight_layout()
        plt.savefig(f"figs/kheperax_comparison_scatter_plots.png")
        plt.close()


