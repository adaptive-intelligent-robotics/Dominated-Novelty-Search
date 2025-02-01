import pickle
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

grid_file_name = "multirun/2024-12-19/03-00-48/0/population.pickle"
adaptive_file_name = "multirun/2024-12-19/00-12-03/0/population.pickle"
adaptive_centroids_file_name = "multirun/2024-12-19/00-50-47/0/population.pickle"


with open(grid_file_name, "rb") as f:
    grid_population = pickle.load(f)

with open(adaptive_file_name, "rb") as f:
    adaptive_population = pickle.load(f)

with open(adaptive_centroids_file_name, "rb") as f:
    adaptive_centroids_population = pickle.load(f)


# Create figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Plot grid population
scatter1 = ax1.scatter(grid_population.descriptors[:, 0], grid_population.descriptors[:, 1],
                      c=grid_population.fitnesses, cmap="viridis")
fig.colorbar(scatter1, ax=ax1)
ax1.set_title(f"Grid Population\n({len(grid_population.descriptors)} individuals)")
ax1.set_xlabel("BD 1")
ax1.set_ylabel("BD 2")

# Plot adaptive population 
scatter2 = ax2.scatter(adaptive_population.descriptors[:, 0], adaptive_population.descriptors[:, 1],
                      c=adaptive_population.fitnesses, cmap="viridis")
fig.colorbar(scatter2, ax=ax2)
ax2.set_title(f"Adaptive Population\n({len(adaptive_population.descriptors)} individuals)")
ax2.set_xlabel("BD 1")
ax2.set_ylabel("BD 2")

# Plot adaptive centroids population
scatter3 = ax3.scatter(adaptive_centroids_population.descriptors[:, 0], adaptive_centroids_population.descriptors[:, 1],
                      c=adaptive_centroids_population.fitnesses, cmap="viridis")
fig.colorbar(scatter3, ax=ax3)
ax3.set_title(f"Adaptive Centroids Population\n({len(adaptive_centroids_population.descriptors)} individuals)")
ax3.set_xlabel("BD 1")
ax3.set_ylabel("BD 2")

plt.tight_layout()
plt.savefig("comparison_scatter_plots.png")
plt.close()


# plot kheperax maze
import numpy as np
adaptive_centroids_file_name = "multirun/2024-12-01/04-57-25/0/population.pickle"

with open(adaptive_centroids_file_name, "rb") as f:
    adaptive_centroids_population = pickle.load(f)

def plot_kheperax_maze(population, min_bd, max_bd, num_bd_points: int) -> None:
    #population.descriptors is a 2d array of shape (num_individuals, num_points_desc) where num_points_desc is (x_1, y_1, x_2, y_2, ...)
    #we want to plot the positions of the individuals on the maze after a number of timesteps (i.e. x_i, y_i)
    colors = plt.cm.magma(jnp.linspace(0, 1, num_bd_points // 2))
    for i in range(0, num_bd_points, 2):
        plt.scatter(population.descriptors[:, i], population.descriptors[:, i+1], 
                    color=colors[i // 2], label=f'Timestep {i // 2 + 1}', s=10, alpha=0.7)

    plt.xlabel("BD 1")
    plt.ylabel("BD 2")
    plt.title("Scatter plot of the archive over the BD space")
    plt.gcf().set_dpi(300)  # Increase resolution of the image
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Adjust legend position
    plt.tight_layout()  # Adjust layout to prevent legend cutoff
    return plt

def plot_kheperax_maze_arrows(population, min_bd, max_bd, num_bd_points: int) -> None:
    # Plot arrows showing the movement of each individual over the timesteps
    #descriptors is a 2d array of shape (num_individuals, num_points_desc)
    # For each individual, it shows the x,y positions at each timestep.
    colors = plt.cm.magma(np.linspace(0, 1, num_bd_points // 2))
    print(population.descriptors.shape)
    for j in range(0, 100):
        for i in range(2, num_bd_points, 2):
            descriptors = population.descriptors[j]
            plt.arrow(descriptors[i-2], descriptors[i-1],
                      descriptors[i] - descriptors[i-2],
                      descriptors[i+1] - descriptors[i-1], 
                      color=colors[i // 2], 
                      label=f'Timestep {i // 2 + 1}' if j == 2 else "",
                      head_width=0.02, head_length=0.05, alpha=0.5)  # Increase arrow head size
    
    plt.xlabel("BD 1")
    plt.ylabel("BD 2")
    plt.title("Scatter plot of the archive over the BD space with arrows")
    plt.gcf().set_dpi(300)  # Increase resolution of the image
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Move legend outside the figure
    plt.gcf().set_size_inches(12, 5)  # Make the image wider
    plt.tight_layout()  # Adjust layout to prevent legend cutoff
    return plt


fig = plot_kheperax_maze(adaptive_centroids_population, 0, 1, 20)
plt.savefig("kheperax_maze.png")
plt.close()

fig_arrows = plot_kheperax_maze_arrows(adaptive_centroids_population, 0, 1, 20)
plt.savefig("kheperax_maze_arrows.png")
plt.close()