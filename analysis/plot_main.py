#Maxence Faldor, Ryan Bahlous-Boldi

import os
import glob
import yaml
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter, PercentFormatter

# Define environment and algorithm names
ENV_LIST = ["antblockmany_omni", "ant_omni", "walker2d_uni"]
MAZE_TYPES = [10, 25, 50]

MUJOCO_ENV_NAMES = ["antblockmany_omni", "ant_omni", "ant_omni_nobound", "walker2d_uni"]
METRICS_TO_GRAPH = ["proj_coverage", "proj_max_fitness", "proj_qd_score"]
POPULATION_NAMES = ["DNS (ours)", "Cluster-Elites", "Threshold", "MAP-Elites", "MAP-Elites (Upper Bound)"]

#only dashed for MAP-Elites upper bound
dash_types = {"DNS (ours)": (None, None), "Cluster-Elites": (None, None), "Threshold": (None, None), "MAP-Elites": (None, None), "MAP-Elites (Upper Bound)": (5, 5)}

XLABEL = "Evaluations"
YMAX=1.05

def pretty_env_name(env_name):
    if env_name.startswith("maze_min_energy"):
        if env_name.split('_')[-1] == "aurora":
            return f"Maze (unsupervised)"
        return f"Maze (n={env_name.split('_')[-1]})"
    else:
        return env_name

def load_data(files, metrics_to_graph, population_names, load_type="pickle"):
    print(f"load_data called")
    proj_qd_score = None

    if load_type == "pickle":
        with open(files[0][0] + "metrics.pickle", "rb") as f:
            metrics = pickle.load(f)
            proj_qd_score = metrics["proj_qd_score"]
    elif load_type == "csv":
        with open(files[0][0] + "log.csv", "r") as f:
            metrics = pd.read_csv(f)
            proj_qd_score = metrics["proj_qd_score"]

    values = np.zeros((len(metrics_to_graph), seeds, len(population_names), proj_qd_score.shape[0]))

   
    #deconstruct files
    indices = [0, 1, 2, 3, 4] #graphing order, should match names

    for index, population in zip(indices, files):
        for i, filename in enumerate(population):
            if load_type == "pickle":
                with open(filename + "metrics.pickle", "rb") as f:
                    metrics = pickle.load(f)
                    for j, metric in enumerate(metrics_to_graph):
                        values[j, i, index, :] = metrics[metric]
            elif load_type == "csv":
                with open(filename + "log.csv", "r") as f:
                    metrics = pd.read_csv(f)
                    for j, metric in enumerate(metrics_to_graph):
                        values[j, i, index, :] = metrics[metric]

    data = []
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            for k in range(values.shape[2]):
                for l in range(values.shape[3]):
                    data.append([metrics_to_graph[i], j, k, l, values[i, j, k, l]])

    df = pd.DataFrame(data, columns=["Metric", "Seed", "Population", "Iteration", "Value"])
    population_names = dict(enumerate(population_names))
    print(f"population_names: {population_names}")
    df["Population"] = df["Population"].map(population_names)
    return df

def customize_axis(ax):
    #remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Remove ticks
    # ax.tick_params(axis="y", length=0)

    # Add grid
    ax.grid(which="major", axis="y", color="0.9")
    return ax


def plot_all_environments(env_list, metrics_to_graph, df_dict):
    print(f"plotting all environments")
    # Create subplots
    fig, axes = plt.subplots(nrows=len(env_list), ncols=3, figsize=(15, 3 * len(env_list)))

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))

    #Don't graph ME in aurora
    if "maze_min_energy_aurora" in df_dict:
        df_dict["maze_min_energy_aurora"] = df_dict["maze_min_energy_aurora"][df_dict["maze_min_energy_aurora"]["Population"] != "ME"]

    for row, env_name in enumerate(env_list):

        axes[row, 0].set_ylabel(pretty_env_name(env_name))
        axes[row, 0].yaxis.set_major_formatter(formatter)

        df = df_dict[env_name]
        # QD-Score
        print(f"plotting {env_name} - QD-Score")
        metric_df = df[df["Metric"] == "proj_qd_score"]
        ax = axes[row, 0]
        sns.lineplot(
            data=metric_df,
            x="num_evaluations",
            y="Value",
            hue="Population",
            style="Population",
            dashes=dash_types,
            legend=False,
            estimator=np.median,
            errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
            ax=ax,
        )

        if row == 0:
            axes[row, 0].set_title("Proj. QD Score")

        customize_axis(axes[row, 0])

        # Coverage
        axes[row, 1].set_ylim(0, YMAX)
        axes[row, 1].yaxis.set_major_formatter(PercentFormatter(1))

        print(f"plotting {env_name} - Coverage")
        metric_df = df[df["Metric"] == "proj_coverage"]
        ax = axes[row, 1]
        sns.lineplot(
            data=metric_df,
            x="num_evaluations",
            y="Value",
            hue="Population",
            style="Population",
            dashes=dash_types,
            legend=False,
            estimator=np.median,
            errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
            ax=ax,
        )

        if row == 0:
            axes[row, 1].set_title("Proj. Coverage")

        customize_axis(axes[row, 1])

        #max fitness
        sns.lineplot(
            data=df[df["Metric"] == "proj_max_fitness"],
            x="num_evaluations",
            y="Value",
            hue="Population",
            style="Population",
            dashes=dash_types,
            legend=False,
            estimator=np.median,
            errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
            ax=axes[row, 2],
        )

        if row == 0:
            axes[row, 2].set_title("Proj. Max Fitness")

        customize_axis(axes[row, 2])

         # Remove x-axis labels for all but the last row
        if row != len(ENV_LIST) - 1:
            for col in range(3):
                axes[row, col].set_xlabel('')

    # Set x-axis label only for the bottom row
    for col in range(3):
        axes[-1, col].set_xlabel(XLABEL)
        axes[-1, col].xaxis.set_major_formatter(formatter)

    # Remove y-axis labels for the second and third columns
    for row in range(len(ENV_LIST)):
        axes[row, 1].set_ylabel('')
        axes[row, 2].set_ylabel('')

    # Legend
    fig.legend(ax.get_lines(), POPULATION_NAMES, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncols=len(POPULATION_NAMES), frameon=False)

    # Aesthetic
    fig.align_ylabels(axes)
    fig.tight_layout()

    plt.tight_layout()
    plt.savefig(f"figs/all_environments_metrics_plot_1.png", bbox_inches="tight")
    plt.close()

    #done 
    print(f"done plotting all environments")

if __name__ == "__main__":
    #avoid type3 fonts in matplotlib
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rc("font", size=16)

    df_dict = {}

    for cur_env_name in ENV_LIST:
        # Load data
        files = glob.glob(f"./multirun/*/*/0/")
        files.sort(key=lambda x: os.path.getmtime(x))
        #make sure cur name is in the file, and all other env names are not

        files_to_use = []

        if cur_env_name.startswith("maze_min_energy"):
            num_points_desc = cur_env_name.split("_")[-1]

            for f in files:
                config = yaml.safe_load(open(f"{f}/.hydra/config.yaml", "r"))
                if num_points_desc == "aurora":
                    print(f"using aurora")
                    if config["algo"]["name"] == "aurora":
                        print(f"found aurora file")
                        files_to_use.append(f)
                else:
                    if config["env"]["name"] == "maze_min_energy":
                        npd = config["env"]["num_points_desc"]
                        if npd == int(num_points_desc):
                            files_to_use.append(f)
        else:
            for f in files:
                config = yaml.safe_load(open(f"{f}/.hydra/config.yaml", "r"))
                if cur_env_name == config["env"]["name"]:
                    files_to_use.append(f)

        files = files_to_use

        files_to_use = []
        if cur_env_name in MUJOCO_ENV_NAMES:
            #only use the ones that have policy params [16]
            for f in files:
                config = yaml.safe_load(open(f"{f}/.hydra/config.yaml", "r"))
                if config["algo"]["policy_hidden_layer_sizes"] == [16]:
                    files_to_use.append(f)
        else:
            files_to_use = files

        files = files_to_use

        adaptive_population = [f for f in files if "adaptive_population" in open(f + "/.hydra/config.yaml").read()]
        grid_population_both = [f for f in files if "grid_population" in open(f + "/.hydra/config.yaml").read()]

        #split grid_population into grid_population and grid_population_ground_truth
        grid_pop = []
        grid_pop_ground_truth = []
        for f in grid_population_both:
            config = yaml.safe_load(open(f + "/.hydra/config.yaml", "r"))
            if "resample_centroids" in config["algo"]:
                if config["algo"]["resample_centroids"] == True:
                    grid_pop.append(f)
                else:
                    grid_pop_ground_truth.append(f)
            else:
                grid_pop_ground_truth.append(f)

        print(f"grid_pop: {len(grid_pop)}")
        print(f"grid_pop_ground_truth: {len(grid_pop_ground_truth)}")

        threshold_population = [f for f in files if "threshold_population" in open(f + "/.hydra/config.yaml").read()]

        with open(f"{adaptive_population[-1]}/.hydra/config.yaml", "r") as f:
            config = yaml.safe_load(f)

        max_size = config["population"]["max_size"]

        adaptive_centroids_populations = [f for f in files if "adaptive_centroids_population" in open(f + "/.hydra/config.yaml").read()]
        adaptive_centroids_doublesize = [f for f in adaptive_centroids_populations if str(max_size) in open(f + "/.hydra/config.yaml").read()]
        adaptive_centroids = [f for f in adaptive_centroids_populations if str(max_size // 2) in open(f + "/.hydra/config.yaml").read()]

        adaptive_population = [f for f in adaptive_population if os.path.exists(f + "metrics.pickle")]
        grid_population = [f for f in grid_pop if os.path.exists(f + "metrics.pickle")]
        grid_population_ground_truth = [f for f in grid_pop_ground_truth if os.path.exists(f + "metrics.pickle")]

        print(f"grid_population_ground_truth: {len(grid_population_ground_truth)}")
        print(f"grid_population: {len(grid_population)}")
        threshold_population = [f for f in threshold_population if os.path.exists(f + "metrics.pickle")]
        adaptive_centroids = [f for f in adaptive_centroids if os.path.exists(f + "metrics.pickle")]

        seeds = 5  # Assuming seeds is defined somewhere in the script
        adaptive_centroids_doublesize = adaptive_centroids_doublesize[-seeds:]
        adaptive_centroids = adaptive_centroids[-seeds:]
        adaptive_population = adaptive_population[-seeds:]
        grid_population = grid_population[-seeds:]
        grid_population_ground_truth = grid_population_ground_truth[-seeds:]
        threshold_population = threshold_population[-seeds:]

        files = (adaptive_population, adaptive_centroids, threshold_population, grid_population, grid_population_ground_truth)
        
        if cur_env_name == "maze_min_energy_aurora":
            df = load_data(files, METRICS_TO_GRAPH, POPULATION_NAMES, load_type="csv")
            df["num_evaluations"] = df["Iteration"] * 256 * 10
        else:
            df = load_data(files, METRICS_TO_GRAPH, POPULATION_NAMES)
            df["num_evaluations"] = df["Iteration"] * 256

        df_dict[cur_env_name] = df

    #save the df_dict to a pickle file
    with open(f"df_dict.pickle", "wb") as f:
        pickle.dump(df_dict, f)

    # Plot all environments in the same figure
    plot_all_environments(ENV_LIST, METRICS_TO_GRAPH, df_dict)
    print(f"Plotted all environments")

