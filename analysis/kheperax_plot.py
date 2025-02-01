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
MAZE_TYPES = [2, 5, 10, 20, 30, 40, 50, 100, 1000]

ENV_NAME = "maze_min_energy"
ENV_LIST = [f"{ENV_NAME}_aurora"] + [f"{ENV_NAME}_{n}" for n in MAZE_TYPES]

METRICS_TO_GRAPH = ["proj_coverage", "proj_qd_score"]
POPULATION_NAMES = ["DNS (ours)", "Cluster-Elites", "Threshold", "MAP-Elites"]

#only dashed for MAP-Elites upper bound
dash_types = {"DNS (ours)": (None, None), "Cluster-Elites": (None, None), "Threshold": (None, None), "MAP-Elites": (None, None), "MAP-Elites (Upper Bound)": (5, 5)}

XLABEL = "Evaluations"
YMAX=1.05

def pretty_env_name(env_name):
    if env_name.startswith("maze_min_energy"):
        if env_name.split('_')[-1] == "aurora":
            return f"A"
        return f"{env_name.split('_')[-1]}"
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

    values = np.zeros((len(metrics_to_graph), seeds, len(population_names)))

   
    #deconstruct files
    indices = list(range(len(files))) #graphing order, should match names

    for index, population in zip(indices, files):
        for i, filename in enumerate(population):
            if load_type == "pickle":
                with open(filename + "metrics.pickle", "rb") as f:
                    metrics = pickle.load(f)
                    for j, metric in enumerate(metrics_to_graph):
                        print(f"metrics {metrics[metric].iloc[-1]}")
                        values[j, i, index] = metrics[metric].iloc[-1]

            elif load_type == "csv":
                with open(filename + "log.csv", "r") as f:
                    metrics = pd.read_csv(f)
                    for j, metric in enumerate(metrics_to_graph):
                        print(f"metrics {metrics[metric].iloc[-1]}")
                        values[j, i, index] = metrics[metric].iloc[-1]

    data = []
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            for k in range(values.shape[2]):
                    data.append([metrics_to_graph[i], j, k, values[i, j, k]])

    df = pd.DataFrame(data, columns=["Metric", "Seed", "Population", "Value"])
    print(f"df: {df}")
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

def plot_aurora(df_dict, metrics_to_graph):
    print(f"plotting aurora")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))  # One column for each metric

    for col, metric in enumerate(metrics_to_graph):
        ax = axes[col]
        metric_df = df_dict["maze_min_energy_aurora"]
        metric_df = metric_df[metric_df["Metric"] == metric]

        sns.boxplot(
            data=metric_df,
            x="Population",
            y="Value",
            ax=ax,
            palette="tab10",  # Use a consistent color palette
            linewidth=1.5,
            fliersize=3,
            boxprops=dict(edgecolor="k"),
            medianprops=dict(color="k"),
            whiskerprops=dict(color="k"),
            capprops=dict(color="k")
        )

        # Set titles for each subplot
        if metric == "proj_qd_score":
            ax.set_title("Proj. QD Score")
        elif metric == "proj_coverage":
            ax.set_title("Proj. Coverage")
            ax.set_ylim(0, None)  # Make y-axis adaptable
            ax.yaxis.set_major_formatter(PercentFormatter(1))

        ax.set_xlabel("Method")
        ax.set_ylabel("Value")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        customize_axis(ax)

    fig.tight_layout()
    plt.savefig(f"figs/kheperax_aurora_metrics_plot.png", bbox_inches="tight")
    plt.close(fig)
    print(f"done plotting aurora")

def plot_other_environments(env_list, metrics_to_graph, df_dict):
    print(f"plotting other environments")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))  # Adjusted for two metrics

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))

    for col, metric in enumerate(metrics_to_graph):
        ax = axes[col]
        metric_df_list = []

        for env_name in env_list:
            if env_name == "maze_min_energy_aurora":
                continue
            df = df_dict[env_name]
            metric_df = df[df["Metric"] == metric].copy()
            metric_df["Environment"] = pretty_env_name(env_name)
            metric_df_list.append(metric_df)

        combined_metric_df = pd.concat(metric_df_list)
        sns.lineplot(
            data=combined_metric_df,
            x="Environment",
            y="Value",
            hue="Population",
            style="Population",
            dashes=dash_types,
            legend=False,
            estimator=np.median,
            errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
            ax=ax,
            palette="tab10"  # Use a consistent color palette
        )

        if metric == "proj_qd_score":
            ax.set_title("Proj. QD Score")
        elif metric == "proj_coverage":
            ax.set_title("Proj. Coverage")
            ax.set_ylim(0, None)  # Make y-axis adaptable
            ax.yaxis.set_major_formatter(PercentFormatter(1))

        ax.set_xlabel("Number of Descriptors")
        customize_axis(ax)

    fig.legend(axes[0].get_lines(), POPULATION_NAMES, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncols=len(POPULATION_NAMES), frameon=False)
    fig.align_ylabels(axes)
    fig.tight_layout()
    plt.savefig(f"figs/kheperax_other_environments_metrics_plot.png", bbox_inches="tight")
    plt.close(fig)
    print(f"done plotting other environments")

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
                config = yaml.safe_load(open(f + "/.hydra/config.yaml", "r"))
                if cur_env_name == config["env"]["name"]:
                    files_to_use.append(f)

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

        threshold_population = [
            f for f in files 
            if "threshold_population" in open(f + "/.hydra/config.yaml").read() 
            and ("l_value" not in yaml.safe_load(open(f + "/.hydra/config.yaml"))["population"] or yaml.safe_load(open(f + "/.hydra/config.yaml"))["population"]["l_value"] == 0.5)
        ]

        with open(f"{adaptive_population[-1]}/.hydra/config.yaml", "r") as f:
            config = yaml.safe_load(f)

        max_size = config["population"]["max_size"]

        adaptive_centroids_populations = [f for f in files if "adaptive_centroids_population" in open(f + "/.hydra/config.yaml").read()]
        adaptive_centroids_doublesize = [f for f in adaptive_centroids_populations if str(max_size) in open(f + "/.hydra/config.yaml").read()]
        adaptive_centroids = [f for f in adaptive_centroids_populations if str(max_size // 2) in open(f + "/.hydra/config.yaml").read()]

        adaptive_population = [f for f in adaptive_population if os.path.exists(f + "metrics.pickle")]
        grid_population = [f for f in grid_pop if os.path.exists(f + "metrics.pickle")]
        #grid_population_ground_truth = [f for f in grid_pop_ground_truth if os.path.exists(f + "metrics.pickle")]

        #print(f"grid_population_ground_truth: {len(grid_population_ground_truth)}")
        #print(f"grid_population: {len(grid_population)}")
        threshold_population = [f for f in threshold_population if os.path.exists(f + "metrics.pickle")]
        adaptive_centroids = [f for f in adaptive_centroids if os.path.exists(f + "metrics.pickle")]

        seeds = 5 
        adaptive_centroids_doublesize = adaptive_centroids_doublesize[-seeds:]
        adaptive_centroids = adaptive_centroids[-seeds:]
        adaptive_population = adaptive_population[-seeds:]
        grid_population = grid_population[-seeds:]
        #grid_population_ground_truth = grid_population_ground_truth[-seeds:]
        threshold_population = threshold_population[-seeds:]

        files = (adaptive_population, adaptive_centroids, threshold_population, grid_population) #grid_population_ground_truth)
        
        if cur_env_name == "maze_min_energy_aurora":
            print(f"cur_env: {cur_env_name}")
            df = load_data(files, METRICS_TO_GRAPH, POPULATION_NAMES, load_type="csv")
        else:
            print(f"cur_env: {cur_env_name}")
            df = load_data(files, METRICS_TO_GRAPH, POPULATION_NAMES, load_type="csv")
        
        #add the descriptor number to the df
        df["type"] = cur_env_name.split("_")[-1]

        df_dict[cur_env_name] = df

    #save the df_dict to a pickle file
    with open(f"kheperax_df_dict.pickle", "wb") as f:
        pickle.dump(df_dict, f)

    # Plot Aurora separately
    plot_aurora(df_dict, METRICS_TO_GRAPH)

    # Plot other environments
    plot_other_environments(ENV_LIST, METRICS_TO_GRAPH, df_dict)

    print(f"Plotted all environments")

