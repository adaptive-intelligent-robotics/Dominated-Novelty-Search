import pandas as pd
import pickle

from scipy.stats import wilcoxon, ranksums, mannwhitneyu
from statsmodels.stats.multitest import multipletests


ENV_LIST = [
    "ant_omni",
    "antblockmany_omni",
    "walker2d_uni",
]

POPULATION_LIST = [
    "DNS (ours)",
    "Cluster-Elites",
    "Threshold",
    "MAP-Elites",
    "MAP-Elites (Upper Bound)",
]

METRICS_LIST = [
    "proj_coverage",
    "proj_max_fitness",
    "proj_qd_score",
]

P_VALUE_LIST = [
    ["proj_coverage", "ant_omni", "DNS (ours)", "Threshold"],
    ["proj_coverage", "ant_omni", "DNS (ours)", "Cluster-Elites"],
    ["proj_coverage", "ant_omni", "DNS (ours)", "MAP-Elites"],
    ["proj_coverage", "ant_omni", "DNS (ours)", "MAP-Elites (Upper Bound)"],

    ["proj_coverage", "antblockmany_omni", "DNS (ours)", "Threshold"],
    ["proj_coverage", "antblockmany_omni", "DNS (ours)", "Cluster-Elites"],
    ["proj_coverage", "antblockmany_omni", "DNS (ours)", "MAP-Elites"],
    ["proj_coverage", "antblockmany_omni", "DNS (ours)", "MAP-Elites (Upper Bound)"],

    ["proj_max_fitness", "ant_omni", "DNS (ours)", "Threshold"],
    ["proj_max_fitness", "ant_omni", "DNS (ours)", "Cluster-Elites"],
    ["proj_max_fitness", "ant_omni", "DNS (ours)", "MAP-Elites"],
    ["proj_max_fitness", "ant_omni", "DNS (ours)", "MAP-Elites (Upper Bound)"],

    ["proj_max_fitness", "antblockmany_omni", "DNS (ours)", "Threshold"],
    ["proj_max_fitness", "antblockmany_omni", "DNS (ours)", "Cluster-Elites"],
    ["proj_max_fitness", "antblockmany_omni", "DNS (ours)", "MAP-Elites"],
    ["proj_max_fitness", "antblockmany_omni", "DNS (ours)", "MAP-Elites (Upper Bound)"],

    ["proj_qd_score", "ant_omni", "DNS (ours)", "Threshold"],
    ["proj_qd_score", "ant_omni", "DNS (ours)", "Cluster-Elites"],
    ["proj_qd_score", "ant_omni", "DNS (ours)", "MAP-Elites"],
    ["proj_qd_score", "ant_omni", "DNS (ours)", "MAP-Elites (Upper Bound)"],

    ["proj_qd_score", "antblockmany_omni", "DNS (ours)", "Threshold"],
    ["proj_qd_score", "antblockmany_omni", "DNS (ours)", "Cluster-Elites"],
    ["proj_qd_score", "antblockmany_omni", "DNS (ours)", "MAP-Elites"],
    ["proj_qd_score", "antblockmany_omni", "DNS (ours)", "MAP-Elites (Upper Bound)"],


]

if __name__ == "__main__":
   
    #read results from pickle:
    with open("df_dict.pickle", "rb") as f:
        df_dict = pickle.load(f)

    #df_dict is a dictionary of dataframes, where each dataframe is a different environment
    #we need to concatenate all the dataframes into one dataframe, adding a new column for the environment name
    df = pd.concat([df_dict[env].assign(env=env) for env in ENV_LIST])

    # Filter
    df = df[df["Population"].isin(POPULATION_LIST)]
    df = df[df["num_evaluations"] <= 1_000_000]

    # Keep only the last iteration
    idx = df.groupby(["Metric","env", "Population", "Seed"])["Iteration"].idxmax()
    df = df.loc[idx]


    # Compute p-values
    p_value_df = pd.DataFrame(columns=["Metric", "env", "Population_1", "Population_2", "p_value"])
    for metric in METRICS_LIST:
        for env in ENV_LIST:
            for Population_1 in POPULATION_LIST:
                for Population_2 in POPULATION_LIST:
                    print(f"Computing p-value for {metric} in {env} between {Population_1} and {Population_2}")
                    print(df[(df["env"] == env) & (df["Population"] == Population_1) & (df["Metric"] == metric)].head())
                    print(df[(df["env"] == env) & (df["Population"] == Population_2) & (df["Metric"] == metric)].head())
                    stat = mannwhitneyu(
                        df[(df["env"] == env) & (df["Population"] == Population_1) & (df["Metric"] == metric)]["Value"],
                        df[(df["env"] == env) & (df["Population"] == Population_2) & (df["Metric"] == metric)]["Value"],
                    )
                    p_value_df.loc[len(p_value_df)] = {"Metric": metric, "env": env, "Population_1": Population_1, "Population_2": Population_2, "p_value": stat.pvalue}

    # Filter p-values
    p_value_df.set_index(["Metric", "env", "Population_1", "Population_2"], inplace=True)

    print(p_value_df.head(30))
    #print unique metrics

    print(P_VALUE_LIST)
    p_value_df = p_value_df.loc[P_VALUE_LIST]


    print(p_value_df.head())
    # Correct p-values
    p_value_df.reset_index(inplace=True)
    p_value_df["p_value_corrected"] = multipletests(p_value_df["p_value"], method="holm")[1]

    print(p_value_df.head())

    p_value_df = p_value_df.pivot(index=["env", "Population_1", "Population_2"], columns="Metric", values="p_value_corrected")

    print(p_value_df.head())
    # Save p-values
    p_value_df.to_csv("mujoco_p_value.csv")



