"""Helper functions to make figures """

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Define plotting parameters
sns.set(
    context="paper",
    style="white",
    font_scale=2.5,
    palette="Blues_r",
    rc={
        "figure.figsize": (20, 10),
        "legend.fontsize": 20,
        "legend.title_fontsize": 20,
        "xtick.major.size": 8,
        "ytick.major.size": 8,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "lines.linewidth": 3,
        "font.family": "sans-serif",
        "font.sans-serif": "Arial",
    },
)

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

DATASET_NAME_MAP = {
    "duf_binary": "BKACE",
    "gt_acceptors_chiral_binary": "Glyco.",
    "halogenase_NaBr_binary": "Halogenase",
    "olea_binary": "Thiolase",
    "phosphatase_chiral_binary": "Phosphatase",
    "esterase_binary": "Esterase",
    "davis_filtered": "Kinases",
}
BENCHMARK_PANEL_HEIGHT = 3.0
BENCHMARK_PANEL_WIDTH = 3.5
GAP_SIZE = 0.2
METRIC_NAME_MAPPING = {"avg-pr": "AUPRC", "auc-roc": "AUCROC"}


### Extracting data functions


def rename_metrics_errors(
    df,
    result_kws=[
        "rmse",
        "accuracy",
        "avg-pr",
        "precision",
        "recall",
        "auc-roc",
        "mae",
        "f1",
        "spearman",
        "mcc",
    ],
):
    """Rename the metrics that are task specific from each dataest to be uniform"""
    old_keys = set(df.keys())
    # Map the dataset specific key to the other key
    # e.g., new_name mapping converts auc_conversion_smiles : auc, etc.
    new_name_mapping = {
        i: j for i in old_keys for j in result_kws if i.startswith(j) and "_" in i
    }

    carry_over_cols = [i for i in old_keys if not np.any([j in i for j in result_kws])]

    # Map dataset specific key to the df that it stands for
    compound_mapping = {
        i: i.split("_")[-1]
        for i in old_keys
        for j in result_kws
        if i.startswith(j) and "_" in i
    }

    # Now let's filter through all the keys
    new_pd_df = []
    for row in df.to_dict(orient="records"):
        # Create new entries for every single molecule tested in this row
        # Copy over the default parameters that belong with this row
        new_dict = defaultdict(lambda: {j: row[j] for j in carry_over_cols})
        # Loop over each column
        for col, val in row.items():
            # If this is one of the default vals, ignore it
            if col not in compound_mapping:
                continue
            # Else get the new column anme (e.g., auc)
            # and get the molecule it belongs to
            new_col_name = new_name_mapping[col]
            new_smiles = compound_mapping[col]

            if not np.isnan(val):
                new_dict
                new_dict[new_smiles][new_col_name] = val

        for smiles, dict_ in new_dict.items():
            dict_["TASK"] = smiles
            new_pd_df.append(dict_)

    return pd.DataFrame(new_pd_df)


def extract_model_name(entry):
    """This model takes a single entry and computes a model name"""

    return {"model_name": entry["model"]}


def extract_dataset_name(entry):
    """This model takes a single entry and computes a model name"""

    dataset = entry["hts_csv_file"].split("/")[-1].split(".")[0]
    return {"dataset_name": dataset}


def extract_csv_file(
    results_file, extract_model_name, extract_dataset_name, rename=True
):
    """Extract results file and rename"""
    df = pd.read_csv(results_file, index_col=0)
    if rename:
        df = rename_metrics_errors(df)

    # df['model_name'] = [extract_model_name(i)['model_name'] for _,i in df.iterrows()]
    temp_df = pd.DataFrame([extract_model_name(i) for _, i in df.iterrows()])
    for k in temp_df.columns:
        df[k] = temp_df[k].values

    df["dataset_name"] = [
        extract_dataset_name(i)["dataset_name"] for _, i in df.iterrows()
    ]
    if "pairwise_pearson_correlation_list" in df.columns:
        df["pairwise_pearson_correlation_list"] = [
            eval(i) for i in df["pairwise_pearson_correlation_list"]
        ]
    if "pairwise_spearman_correlation_list" in df.columns:
        df["pairwise_spearman_correlation_list"] = [
            eval(i) for i in df["pairwise_spearman_correlation_list"]
        ]
    return df


### Plotting functions


def compute_results_df_clustering(results_obj, verbose=True) -> pd.DataFrame:
    """Process a clustering model and turn it into a dataframe

    For this analysis, assume that we had separate functions act on results obj to add
    "model_name" and "dataset_name". This abstracts that logic

    """
    fold_index = defaultdict(lambda: 0)
    results_df = []
    for i in results_obj:
        # Get full title info of this run
        model = i["model_name"]
        dataset = i["dataset_name"]

        # Now process results
        # Only take the split indicated
        for metric_name, metric_dict in i["cluster_results"].items():
            for partition, results in metric_dict.items():
                vals = results["pred_ar"]
                num_clusters = results["num_clusters"]

                for index, (val, num_cluster) in enumerate(zip(vals, num_clusters)):
                    new_entry = {
                        "model": model,
                        "pred": val,
                        "num_cluster": num_cluster,
                        "dataset": dataset,
                        "fold": fold_index[model],
                        "metric": metric_name,
                        "partition": partition,
                    }
                    results_df.append(new_entry)
                fold_index[model] += 1

                vals = results["target_ar"]
                for index, (val, num_cluster) in enumerate(zip(vals, num_clusters)):
                    new_entry = {
                        "model": "idealized",
                        "pred": val,
                        "num_cluster": num_cluster,
                        "dataset": dataset,
                        "fold": fold_index["idealized"],
                        "metric": metric_name,
                        "partition": partition,
                    }
                    results_df.append(new_entry)
                fold_index["idealized"] += 1
    if verbose:
        for k, v in fold_index.items():
            print(f"Saw model {k} {v} times")
    results_df = pd.DataFrame(results_df)
    return results_df


## For loading in the CSV outputs


def get_colors(x: list):
    """Get colors for x, a list of lists, where each sublist gets its own hue"""
    num_cats = len(x)

    color_list = sns.color_palette("Paired", n_colors=num_cats * 2)
    colors = []
    for index, i in enumerate(x):

        num_new_cols = len(i)

        start_col = color_list[index * 2]
        end_col = color_list[index * 2 + 1]
        start, end = np.array(start_col), np.array(end_col)
        new_cols = np.linspace(start, end, num_new_cols).tolist()
        colors.extend(new_cols)
    return colors


def calc_p_val(m1, m2, paired=True):
    """Compute a p value"""
    ### :Checking that mean 1 is greater than mean 2
    mean_1 = np.mean(m1)
    mean_2 = np.mean(m2)
    N_1 = len(m1)
    N_2 = len(m2)
    std_1 = np.std(m1, ddof=1)
    std_2 = np.std(m2, ddof=1)
    ddof_num = ((std_1 ** 2 / N_1) + (std_2 ** 2 / N_2)) ** 2
    ddof_denom = std_1 ** 4 / (N_1 ** 2 * (N_1 - 1)) + std_2 ** 4 / (
        N_2 ** 2 * (N_2 - 1)
    )
    ddof = ddof_num / ddof_denom

    t = (mean_1 - mean_2) / np.sqrt(std_1 ** 2 / N_1 + std_2 ** 2 / N_2)
    return stats.t.sf(t, ddof)  # stats.t.cdf(t, ddof)


def compute_pvals(
    df, metric, central_method, alternative="greater", stat_test="wilcox"
):
    """Compute P val pairs"""
    p_vals, pairs = [], []
    for sub_df_name, sub_df in df.groupby("dataset_name"):
        central_data = sub_df[sub_df["model_name"] == central_method]
        methods_temp, pvals = [], []
        methods = pd.unique(sub_df["model_name"])
        if central_method not in methods:
            continue
        for method in methods:
            if method == central_method:
                continue

            method_data = sub_df[sub_df["model_name"] == method]
            sorted_central = central_data.sort_values("TASK", axis=0)[metric].values
            sorted_method = method_data.sort_values("TASK", axis=0)[metric].values

            sorted_central = central_data.groupby("seed").mean()[metric].values
            sorted_method = method_data.groupby("seed").mean()[metric].values

            ### Now do statistical comparison
            t_pval = calc_p_val(sorted_central, sorted_method)
            t_welsh = stats.ttest_ind(
                sorted_central, sorted_method, equal_var=False
            ).pvalue

            w_pval = stats.wilcoxon(
                sorted_central, sorted_method, alternative=alternative
            ).pvalue
            if stat_test == "wilcox":
                pvals.append(w_pval)
            elif stat_test == "t":
                pvals.append(t_pval)
            elif stat_test == "t_welsh":
                pvals.append(t_welsh)
            else:
                raise ValueError()

            methods_temp.append(method)

        new_p = multipletests(pvals, method="fdr_bh", alpha=0.05)[1]
        new_entries = [
            ((sub_df_name, central_method), (sub_df_name, method))
            for method in methods_temp
        ]
        p_vals.extend(new_p)
        pairs.extend(new_entries)
    return p_vals, pairs


def convert_p_val_to_stars(dataset_order, compare_method, method_order, pairs, p_vals):
    """Convert p values to asterisks to annotate bar plots for stat. sig."""
    res_dict = {}
    # Convert pairs into dict
    for ((d1, m1), (d2, m2)), p_val in zip(pairs, p_vals):
        res_dict[(d2, m2)] = p_val

    annot_strings = []
    for dataset in dataset_order:
        for compare_method in method_order:
            p_val = res_dict.get((dataset, compare_method), 1)
            # Add asterisks for each cutoff it passes
            annot_string = "".join(
                ["*" for x in [1e-4, 1e-3, 1e-2, 5e-2] if p_val <= x]
            )

            annot_strings.append(annot_string)
    return annot_strings


def extract_pqsar_sweep_names(entry):
    """Extract sweep names for the pQSAR task"""
    model = entry["model"]

    # Add model params
    if model == "linear":
        scale_prot = entry["scale_prot"]
        scaled_str = f" (Scaled)" if scale_prot else ""
        model = f"{model}"  # {scaled_str}"
    elif model == "knn":
        seq_dist_type = entry["seq_dist_type"]
        align_dist_type = entry["align_dist"]
        align_dist_type = f", {align_dist_type}" if align_dist_type else ""
        scale_prot = entry["scale_prot"]
        scaled_str = f" (Scaled)" if scale_prot else ""
        model = rf"{seq_dist_type} {model}{scaled_str}"
    elif "ffn" in model:
        if not pd.isna(entry["chem_featurizer"]):
            if entry["chem_featurizer"] == "random":
                model = rf"{model}(prot,randchem)"
            elif entry["chem_featurizer"] == "cat":
                model = rf"{model}(prot,catchem)"
            else:
                model = rf"{model}(prot,chem)"

    prot_featurizer = entry["prot_featurizer"]
    if prot_featurizer in ["esm", "bepler", "msa"]:

        # Featurizer
        ref_file = entry["ssa_ref_file"]
        pool_strat = entry["pool_prot_strategy"]

        if isinstance(ref_file, str) and pool_strat != "mean":
            ssa_ref = re.search("reference_([0-9]*).txt$", ref_file)
            ang_num = int(ssa_ref.groups()[0])
            ang_str = f"{ang_num:02} å"
        else:
            ang_str = ""

        if isinstance(pool_strat, str):
            prot_featurizer = f"{prot_featurizer}-{pool_strat}"

        prot_featurizer = f"{prot_featurizer}"

    elif not isinstance(prot_featurizer, str):
        prot_featurizer = ""

    model = f"{prot_featurizer} {model} "
    return {"model_name": model.strip()}


def extract_qsar_sweep_names(entry):
    """Extract sweep names for the pQSAR task"""
    model = entry["model"]

    # Add model params
    if model == "linear":
        model = rf"{model}"
    elif model == "knn":
        sub_dist_type = entry["sub_dist_type"]
        align_dist_type = entry["align_dist"]
        align_dist_type = f", {align_dist_type}" if align_dist_type else ""
        model = rf"{model}-{sub_dist_type}"
    elif "ffn" in model:
        if not pd.isna(entry["prot_featurizer"]):
            if entry["prot_featurizer"] == "random":
                model = rf"{model}(prot,randchem)"
            elif entry["prot_featurizer"] == "cat":
                model = rf"{model}(prot,catchem)"
            else:
                model = rf"{model}(prot,chem)"

    substrate_featurizer = entry["chem_featurizer"]

    model = f"{substrate_featurizer} {model}"

    return {"model_name": model}


def extract_pqsar_pooling(entry):
    """Extract sweep names for the pQSAR task pooling experiment"""

    model = entry["model"]
    prot_featurizer = entry["prot_featurizer"]
    ref_file = entry["ssa_ref_file"]
    pool_strat = entry["pool_prot_strategy"]
    pool_num = entry["pool_num"]

    if isinstance(ref_file, str) and pool_strat != "mean":
        ssa_ref = re.search("reference_([0-9]*).txt$", ref_file)
        ang_num = int(ssa_ref.groups()[0])
        ang_str = f"{ang_num:02} å"
    else:
        ang_num = None
        ang_str = ""

    return {
        "model_name": model,
        "embedding": prot_featurizer,
        "pool_strat": pool_strat,
        "pool_num": pool_num,
        "angstroms": ang_num,
    }
