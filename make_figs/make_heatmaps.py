"""Try to make heatmaps from results data"""

import os
import json
from collections import defaultdict
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

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
        "axes.spines.top": True,  # False,
        "axes.spines.right": True,  # False,
        "lines.linewidth": 3,
        "font.family": "sans-serif",
        "font.sans-serif": "Arial",
    },
)

# Matplotlib export
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

SEED = 1

DATASET_RENAME = {
    "Gt": "Glyco.",
    "Duf": "BKACE",
    "Esterase": "Esterase",
    "Phosphatase": "Phosphatase",
    "Halogenase": "Halogenase",
}


def threshold(df_temp, threshold_val=0.5, use_threshold=True):
    """Threshold a data frame"""
    df_copy = df_temp.copy()
    if use_threshold:
        bools_1 = df_temp > threshold_val
        bools_0 = ~bools_1
        df_copy[bools_1] = 1
        df_copy[bools_0] = 0
    return df_copy


def plot_dataframe(df_list, name_list, save_dir, save_name, dataset_title, qsar=False):
    """Plot dataframe"""

    # Set same axes
    col_list = [set(i.columns) for i in df_list]
    row_list = [set(i.index) for i in df_list]
    intersecting_cols = list(set.intersection(*col_list))
    intersecting_rows = list(set.intersection(*row_list))

    # Modify all outputs to hvae the same rows and cols
    df_list = [j.loc[intersecting_rows, intersecting_cols] for j in df_list]

    fig, axes = plt.subplots(nrows=1, ncols=len(df_list))
    for df, name, ax in zip(df_list, name_list, axes):
        ax.imshow(df, cmap="hot")
        if qsar:
            ax.set_ylabel("Substrates")
            ax.set_xlabel("Enzymes")
        else:
            ax.set_xlabel("Substrates")
            ax.set_ylabel("Enzymes")

        ax.set_title(name)

        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(dataset_title)

    plt.savefig(os.path.join(save_dir, save_name), bbox_inches="tight")


def get_bool_index_ar(df, model_select):
    """get appropriate bool index"""
    if model_select == "PSAR Single":
        bools_to_fold = []
        bools_to_fold.append(df["prot_featurizer"] == "esm")
        bools_to_fold.append(pd.isnull(df["chem_featurizer"]))
        bools_to_fold.append(df["model"] == "linear")
        bool_index = np.all(np.array(bools_to_fold), axis=0)
    elif model_select == "QSAR Single":
        bools_to_fold = []
        bools_to_fold.append(df["chem_featurizer"] == "morgan1024")
        bools_to_fold.append(pd.isnull(df["prot_featurizer"]))
        bools_to_fold.append(df["model"] == "linear")
        bool_index = np.all(np.array(bools_to_fold), axis=0)
    elif model_select == "CPI":
        bools_to_fold = []
        bools_to_fold.append(df["prot_featurizer"] == "esm")
        bools_to_fold.append(df["chem_featurizer"] == "morgan1024")
        bools_to_fold.append(df["model"] == "ffn")
        bool_index = np.all(np.array(bools_to_fold), axis=0)
    else:
        raise ValueError()
    return bool_index


def make_QSAR_plots(results_file, outdir, use_threshold=False):
    df = pd.read_csv(results_file, index_col=0, low_memory=False)

    # Filter down to test set and only seed == 1
    df = df.query('dataset_split == "test"').reset_index(drop=True)
    df = df.query(f"seed == {SEED}").reset_index(drop=True)

    models = ["QSAR Single", "CPI"]
    model_dict = {}
    for model in models:
        # Now only get the columns we want to work with
        # select only CPI
        bool_index = get_bool_index_ar(df, model)
        hts_to_out_dir = {}
        hts_to_out_dir = dict((df[bool_index][["hts_csv_file", "out"]].values))

        dataset_dict = {}
        # Now work on dataset
        for hts, out in hts_to_out_dir.items():
            dataset_name = hts.split("/")[-1].split("_")[0].split(".")[0]
            results_dir = out.rsplit("/", 1)[0]
            dir_contents = [
                i for i in os.listdir(results_dir) if i.endswith("preds.json")
            ]

            if len(dir_contents) != 1:
                raise RuntimeError()

            pred_input = dir_contents[0]

            # Now extract predicted inputs
            predictions = json.load(open(os.path.join(results_dir, pred_input), "rb"))

            df_json = pd.DataFrame(predictions).query("split == 'test'")

            df_pred = df_json.pivot_table(
                values="pred", index="SUBSTRATES", columns="SEQ"
            )
            df_val = df_json.pivot_table(
                values="val", index="SUBSTRATES", columns="SEQ"
            )
            dataset_dict[dataset_name] = [df_pred, df_val]
        model_dict[model] = dataset_dict

    # Convert to dataset to model dict
    new_dict = defaultdict(lambda: dict())
    for model, dataset_dict in model_dict.items():
        for dataset, df_list in dataset_dict.items():
            new_dict[dataset][model] = df_list

    for dataset, model_dict in new_dict.items():
        df_list = [
            model_dict["QSAR Single"][1],
            threshold(
                model_dict["QSAR Single"][0],
                threshold_val=0.5,
                use_threshold=use_threshold,
            ),
            threshold(
                model_dict["CPI"][0], threshold_val=0.5, use_threshold=use_threshold
            ),
        ]
        name_list = ["Actual", "QSAR Single", "CPI"]

        plot_dataframe(
            df_list,
            name_list,
            save_dir=outdir,
            save_name=f"appendix_{dataset}_qsar_threshold_{use_threshold}.pdf",
            dataset_title=DATASET_RENAME.get(dataset.title()),
            qsar=True,
        )


def make_PSAR_plots(results_file, outdir, use_threshold=False):
    df = pd.read_csv(results_file, index_col=0, low_memory=False)

    # Filter down to test set and only seed == 1
    df = df.query('dataset_split == "test"').reset_index(drop=True)
    df = df.query(f"seed == {SEED}").reset_index(drop=True)

    models = ["PSAR Single", "CPI"]
    model_dict = {}
    for model in models:
        # Now only get the columns we want to work with
        # select only CPI
        bool_index = get_bool_index_ar(df, model)
        hts_to_out_dir = {}
        hts_to_out_dir = dict((df[bool_index][["hts_csv_file", "out"]].values))

        dataset_dict = {}
        # Now work on dataset
        for hts, out in hts_to_out_dir.items():
            dataset_name = hts.split("/")[-1].split("_")[0].split(".")[0]
            results_dir = out.rsplit("/", 1)[0]
            dir_contents = [
                i for i in os.listdir(results_dir) if i.endswith("preds.json")
            ]

            if len(dir_contents) != 1:
                raise RuntimeError()

            pred_input = dir_contents[0]

            # Now extract predicted inputs
            predictions = json.load(open(os.path.join(results_dir, pred_input), "rb"))

            df_json = pd.DataFrame(predictions).query("split == 'test'")

            df_pred = df_json.pivot_table(
                values="pred", index="SEQ", columns="SUBSTRATES"
            )
            df_val = df_json.pivot_table(
                values="val", index="SEQ", columns="SUBSTRATES"
            )
            dataset_dict[dataset_name] = [df_pred, df_val]
        model_dict[model] = dataset_dict

    # Convert to dataset to model dict
    new_dict = defaultdict(lambda: dict())
    for model, dataset_dict in model_dict.items():
        for dataset, df_list in dataset_dict.items():
            new_dict[dataset][model] = df_list

    for dataset, model_dict in new_dict.items():
        df_list = [
            model_dict["PSAR Single"][1],
            threshold(
                model_dict["PSAR Single"][0],
                threshold_val=0.5,
                use_threshold=use_threshold,
            ),
            threshold(
                model_dict["CPI"][0], threshold_val=0.5, use_threshold=use_threshold
            ),
        ]
        name_list = ["Actual", "PSAR Single", "CPI"]

        plot_dataframe(
            df_list,
            name_list,
            save_dir=outdir,
            save_name=f"appendix_{dataset}_psar_threshold_{use_threshold}.pdf",
            dataset_title=DATASET_RENAME.get(dataset.title()),
        )


def main():
    """main."""

    INPUT_FILE_PSAR = (
        "results/dense/2021_05_27_psar_with_multi/consolidated_psar_multi.csv"
    )
    RESULTS_DIR_PSAR = "results/figure_export/psar_heatmaps"

    INPUT_FILE_QSAR = "results/dense/2021_05_28_qsar_with_multi/qsar_combined.csv"
    RESULTS_DIR_QSAR = "results/figure_export/qsar_heatmaps"

    os.makedirs(RESULTS_DIR_PSAR, exist_ok=True)
    os.makedirs(RESULTS_DIR_QSAR, exist_ok=True)

    make_PSAR_plots(INPUT_FILE_PSAR, RESULTS_DIR_PSAR, use_threshold=False)
    make_QSAR_plots(INPUT_FILE_QSAR, RESULTS_DIR_QSAR, use_threshold=False)

    make_PSAR_plots(INPUT_FILE_PSAR, RESULTS_DIR_PSAR, use_threshold=True)
    make_QSAR_plots(INPUT_FILE_QSAR, RESULTS_DIR_QSAR, use_threshold=True)


if __name__ == "__main__":
    main()
