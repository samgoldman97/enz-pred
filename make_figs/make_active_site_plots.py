""" Make active site pooling plots """
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from helpers import *

REGR_METRICS = ["spearman", "rmse", "mae"]
CLASS_METRICS = ["auc-roc", "avg-pr", "mcc"]

AX_LABELS = {
    "spearman": "Spearman",
    "avg-pr": "AUPRC",
    "mcc": "MCC",
    "auc-roc": "AUC-ROC",
    "rmse": "RMSE",
    "mae": "MAE",
}

METHODS_RENAMES = {
    "hard": "Active Site",
    "msacover": "Coverage",
    "msaconserv": "Conservation",
    "randmsa": "MSA rand.",
    "mean": "Mean",
    "knn": "Levenshtein distance",
}

LINEPLOT_METHODS = ["hard", "msacover", "msaconserv"]
BASELINES = ["mean"]
MODEL_BASELINES = ["knn"]

DATASETS_TO_PLOT = [
    "esterase_binary",
    "phosphatase_chiral_binary",
    "davis",
    "gt_acceptors_chiral_binary",
    "duf_binary",
    "halogenase_NaBr_binary",
    "davis_filtered",
]


def make_lineplots(df, outdir):
    """Make all the lineplots"""

    num_colors = len(LINEPLOT_METHODS) + len(BASELINES) + len(MODEL_BASELINES)
    pal = sns.color_palette("Set1", n_colors=num_colors)
    sns.set_palette(pal)

    for dataset_name in DATASETS_TO_PLOT:
        dataset_df = df[df["dataset_name"] == dataset_name]
        if len(dataset_df) == 0:
            continue

        metrics = REGR_METRICS if dataset_df["regression"].values[0] else CLASS_METRICS
        for metric_name in metrics:
            fig = plt.figure(figsize=(7 * 3 / 4, 5 * 3 / 4))
            for method in LINEPLOT_METHODS:
                method_df = dataset_df[dataset_df["pool_strat"] == method]

                # Average by the pool_num
                task_avged = (
                    method_df.groupby(["pool_num", "seed"]).mean().reset_index()
                )

                # Sort by pool num
                task_avged = task_avged.sort_values(by="pool_num")

                # Avg again
                scatter_avg = method_df.groupby(["pool_num"]).mean().reset_index()
                x, y = scatter_avg["pool_num"].values, scatter_avg[metric_name]
                label = METHODS_RENAMES[method]
                if method == "randmsa":
                    extra_args = {"alpha": 0.4}
                else:
                    extra_args = {}

                sns.lineplot(
                    data=task_avged,
                    x="pool_num",
                    y=metric_name,
                    label=label,
                    **extra_args,
                )

                plt.scatter(x, y, color="black", zorder=10, **extra_args)

            for method in BASELINES:
                method_df = dataset_df[dataset_df["pool_strat"] == method]
                y_avg = np.mean(method_df[metric_name])
                label = METHODS_RENAMES[method]

                plt.axhline(y=y_avg, linestyle=f"dotted", label=label, color="black")

            for method in MODEL_BASELINES:
                method_df = dataset_df[dataset_df["model"] == method]
                y_avg = np.mean(method_df[metric_name])
                label = METHODS_RENAMES[method]

                plt.axhline(y=y_avg, linestyle=f"dashed", label=label, color="black")

            ylabel = AX_LABELS[metric_name]
            plt.legend(
                bbox_to_anchor=(0.5, -0.8),
                ncol=num_colors,
                loc="center",
            )
            plt.xlabel("Num. residues pooled")
            plt.xlim([0, 200])

            plt.ylabel(ylabel)
            plt.title(DATASET_NAME_MAP.get(dataset_name))
            out_loc = os.path.join(
                outdir, f"PSAR_angstrom_plot_{dataset_name}_{ylabel}.pdf"
            )

            plt.savefig(out_loc, bbox_inches="tight")
            plt.close()


if __name__ == "__main__":
    results_file = "results/dense/2021_05_25_pqsar_shell/pqsar_angstrom_dist_5_25.csv"
    outdir = "results/figure_export/"
    os.makedirs(outdir, exist_ok=True)

    df = extract_csv_file(
        results_file, extract_pqsar_pooling, extract_dataset_name, rename=True
    )
    df = df.query("dataset_split == 'test'").reset_index()
    make_lineplots(df, outdir)
