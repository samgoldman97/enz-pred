"""Make all qsar benchmarking plots"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from helpers import *

method_name_map = {
    "morgan1024 ffn(prot,chem)": "FFN: [pretrained, Morgan]",
    "morgan1024 ffndot(prot,chem)": "FFN: pretrained • Morgan",
    "morgan1024 ffn": "FFN: Morgan",
    "morgan1024 ffnsingle": "FFN: Morgan",
    "jtvae linear": "Ridge: pretrained",
    "morgan1024 linear": "Ridge: Morgan",
    "maccs linear": "Ridge: MACCS",
    "morgan1024 knn-tanimoto": "KNN: Tanimoto",
    "random linear": "Ridge: random feats.",
    "morgan1024 ffn(prot,randchem)": "FFN: [rand., Morgan]",
    "morgan1024 ffn(prot,catchem)": "FFN: [one-hot, Morgan]",
    # Not shown
    "jtvae knn-l1-dist": "KNN: JT-VAE",
    "maccs knn-tanimoto": "KNN: Tanimoto",
}

metric = "avg-pr"
metric_name = "AUPRC"
metric = "aucroc"
metric_name = "AUCROC"
panel_height = 3.0
panel_width = 3.5
baseline_models = ["random linear", "morgan1024 knn-tanimoto"]
singletask_models = ["jtvae linear", "morgan1024 linear"]
multitask_models = ["morgan1024 ffn", "morgan1024 ffn(prot,catchem)"]
cpi_models = ["morgan1024 ffn(prot,chem)", "morgan1024 ffndot(prot,chem)"]

dataset_names = [
    "gt_acceptors_chiral_binary",
    "esterase_binary",
    "phosphatase_chiral_binary",
]

compare_method = "morgan1024 linear"

x_label_names = ["Baselines", "Multi-task", "CPI", "Single-task"]

x_hatch_map = {
    x_label_names[0]: None,
    x_label_names[1]: None,
    x_label_names[2]: None,
    x_label_names[3]: None,
}

joint_list = [baseline_models, multitask_models, cpi_models, singletask_models]
x_colors = get_colors(joint_list)
full_list = [x for i in joint_list for x in i]
gap_size = 0.2
shift_factors = np.array(
    [index * gap_size for index, i in enumerate(joint_list) for j in i]
)

plot_positions = np.arange(len(full_list)) + shift_factors
x_labels = (
    [x_label_names[0]] * len(baseline_models)
    + [x_label_names[1]] * len(multitask_models)
    + [x_label_names[2]] * len(cpi_models)
    + [x_label_names[3]] * len(singletask_models)
)

x_label_map = dict(zip([method_name_map[i] for i in full_list], x_labels))
x_colors = get_colors(joint_list)


def make_qsar_results_tables(df, outdir):
    """Make qsar results table"""
    new_table = []
    entries = []
    for index, (dataset_name) in enumerate(dataset_names):
        plot_df = df.query(f"dataset_name == '{dataset_name}'").reset_index(drop=True)
        plot_df = plot_df[[j in full_list for j in plot_df["model_name"]]]
        dataset_rename = DATASET_NAME_MAP[dataset_name]
        for metric, metric_rename in METRIC_NAME_MAPPING.items():
            cur_max = None
            method_entries = []
            for method, method_type in zip(full_list, x_labels):
                temp_df = plot_df[plot_df["model_name"] == method]
                temp_df = temp_df.groupby("seed").mean().reset_index()
                vals = temp_df[metric]
                bar_height = np.mean(vals)
                error_height = 1.96 * stats.sem(vals)
                new_entry_numeric = fr"${bar_height:0.3f}\pm {error_height:0.3f}$"
                method_rename = method_name_map[method]

                new_entry = {
                    "Metric": metric_rename,
                    "Numeric_val": bar_height,
                    "Numeric_sem": error_height,
                    "Value": new_entry_numeric,
                    "Dataset": dataset_rename,
                    "Method Type": method_type,
                    "Method": method_rename,
                }

                if cur_max is None or new_entry["Numeric_val"] > cur_max["Numeric_val"]:
                    cur_max = new_entry

                method_entries.append(new_entry)
            lower_bound = cur_max["Numeric_val"] - cur_max["Numeric_sem"]
            for entry_ in method_entries:
                if entry_["Numeric_val"] >= lower_bound:
                    bar_height = entry_["Numeric_val"]
                    bar_sem = entry_["Numeric_sem"]
                    numeric_piece = rf"{bar_height:0.3f}\pm {bar_sem:0.3f}"
                    entry_["Value"] = r"$\mathbf{" + numeric_piece + r"}$"

            entries.extend(method_entries)

    # Extend entries
    global_df = pd.DataFrame(entries)
    tables_to_build = ["AUPRC", "AUCROC"]
    for table_metric in tables_to_build:
        new_df = global_df.query(f"Metric == '{table_metric}'")
        new_df.set_index(["Method Type", "Method"])
        pivoted_df = new_df.pivot_table(
            values=["Value"],
            columns=["Dataset"],  # , "Metric"],
            index=["Method Type", "Method"],
            aggfunc=lambda x: "".join(str(v) for v in x),
        )
        # Remove value from beginning
        pivoted_df.columns = pivoted_df.columns.droplevel()

        with open(
            os.path.join(outdir, f"QSAR_{table_metric}_result_tbl.txt"), "w",
            encoding="utf-8"
        ) as fp:
            latex_str = pivoted_df.to_latex(
                caption=f"Full QSAR results table of {table_metric}",
                escape=False,
                bold_rows=True,
            )
            fp.write(latex_str)


def make_qsar_plot(df, outdir):
    """Make qsar plots"""

    for metric, metric_name in METRIC_NAME_MAPPING.items():

        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(dataset_names),
            sharex=False,
            sharey=False,
            figsize=(panel_width * len(dataset_names), panel_height),
        )
        for index, (ax, dataset_name) in enumerate(zip(axes, dataset_names)):
            plt.sca(ax)
            plot_df = df.query(f"dataset_name == '{dataset_name}'").reset_index(
                drop=True
            )
            dataset_rename = DATASET_NAME_MAP[dataset_name]
            bars = []
            max_std = -99
            min_y, max_y = 100, -99
            for color, plot_position, method in zip(
                x_colors, plot_positions, full_list
            ):
                temp_df = plot_df[plot_df["model_name"] == method]
                temp_df = temp_df.groupby("seed").mean().reset_index()

                vals = temp_df[metric]
                bar_height = np.mean(vals)
                error_height = 1.96 * stats.sem(vals)

                min_y = min(min_y, bar_height - error_height)
                max_y = max(max_y, bar_height + error_height)

                max_std = max(error_height, max_std)
                label_name = method_name_map[method]

                hatch = x_hatch_map.get(x_label_map[label_name], None)
                bars.append(
                    plt.bar(
                        plot_position,
                        bar_height,
                        color=color,
                        label=label_name,
                        width=1.01,
                        hatch=hatch,
                    )
                )
                plt.errorbar(
                    plot_position,
                    bar_height,
                    yerr=error_height,
                    color="Black",
                    capsize=5,
                    capthick=2,
                    linewidth=1,
                )

            p_vals, pairs = compute_pvals(
                plot_df,
                metric,
                compare_method,
                alternative="two-sided",
                stat_test="t_welsh",
            )
            annot_strings = convert_p_val_to_stars(
                [dataset_name], compare_method, full_list, pairs, p_vals
            )

            sorted_patches = [i for j in bars for i in j]
            for p, annot in zip(sorted_patches, annot_strings):
                plt.text(
                    p.get_x() + p.get_width() / 2.0,
                    p.get_height() + max_std / 2 + 0.07,
                    annot,
                    ha="center",
                    fontsize=10,
                    color="red",
                )

            # Set labels
            ticks, labels = [], []
            for label in np.unique(x_labels):
                # Get max pos.
                avg_pos = np.max(np.array(plot_positions)[np.array(x_labels) == label])
                ticks.append(avg_pos)
                labels.append(label)
            plt.xticks(ticks, labels=labels, rotation=40, horizontalalignment="right")
            if index == 0:
                plt.ylabel(metric_name)

            plt.title(dataset_rename)

            # Set y lim
            plt.ylim([min_y - 0.02, max_y + 0.2])

        plt.sca(axes[1])
        plt.legend(
            ncol=3, bbox_to_anchor=(0.5, -0.9), loc="center"  # int(len(full_list) / 2),
        )

        plt.subplots_adjust(wspace=0.3)
        plt.savefig(os.path.join(outdir, f"QSAR_{metric}.pdf"), bbox_inches="tight")


def make_auxilary_qsar(df, outdir):
    """Make auxilary qsar"""
    metric, metric_name = "avg-pr", "Average Precision"

    entries = []
    for dataset_name in dataset_names:
        temp_df = df.query(f"dataset_name == '{dataset_name}'").reset_index()
        temp_df = temp_df.query("model_name == 'morgan1024 linear'").reset_index()

        data = f"data/processed/{dataset_name}.csv"
        temp_data = pd.read_csv(data, index_col=0)
        balance = temp_data.groupby("SEQ").mean().reset_index()
        val_key = list(set(balance.keys()).difference({"SEQ"}))[0]
        balance_dict = dict(zip(balance["SEQ"].values, balance[val_key].values))

        prs = temp_df.groupby("TASK")[metric].mean().reset_index()
        pr_dict = dict(zip(prs["TASK"].values, prs[metric].values))

        x, y = [], []
        for sub, pr in pr_dict.items():
            balance = balance_dict[sub]
            entropy = -balance * np.log(balance) - (1 - balance) * np.log(1 - balance)
            entries.append(
                {
                    "Dataset": DATASET_NAME_MAP[dataset_name],
                    "Balance": balance,
                    metric_name: pr,
                    "Entropy": entropy,
                    "Num. actives": balance,
                }
            )

    balance_df = pd.DataFrame(entries)

    fig = plt.figure(figsize=(6.2, 4))
    ax = fig.gca()
    pal = sns.color_palette("Set1", n_colors=len(dataset_names))
    colors = {
        dataset: pal[index]
        for index, dataset in enumerate(pd.unique(balance_df["Dataset"]))
    }
    for dataset_name, sub_df in balance_df.groupby("Dataset"):
        color = colors[dataset_name]
        sns.regplot(
            data=sub_df,
            x="Num. actives",
            y=metric_name,
            order=1,
            line_kws={"linewidth": 1.4},
            scatter_kws={"edgecolor": "none", "s": 60},
            ci=None,
            label=dataset_name,
            color=color,
            ax=ax,
        )

    plt.ylabel("AUPRC")
    plt.legend(bbox_to_anchor=(0.5, -0.5), ncol=3, loc="center")
    out_loc = os.path.join(outdir, f"QSAR_data_balance.pdf")
    plt.savefig(out_loc, bbox_inches="tight")

    sub_df = df[
        [
            i in ["morgan1024 ffn(prot,chem)", "morgan1024 linear"]
            for i in df["model_name"]
        ]
    ]

    sub_df = sub_df.pivot_table(
        values="avg-pr", index=["dataset_name", "TASK"], columns=["model_name"]
    )

    sub_df = sub_df.reset_index()

    sub_df["Dataset"] = [DATASET_NAME_MAP[i] for i in sub_df["dataset_name"]]

    model_1 = "morgan1024 linear"
    model_2 = "morgan1024 ffn(prot,chem)"

    panel_height = 3.5
    panel_width = 6
    fig = plt.figure(figsize=(panel_width, panel_height))

    for dataset_name in dataset_names:
        dataset_name = DATASET_NAME_MAP[dataset_name]
        dataset_df = sub_df[sub_df["Dataset"] == dataset_name]
        if len(dataset_df) == 0:
            continue
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            dataset_df[model_1], dataset_df[model_2]
        )
        color = colors[dataset_name]
        ax = sns.regplot(
            data=dataset_df,
            x=model_1,
            y=model_2,
            order=1,
            line_kws={"linewidth": 1.4},
            ci=None,
            label=f"{dataset_name} ($m={slope:0.2f}$)",
            color=color,
        )

    x_min, x_max = sub_df[model_1].min(), sub_df[model_1].max()
    y_min, y_max = sub_df[model_2].min(), sub_df[model_2].max()
    plt.plot(
        np.linspace(0, 1, 100),
        np.linspace(0, 1, 100),
        color="Black",
        alpha=0.5,
        linestyle="--",
    )
    plt.ylim([y_min - 0.02, y_max + 0.02])
    plt.xlim([x_min - 0.02, x_max + 0.02])
    plt.xlabel(method_name_map[model_1])
    plt.ylabel(method_name_map[model_2])
    ax.legend(
        bbox_to_anchor=(1.24, 1.0),
    )
    out_loc = os.path.join(outdir, f"QSAR_data_regress_scatter.pdf")
    plt.savefig(out_loc, bbox_inches="tight")

    sub_df[r"$\frac{CPI}{Single task}$"] = sub_df[model_2] / sub_df[model_1]

    plt.figure(figsize=(4.2, 4.0))
    hue_col_order = [DATASET_NAME_MAP[dataset_name] for dataset_name in dataset_names]
    sub_df = sub_df[[j in hue_col_order for j in sub_df["Dataset"]]]
    sns.stripplot(
        x="Dataset",
        y=r"$\frac{CPI}{Single task}$",
        data=sub_df,
        hue_order=hue_col_order,
        palette=colors,
        linewidth=0.9,
    )
    # fliersize = 1.5)#, width=2.5)

    plt.hlines(y=1.0, xmin=-0.15, xmax=2.1, linestyle="--", color="red")
    plt.xticks(rotation=20, horizontalalignment="center")
    plt.xlabel("")
    out_loc = os.path.join(outdir, f"QSAR_cpi_v_single.pdf")
    plt.savefig(out_loc, bbox_inches="tight")


if __name__ == "__main__":
    results_file = "results/dense/2021_05_28_qsar_with_multi/qsar_combined.csv"
    outdir = "results/figure_export/"
    os.makedirs(outdir, exist_ok=True)

    df = extract_csv_file(
        results_file, extract_qsar_sweep_names, extract_dataset_name, rename=True
    )
    df = df.query("dataset_split == 'test'").reset_index(drop=True)

    # Make auxilary qsar table
    make_auxilary_qsar(df, outdir)

    # Make qsar results table
    make_qsar_results_tables(df, outdir)

    # Make qsar results plots
    make_qsar_plot(df, outdir)
