"""Create datset summary"""
import os
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw, rdDepictor

from enzpred.dataset import dataloader
from enzpred.utils import parse_utils

from helpers import *

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

VALID_FILES = [
    "phosphatase_chiral_binary.csv",
    "gt_acceptors_chiral_binary.csv",
    "olea_binary.csv",
    "halogenase_NaBr_binary.csv",
    "esterase_binary.csv",
    "duf_binary.csv",
]


def make_dataset_summary_table(dir_prefix, outdir):
    """Make latex summaries"""
    entries = []
    enzyme_files = [i for i in os.listdir(dir_prefix) if ".csv" in i]
    for enzyme_file in enzyme_files:
        if enzyme_file.startswith("."):
            continue
        df = pd.read_csv(os.path.join(dir_prefix, enzyme_file))
        subs = df["SUBSTRATES"]
        num_entries = len(df)
        num_subs = len(pd.unique(df["SUBSTRATES"]))
        num_seqs = len(pd.unique(df["SEQ"]))

        ### DF value column:
        value_col = list(
            set(df.keys()).difference(set(["Unnamed: 0", "SEQ", "SUBSTRATES"]))
        )[0]

        entry = {
            "Dataset": enzyme_file,
            "Num entries": num_entries,
            "# Seqs.": num_seqs,
            "# Subs.": num_subs,
        }

        ### If binary
        if "binary" in enzyme_file:
            for axis in ["SUBSTRATES", "SEQ"]:
                num_valid = 0
                # Check how many enzymes are active here.
                for sub_name, df_sub in df.groupby(axis):
                    temp_values = df_sub[value_col].values
                    skip = dataloader.skip_col(temp_values)
                    if not skip:
                        num_valid += 1
                entry[f"Valid {axis}"] = num_valid
        entries.append(entry)
    df = pd.DataFrame(entries)

    out_table = df[~pd.isnull(df["Valid SEQ"])].rename(
        {
            "Valid SUBSTRATES": "Valid subs.",
            "Valid SEQ": "Valid seqs.",
        },
        axis=1,
    )
    caption = "Summary of valid substrate and sequence ``tasks'' in each dataset"
    out_name = "dataset_latex_table.txt"
    with open(os.path.join(outdir, out_name), "w") as fp:
        out_str = out_table.to_latex(caption=caption, index=False, bold_rows=True)
        fp.write(out_str)


def smi_to_fp_ar(smi):
    """Convert smiles to morgan FP"""
    mol = Chem.MolFromSmiles(smi)
    fp_fn = lambda m: AllChem.GetMorganFingerprintAsBitVect(
        m, 2, nBits=2048, useChirality=True
    )
    fingerprint = fp_fn(mol)
    array = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return array


def diversity_analysis(alignment_dir, dir_prefix, outdir):
    """Dataset diversity analysis"""

    # Get enzymes diversity df
    entries = []
    for file_ in VALID_FILES:
        full_file = os.path.join(dir_prefix, file_)
        alignment_post = file_.split("_")[0].split(".")[0]
        alignment_file = os.path.join(
            alignment_dir, f"{alignment_post}_alignment.fasta"
        )
        alignment_dict = {
            j.replace("-", "").strip(): j
            for i, j in parse_utils.fasta_iter(alignment_file)
        }

        dataset_name = DATASET_NAME_MAP.get(file_.split(".")[0].strip())

        df = pd.read_csv(full_file, index_col=0)
        seqs = pd.unique(df["SEQ"])

        aligned_seqs = [np.array(list(alignment_dict.get(seq))) for seq in seqs]

        sim_mat = np.zeros((len(seqs), len(seqs)))
        for index_1, i in enumerate(aligned_seqs):
            for index_2, j in enumerate(aligned_seqs[index_1:], index_1):
                # Remove identity
                if index_1 == index_2:
                    continue
                both_gapped = np.logical_and(i == "-", j == "-")
                temp_i = i[~both_gapped]
                temp_j = j[~both_gapped]

                percent_sim = np.mean(temp_i == temp_j)
                sim_mat[index_1, index_2] = percent_sim
                sim_mat[index_2, index_1] = percent_sim

        # Top 5 percent identity
        k = 5
        percent_identities = np.sort(sim_mat, axis=1,)[:, ::-1][
            :, :k
        ].mean(1)
        for percent_identity in percent_identities:
            new_entry = {
                "Dataset": dataset_name,
                f"Top-{k} Similarity": percent_identity,
            }
            entries.append(new_entry)
    df_enzyme = pd.DataFrame(entries)

    entries = []
    for file_ in VALID_FILES:
        full_file = os.path.join(dir_prefix, file_)

        dataset_name = DATASET_NAME_MAP.get(file_.split(".")[0].strip())

        df = pd.read_csv(full_file, index_col=0)
        subs = pd.unique(df["SUBSTRATES"])
        feats = np.vstack([smi_to_fp_ar(j) for j in subs])

        # Calculate tanimoto distance with binary fingerprint
        intersect_mat = feats[:, None, :] & feats[None, :, :]
        union_mat = feats[:, None, :] | feats[None, :, :]

        intersection = intersect_mat.sum(-1)
        union = union_mat.sum(-1)
        sim_mat = intersection / union

        # Zero diagonal
        ar_ind = np.arange(len(sim_mat))
        sim_mat[ar_ind, ar_ind] = 0

        k = 5
        tani_sims = np.sort(sim_mat, axis=1,)[:, ::-1][
            :, :k
        ].mean(1)
        for tani_sim in tani_sims:
            new_entry = {"Dataset": dataset_name, f"Top-{k} Tanimoto": tani_sim}
            entries.append(new_entry)

    df_sub = pd.DataFrame(entries)

    sns.set_palette(sns.color_palette("tab10"))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    plt.sca(axes[0])
    sns.violinplot(
        data=df_enzyme,
        x="Dataset",
        y=f"Top-{k} Similarity",
    )
    plt.xticks(rotation=60)
    plt.ylabel(f"Top-{k} Enzyme MSA Percent Identity")

    plt.sca(axes[1])
    sns.violinplot(
        data=df_sub,
        x="Dataset",
        y=f"Top-{k} Tanimoto",
    )
    plt.xticks(rotation=60)

    plt.ylabel(f"Top-{k} Tanimoto Similarity")
    plt.suptitle("Enzyme and Substrate Dataset Diversity")
    save_name = os.path.join(outdir, "appendix_dataset_diversity.pdf")
    plt.savefig(save_name, bbox_inches="tight")


def draw_substrates(outdir, dir_prefix, k=6):
    """Draw substrates."""

    outdir = os.path.join(outdir, "appendix_substrate_figs")
    os.makedirs(outdir, exist_ok=True)
    rdDepictor.SetPreferCoordGen(True)
    for file_ in VALID_FILES:
        full_file = os.path.join(dir_prefix, file_)
        dataset_name = DATASET_NAME_MAP.get(file_.split(".")[0].strip())

        df = pd.read_csv(full_file, index_col=0)
        subs = pd.unique(df["SUBSTRATES"])
        mols = np.random.choice([Chem.MolFromSmiles(j) for j in subs], k)
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=k,
            useSVG=True,
            subImgSize=(400, 400),
        )
        output_svg = os.path.join(outdir, f"{dataset_name}.svg")
        with open(output_svg, "w") as f_handle:
            f_handle.write(img)

if __name__ == "__main__":
    dir_prefix = "data/processed/"
    outdir = "results/figure_export/"
    alignment_dir = "data/processed/alignments/"

    os.makedirs(outdir, exist_ok=True)

    make_dataset_summary_table(dir_prefix, outdir)
    diversity_analysis(alignment_dir, dir_prefix, outdir)
    draw_substrates(outdir, dir_prefix, k=6)
