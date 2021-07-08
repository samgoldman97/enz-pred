"""Classes to handle distance metrics for KNN 
"""

import os
from typing import Tuple, Optional
import tempfile
import logging
import subprocess
from pathos import multiprocessing

import numpy as np
import random
import time
from tqdm import tqdm
import pandas as pd
import pickle
from functools import partial

import torch

from torch.utils.data.dataset import Dataset as TorchDataset
from Levenshtein import distance as levenshtein_distance

# Import blast helpers
from Bio.Blast.Applications import NcbiblastpCommandline
from Bio.Blast import NCBIXML
from io import StringIO

# Helpers for pairwise alignment
from Bio.Align import substitution_matrices

from enzpred.utils import file_utils, parse_utils, ssa_utils

SEQ_CHOICES = [
    "levenshtein",
    "hamming",
    "blast",
    "cosine-dist",
    "l2-dist",
    "l1-dist",
    None,
]
SUBSTRATE_CHOICES = ["tanimoto", "l1-dist", None]

LARGE_INT = 9999

MODEL_ARGS = [
    (
        ["--align-dist"],
        dict(
            action="store",
            default=None,
            choices=[
                None,
                "sw",
                "nw",
                "ssa",
                "rssa",
                "rssa_and",
                "rssa_random",
                "rssa_full",
            ],
            help="How to align the distances first when computing knn with proteins",
        ),
    ),
]

# Helper fn
def none_fn(*args, **kwargs):
    None


def get_seq_dist(seq_dist_type: str, **kwargs):
    """Get protein featurers"""
    return {
        "levenshtein": LevenshteinDistance,
        "blast": BLASTDistance,
        "hamming": HammingDistance,
        "l1-dist": L1ProtFeatureDistance,
        "l2-dist": L2ProtFeatureDistance,
        "cosine-dist": CosineDistance,
        None: none_fn,
    }[seq_dist_type](**kwargs)


def get_sub_dist(sub_dist_type: str, **kwargs):
    """Get substrate featurers"""
    return {"tanimoto": TanimotoDistance, "l1-dist": L1SubDist, None: none_fn}[
        sub_dist_type
    ](**kwargs)


# Sequence helper functions for alignment
def get_nw(seq_1: str, seq_2: str, msa_align_dict: dict = None) -> Tuple[str, str]:
    """
    Needleman wunsch call to subprocess using emboss

    Args:
        seq_1 (str) : Input sequence
        seq_2 (str) : Input sequence
        msa_align_dict (dict): Mapping of each sequence to its

    Return:
        Aligned sequence 1
        Aligned sequence 2
    """
    # alignment = pairwise2.align.globalds(seq_1, seq_2,
    #                                     self.sub_mat,
    #                                     self.gap_open,
    #                                     self.gap_extend,
    #                                     one_alignment_only = True)[0]
    # return alignment.seqA, alignment.seqB

    if msa_align_dict is None:

        with tempfile.TemporaryDirectory() as td:
            s1_fasta = os.path.join(td, "s1.fasta")
            s2_fasta = os.path.join(td, "s2.fasta")
            align_output = os.path.join(td, "aligned.fasta")

            with open(s1_fasta, "w") as fp:
                fp.write(f">item_1\n{seq_1}")
            with open(s2_fasta, "w") as fp:
                fp.write(f">item_2\n{seq_2}")

            cmd = f"needle -asequence {s1_fasta} -bsequence {s2_fasta} -gapopen 10 -gapextend 0.5 -outfile {align_output} -aformat3 fasta"
            subprocess.call(
                cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            outputs = [seq for _, seq in parse_utils.fasta_iter(align_output)]

    else:
        if seq_1 not in msa_align_dict or seq_2 not in msa_align_dict:
            raise ValueError(f"Could not find {seq_1} or {seq_2} in the msa")

        outputs = [msa_align_dict[seq_1], msa_align_dict[seq_2]]

    return outputs[0], outputs[1]


def get_sw(seq_1: str, seq_2: str, readout: str = "e_value"):
    """
    Smith waterman pairwise blast!

    Args:
        seq_1 (str) : Input sequence
        seq_2 (str) : Input sequence
        readout (str): String containing the readout method

    Return:
        if readout == "align_seqs":
            Aligned sequence 1
            Aligned sequence 2
        elif readout == "e_value":
            e_value
    """

    s1_fasta = tempfile.NamedTemporaryFile(mode="w+b")
    s2_fasta = tempfile.NamedTemporaryFile(mode="w+b")

    s1_fasta.write(f">seq1\n{seq_1}".encode())
    s2_fasta.write(f">seq2\n{seq_2}".encode())

    # Seek to 0
    s1_fasta.seek(0)
    s2_fasta.seek(0)

    output = NcbiblastpCommandline(
        query=s1_fasta.name, subject=s2_fasta.name, outfmt=5
    )()[0]

    blast_results = NCBIXML.read(StringIO(output))

    alignments = []
    # Now retrieve and parse alignments
    for alignment in blast_results.alignments:
        hsps = []
        for hsp in alignment.hsps:
            hsp = {
                "query": hsp.query,
                "query_start": hsp.query_start,
                "query_end": hsp.query_end,
                "subject": hsp.sbjct,
                "subject_start": hsp.sbjct_start,
                "subject_end": hsp.sbjct_end,
                "e_val": hsp.expect,
            }
            hsps.append(hsp)
        alignments.append(hsps)

    # Get minimum e value
    if readout == "e_value":
        e_vals = [hsp["e_val"] for align in alignments for hsp in align]
        return np.min(e_vals) if len(e_vals) > 0 else LARGE_INT

    # Return pairwise global alignments, adding gaps for spaces outside of the local align
    elif readout == "align_seqs":
        # If no alignments, offset completely
        if len(alignments) == 0:
            return (("-" * len(seq_2) + seq_1), (seq_2 + "-" * len(seq_1)))
        min_entry = min(
            [hsp for align in alignments for hsp in align], key=lambda x: x["e_val"]
        )

        # Adjust these to both have the appropriate padding in front and in
        # back to offset completely
        qstart, qend, q = (
            min_entry["query_start"],
            min_entry["query_end"],
            min_entry["query"],
        )
        sstart, send, s = (
            min_entry["subject_start"],
            min_entry["subject_end"],
            min_entry["subject"],
        )

        seq_1_prefix, seq_1_suffix = seq_1[: qstart - 1], seq_1[qend:]
        seq_2_prefix, seq_2_suffix = seq_2[: sstart - 1], seq_2[send:]

        s1_pre_len, s1_post_len = len(seq_1_prefix), len(seq_1_suffix)
        s2_pre_len, s2_post_len = len(seq_2_prefix), len(seq_2_suffix)

        seq_1_prefix = seq_1_prefix + "-" * s2_pre_len
        seq_2_prefix = "-" * s1_pre_len + seq_2_prefix

        seq_1_suffix = seq_1_suffix + "-" * s2_post_len
        seq_2_suffix = "-" * s1_post_len + seq_2_suffix

        s1_align = seq_1_prefix + q + seq_1_suffix
        s2_align = seq_2_prefix + s + seq_2_suffix
        return s1_align, s2_align
    else:
        raise NotImplementedError(f"No readout {readout}")


class BaseDistance:
    def __init__(self, **kwargs):
        # Stored distance
        self.distance = {}
        super(BaseDistance, self).__init__()
        self.fill_precomputed(**kwargs)

    def dist(self, ref_dataset: TorchDataset, test_dataset: TorchDataset) -> np.ndarray:
        """Return a distance matrix between x1 and x2

        Return:
            np.ndarray: Distance matrix with shape len(new_dataset) x
                len(old_datset)
        """
        raise NotImplementedError()

    def create_cache_str(self):
        """Create a cache string about the parameters"""
        return None

    def fill_precomputed(self, cache_dir: str = None, **kwargs):
        """Fill self.distance from cache"""
        # Get cache info
        model_cache_str = self.create_cache_str()
        if cache_dir and model_cache_str:
            cache_file = os.path.join(cache_dir, model_cache_str)
            if os.path.isfile(cache_file):
                logging.info(f"Loading cache file {cache_file} for knn")

                # In this while loop we should give room for parallel processes
                # by randomly sleeping a few times
                attempts = 0
                MAX_SLEEP = 10
                preloaded = False
                while attempts < 5 and not preloaded:
                    attempts += 1
                    try:
                        my_obj = file_utils.pickle_load(cache_file)
                        # if we can do this load, we should not go back in this
                        # loop!
                        preloaded = True
                    except pickle.UnpicklingError:
                        logging.info(
                            f"Unable to load {cache_file} due to pickling error"
                        )
                        sleep_time = random.random() * MAX_SLEEP
                        logging.info(
                            f"Sleeping {sleep_time:0.2f} seconds and trying again"
                        )
                        time.sleep(sleep_time)
                        my_obj = {}
                    except EOFError:
                        logging.info(f"Unable to load {cache_file} due to EOF error")
                        sleep_time = random.random() * MAX_SLEEP
                        logging.info(
                            f"Sleeping {sleep_time:0.2f} seconds and trying again"
                        )
                        time.sleep(sleep_time)
                        my_obj = {}
                if isinstance(my_obj, dict):
                    self.distance.update(my_obj)
                    logging.info(f"Loaded feature cache of size {len(self.distance)}")

    def dump_cache(self, cache_dir: str = None, **kwargs):
        """Dump the stored cache to an output file"""
        model_cache_str = self.create_cache_str()
        if cache_dir and model_cache_str:
            cache_file = os.path.join(cache_dir, model_cache_str)
            file_utils.pickle_obj(self.distance, cache_file)

    def pairwise_fn(self, item_1, item_2):
        """Pairwise distance function"""
        raise NotImplementedError()

    def rank_sparse_dist(
        self, items_1: list, items_2: list, top_n: int = 10, **kwargs
    ) -> np.ndarray:
        """rank_sparse_dist.

        Compute only the top N nearest items for everything in items_1 to
        items_2. This is made difficult because there are many repetitions in
        items_1, so brute force doesn't work. Consider the case where we want
        to interpolate in a 200 by 100 grid of seq vs. substrate. This is
        20,000 examples. If we want to do K nearest neighbors on this and make
        the full dist matrix, this becomes 400,000,000 values to compute.

        Instead, for each of the 20,000, only compute the top n nearest ranked
        values. This is easy because there's only 200 unique examples

        Set the default value to (top_n + 1)

        Args:
            items_1 (list): items_1
            items_2 (list): items_2
            top_n (int): top_n
            kwargs
        Returns:
            np.ndarray:
        """

        # Assume that distance is symmetric
        logging.info("Starting to fill dist cache")

        # If items_1 and items_2 are np.ndarray, convert to tuples
        if isinstance(items_1[0], np.ndarray):
            items_1 = [tuple(i) for i in items_1]
        if isinstance(items_2[0], np.ndarray):
            items_2 = [tuple(i) for i in items_2]

        uniq_1, uniq_2 = set(items_1), set(items_2)
        for item_1 in tqdm(uniq_1):
            for item_2 in uniq_2:
                if (item_1, item_2) not in self.distance:
                    dist = self.pairwise_fn(item_1, item_2)
                    self.distance[(item_1, item_2)] = dist
                    self.distance[(item_2, item_1)] = dist

        # Now fill out the distance matrix
        # We need to make this a sparse matrix

        # Make concise matrix
        item_to_index_1 = {v: k for k, v in enumerate(uniq_1)}
        item_to_index_2 = {v: k for k, v in enumerate(uniq_2)}

        # Make index by index grid
        concise_mat = np.zeros((len(item_to_index_1), len(item_to_index_2)))
        for item_1, index_1 in tqdm(item_to_index_1.items()):
            for item_2, index_2 in item_to_index_2.items():
                # row is index 1, column is index 2
                concise_mat[index_1, index_2] = self.distance[(item_1, item_2)]

        # This maps each row to a sorted row from closest to furtherest
        # Sort by columns
        argsorted = np.argsort(concise_mat, 1)[:, :top_n]

        # Now get index selection on second axis
        # This maps everything in second set to the concise index
        true_pos_2_index = np.zeros(len(items_2))
        for index, item_2 in tqdm(enumerate(items_2)):
            true_pos_2_index[index] = item_to_index_2[item_2]

        # Get top n for each item 1
        return_mat = np.ones((len(items_1), len(items_2))).astype(np.uint8) * (
            top_n + 1
        )
        for pos_1, item_1 in tqdm(enumerate(items_1)):
            argsorted_index = item_to_index_1[item_1]
            for rank_value, target_index in enumerate(argsorted[argsorted_index]):
                # Now we want to map target_index to seqs in
                dist_value = concise_mat[argsorted_index, target_index]
                # Now find all seq_2 that satisfy this dist value
                item_2_targets = np.argwhere(true_pos_2_index == target_index).flatten()

                # sparse_dist[pos_1, seq_2_targets] = dist_value
                # Store rank value, not dist value!
                return_mat[pos_1, item_2_targets] = rank_value

        self.dump_cache(**kwargs)
        return return_mat

    def check_shape(self, ar: np.array):
        """Make sure this could be converted effectively to an ar"""
        if not len(ar.shape) == 2:
            raise ValueError(f"Unexpected error converting {ar} to np.array")

        # dataset_data =  dataset.get_feature_df()
        # feature_ar = []

        # if "rxn_features" in dataset_data:
        #    rxn_feats =  np.stack(dataset_data["rxn_features"].values, axis=0)
        #    check_shape(rxn_feats)
        #    feature_ar.append(rxn_feats)

        # if "prot_features" in dataset_data:
        #    prot_feats =  np.stack(dataset_data["prot_features"].values, axis=0)
        #    check_shape(prot_feats)
        #    feature_ar.append(prot_feats)

        # if len(feature_ar) > 0:
        #    concat_ar = np.concatenate(feature_ar, axis=1)
        # else:
        #    concat_ar = []
        ##logging.info(f"Shape of concatenated features {concat_ar.shape}")
        # return concat_ar


class HammingDistance(BaseDistance):
    """Implement levenshtein distance between datasets"""

    def __init__(self, **kwargs):
        super(HammingDistance, self).__init__(**kwargs)

    def create_cache_str(self):
        """Create a cache string about the parameters"""
        obj = {"distance": "hamming"}
        return file_utils.md5(str(obj))

    def pairwise_fn(self, item_1, item_2):
        """Pairwise distance function"""

        assert isinstance(item_1, str)
        assert isinstance(item_2, str)

        # Compute fraction overlap for non gapped positions
        numerator = 0
        divisor = 0
        for i, j in zip(item_1, item_2):
            if i == "-" or j == "-":
                pass
            else:
                divisor += 1
                if i != j:
                    numerator += 1

        # Max distance
        if divisor == 0:
            return 1
        else:
            result = numerator / divisor
            return result

    def rank_sparse_dist(
        self, ref_dataset: TorchDataset, test_dataset: TorchDataset, top_n=8, **kwargs
    ) -> np.ndarray:
        """See super class.

        Return:
            np.ndarray: Distance matrix with shape len(new_dataset) x
                len(old_datset)
        """

        seqs_1 = test_dataset.get_feature_df()["prot_features"]
        seqs_2 = ref_dataset.get_feature_df()["prot_features"]
        ret = super(HammingDistance, self).rank_sparse_dist(
            seqs_1, seqs_2, top_n=top_n, **kwargs
        )
        return ret

    def dist(
        self, ref_dataset: TorchDataset, test_dataset: TorchDataset, top_n=8, **kwargs
    ) -> np.ndarray:
        """Return a distance matrix between x1 and x2

        Return:
            np.ndarray: Distance matrix with shape len(new_dataset) x
                len(old_datset)
        """
        seqs_1 = test_dataset.get_feature_df()["prot_features"].values
        seqs_2 = ref_dataset.get_feature_df()["prot_features"].values

        # Assume that distance is symmetric
        logging.info("Starting to fill dist cache")
        uniq_1, uniq_2 = set(seqs_1), set(seqs_2)
        for item_1 in tqdm(uniq_1):
            for item_2 in uniq_2:
                if (item_1, item_2) not in self.distance:
                    dist = self.pairwise_fn(item_1, item_2)
                    self.distance[(item_1, item_2)] = dist
                    self.distance[(item_2, item_1)] = dist

        # Now fill out the distance matrix
        # We need to make this a sparse matrix
        return_mat = np.zeros((len(seqs_1), len(seqs_2)))
        logging.info("Starting to fill dist matrix with cache")
        for pos_1, seq_1 in tqdm(enumerate(seqs_1)):
            for pos_2, seq_2 in enumerate(seqs_2):
                return_mat[pos_1, pos_2] = self.distance[(seq_1, seq_2)]

        self.dump_cache(**kwargs)
        return return_mat


class LevenshteinDistance(BaseDistance):
    """Implement levenshtein distance between datasets"""

    def __init__(self, **kwargs):
        super(LevenshteinDistance, self).__init__(**kwargs)

    def create_cache_str(self):
        """Create a cache string about the parameters"""
        obj = {"distance": "levenshtein"}
        return file_utils.md5(str(obj))

    def pairwise_fn(self, item_1, item_2):
        """Pairwise distance function"""
        return levenshtein_distance(item_1, item_2)

    def rank_sparse_dist(
        self, ref_dataset: TorchDataset, test_dataset: TorchDataset, top_n=8, **kwargs
    ) -> np.ndarray:
        """See super class.

        Return:
            np.ndarray: Distance matrix with shape len(new_dataset) x
                len(old_datset)
        """

        seqs_1 = test_dataset.get_seqs().values
        seqs_2 = ref_dataset.get_seqs().values
        ret = super(LevenshteinDistance, self).rank_sparse_dist(
            seqs_1, seqs_2, top_n=top_n, **kwargs
        )
        return ret

    def dist(
        self, ref_dataset: TorchDataset, test_dataset: TorchDataset, top_n=8, **kwargs
    ) -> np.ndarray:
        """Return a distance matrix between x1 and x2

        Return:
            np.ndarray: Distance matrix with shape len(new_dataset) x
                len(old_datset)
        """
        seqs_1 = test_dataset.get_seqs().values
        seqs_2 = ref_dataset.get_seqs().values

        # Assume that distance is symmetric
        logging.info("Starting to fill dist cache")
        uniq_1, uniq_2 = set(seqs_1), set(seqs_2)
        for item_1 in tqdm(uniq_1):
            for item_2 in uniq_2:
                if (item_1, item_2) not in self.distance:
                    dist = levenshtein_distance(item_1, item_2)
                    self.distance[(item_1, item_2)] = dist
                    self.distance[(item_2, item_1)] = dist

        # Now fill out the distance matrix
        # We need to make this a sparse matrix
        return_mat = np.zeros((len(seqs_1), len(seqs_2)))
        logging.info("Starting to fill dist matrix with cache")
        for pos_1, seq_1 in tqdm(enumerate(seqs_1)):
            for pos_2, seq_2 in enumerate(seqs_2):
                return_mat[pos_1, pos_2] = self.distance[(seq_1, seq_2)]

        self.dump_cache(**kwargs)
        return return_mat


class BLASTDistance(BaseDistance):
    """Implement BLAST Dist between seqs with an E value"""

    def __init__(self, blast_readout="e_value", **kwargs):
        """__init__.

        Args:
            blast_readout: Defaults to 'e_value'
            kwargs:
        """

        self.readout = blast_readout
        super(BLASTDistance, self).__init__(**kwargs)

    def create_cache_str(self):
        """Create a cache string about the parameters"""
        obj = {"distance": "ncbi_blast", "blast_readout": self.readout}
        return file_utils.md5(str(obj))

    def pairwise_fn(self, item_1, item_2):
        """Pairwise distance function -- blast"""
        return get_sw(item_1, item_2, readout=self.readout)

    def rank_sparse_dist(
        self, ref_dataset: TorchDataset, test_dataset: TorchDataset, top_n=8, **kwargs
    ) -> np.ndarray:
        """See super class.

        Return:
            np.ndarray: Distance matrix with shape len(new_dataset) x
                len(old_datset)
        """

        seqs_1 = test_dataset.get_seqs().values
        seqs_2 = ref_dataset.get_seqs().values
        ret = super(BLASTDistance, self).rank_sparse_dist(
            seqs_1, seqs_2, top_n=top_n, **kwargs
        )
        return ret

    def dist(
        self, ref_dataset: TorchDataset, test_dataset: TorchDataset, top_n=8, **kwargs
    ) -> np.ndarray:
        """Return a distance matrix between x1 and x2

        Return:
            np.ndarray: Distance matrix with shape len(new_dataset) x
                len(old_datset)
        """
        seqs_1 = test_dataset.get_seqs().values
        seqs_2 = ref_dataset.get_seqs().values

        # Assume that distance is symmetric
        logging.info("Starting to fill dist cache")
        uniq_1, uniq_2 = set(seqs_1), set(seqs_2)
        with multiprocessing.ThreadPool(multiprocessing.cpu_count()) as p:
            items_1 = list(uniq_1)
            items_2 = list(uniq_2)

            to_featurize = [
                tuple(sorted((item_1, item_2)))
                for item_1 in items_1
                for item_2 in items_2
                if (item_1, item_2) not in self.distance
            ]
            to_featurize = list(set(to_featurize))

            parallel_fn = lambda x: self.pairwise_fn(x[0], x[1])

            results = []
            if len(to_featurize) > 0:
                results = p.map(parallel_fn, to_featurize)

        for index, ((item_1, item_2), result) in enumerate(zip(to_featurize, results)):
            self.distance[(item_1, item_2)] = result
            self.distance[(item_2, item_1)] = result

        # Now fill out the distance matrix
        # We need to make this a sparse matrix
        return_mat = np.zeros((len(seqs_1), len(seqs_2)))

        logging.info("Starting to fill dist matrix with cache")
        for pos_1, seq_1 in tqdm(enumerate(seqs_1)):
            for pos_2, seq_2 in enumerate(seqs_2):
                return_mat[pos_1, pos_2] = self.distance[(seq_1, seq_2)]

        self.dump_cache(**kwargs)
        return return_mat


class NormFeatureDistance(BaseDistance):
    """Implement generic norm function"""

    def __init__(
        self,
        prot_featurizer: str,
        align_dist: Optional[str] = None,
        seq_msa: str = None,
        coverage_thresh: float = 0.4,
        gpu: bool = False,
        embed_ref_obj: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs,
    ):
        """
        Args:
            prot_featurizer (str): Name of featurizer
            align_dist (str): Parameter for how to align before subtracting
                Options:
                    None: No alignment
                    sw: Smith waterman alignment
                    nw: Needleman wunsch alignment before subtract
                    ssa: Soft symmetric alignment from bepler
                    rssa: Soft symmetric alignment using a reference
                        structure
                    rssa_and: rssa alignment using the ssa alignment and
                        multplication as an "and" gate
                    rssa_random: rssa alignment using the ssa alignment and a
                        reference structure; in this case, define RANDOM
                        residues in the refernce structure rather than the
                        active site chosen. Note: This is randomized in the
                        function call to get the embed ref obj
                    rssa_full: rssa alignment using the ssa alignment and a
                        reference structure; Use the fulll reference structure
                        sequence to compute an alignment
            seq_msa (str): Name of msa file to use as input
            coverage_thresh (float): If using seq_msa, how much coverage do
                you need to consider this to be worth taking the difference
            gpu (bool): If true, use gpu for ssa calc
            embed_ref_obj (Optional[Tuple[np.ndarray,np.ndarray]]):
                Tuple of np.array of positions and np array of embeddings as a
                reference for rssa method. Default None.
        """

        self.align_dist = align_dist
        self.prot_featurizer = prot_featurizer
        self.msa_file = seq_msa
        self.coverage_thresh = coverage_thresh
        self.use_gpu = gpu

        self.embed_ref_obj = embed_ref_obj
        # Recompute with full reference sequence
        if self.align_dist == "rssa_full":
            seqlen = self.embed_ref_obj[1].shape[0]
            self.embed_ref_obj = (np.arange(seqlen), self.embed_ref_obj[1])

        self.norm_fn = None
        self.sub_mat = None
        if self.align_dist:
            # constants
            # self.gap_extend= -0.5
            # self.gap_open = -10
            self.sub_mat = substitution_matrices.load("BLOSUM62")

        # Store a dictionary mapping each sequence to its alignment
        self.msa_align_dict, self.coverage_ar = None, None
        if self.msa_file is not None:
            self.msa_align_dict = {
                seq.replace("-", "").strip(): seq
                for _, seq in parse_utils.fasta_iter(self.msa_file)
            }
            char_matrix = np.array(
                [np.array(list(j)) for j in self.msa_align_dict.values()]
            )
            coverage = np.mean(char_matrix != "-", axis=0)
            self.coverage_ar = coverage > coverage_thresh

        super(NormFeatureDistance, self).__init__(**kwargs)

    def create_cache_str(self):
        """Create a cache string about the parameters"""
        obj = {
            "dist_type": self.__class__,
            "align_dist": self.align_dist,
            "seq_embedding": self.prot_featurizer,
            "msa_align_dict": self.msa_file,
            "coverage_threshold": self.coverage_thresh,
        }

        # Cache based on the reference residues
        # TODO: Correct for ref sequences too
        if self.embed_ref_obj is not None:
            obj["ref_residues"] = str([int(i) for i in self.embed_ref_obj[0]])

        return file_utils.md5(str(obj))

    def extract_feats(self, feat_df: pd.DataFrame, key: str):
        """Extract features of the key"""
        feats = None
        if key in feat_df:
            if self.align_dist is not None:
                feats = list(feat_df[key].values)
            else:
                feats = np.stack(feat_df[key].values, axis=0)
                self.check_shape(feats)

        if feats is None:
            raise ValueError()

        return feats

    def pairwise_fn(self, item_1: tuple, item_2: tuple):
        """Pairwise distance function"""
        raise NotImplementedError()
        # return self.norm_fn(np.array(item_1), np.array(item_2))

    def rank_sparse_dist(
        self,
        ref_dataset: TorchDataset,
        test_dataset: TorchDataset,
        top_n=8,
        key="prot_features",
        **kwargs,
    ) -> np.ndarray:
        """See super class.

        Return:
            np.ndarray: Distance matrix with shape len(new_dataset) x
                len(old_datset)
        """
        feats_1 = test_dataset.get_feature_df()  # .values
        feats_2 = ref_dataset.get_feature_df()  # .values

        feats_1_x = self.extract_feats(feats_1, key)
        feats_2_x = self.extract_feats(feats_2, key)
        ret = super(NormFeatureDistance, self).rank_sparse_dist(
            feats_1_x, feats_2_x, top_n=top_n, **kwargs
        )
        return ret

    def align_dist_fn(
        self,
        norm_fn,
        item_1,
        item_2,
        feat_1,
        feat_2,
        align_dict: Optional[dict] = None,
        coverage_bool: Optional[np.ndarray] = None,
        align_dist: str = "nw",
    ):
        """align_dist_fn."""
        if align_dist == "nw":
            align_1, align_2 = get_nw(item_1, item_2, align_dict)
        elif align_dist == "sw":
            align_1, align_2 = get_sw(item_1, item_2, readout="align_seqs")
        else:
            raise NotImplementedError(f"Align dist '{align_dist}' not implemented ")

        aligned_1, aligned_2 = np.array(list(align_1)), np.array(list(align_2))

        non_gap_1, non_gap_2 = (aligned_1 != "-"), (aligned_2 != "-")

        # the value of align_to_unaligned_1[4] corresponds to
        # the index of position 4 at the aligned sequence IN
        # the unaligned string
        align_to_unaligned_1, align_to_unaligned_2 = (
            np.cumsum(non_gap_1) - 1,
            np.cumsum(non_gap_2) - 1,
        )

        # Need to map aligned_1 and aligned_2 to positional
        # indices that index original sequence

        # Can we only take aligned resiudes and now select from
        # our feats?
        truth_ar = non_gap_1 & non_gap_2

        # subset based on coverage of each position in MSA if supplied
        if coverage_bool is not None:
            truth_ar = truth_ar & coverage_bool

        aligned_residues = np.argwhere(truth_ar).flatten()
        unaligned_position_1 = align_to_unaligned_1[aligned_residues]
        unaligned_position_2 = align_to_unaligned_2[aligned_residues]

        # Now select the aligned positions from the representation
        dist = np.mean(
            norm_fn(feat_1[unaligned_position_1], feat_2[unaligned_position_2])
        )
        return dist

    def dist(
        self,
        ref_dataset: TorchDataset,
        test_dataset: TorchDataset,
        key="prot_features",
        **kwargs,
    ) -> np.ndarray:
        """Return a distance matrix between x1 and x2

        Return:
            np.ndarray: Distance matrix with shape len(new_dataset) x
                len(old_datset)
        """

        feats_1 = test_dataset.get_feature_df()  # .values
        feats_2 = ref_dataset.get_feature_df()  # .values

        feats_1_x = self.extract_feats(feats_1, key)
        feats_2_x = self.extract_feats(feats_2, key)

        seqs_1 = test_dataset.get_seqs().values
        seqs_2 = ref_dataset.get_seqs().values
        start_length_cache = len(self.distance)

        if self.norm_fn is None:
            raise NotImplementedError()

        if self.align_dist is None:

            # If we have more than 10,000 features, don't expand for mem
            # reasons
            if feats_1_x.shape[1] < 10000:
                # Clever way to make a difference matrix with numpy
                return_mat = self.norm_fn(feats_1_x, feats_2_x)
            else:
                return_mat = np.zeros((feats_1_x.shape[0], feats_2_x.shape[0]))
                for i_index, i in tqdm(enumerate(feats_1_x)):
                    for j_index, j in enumerate(feats_2_x):
                        s1, s2 = seqs_1[i_index], seqs_2[j_index]

                        # Pull from cache
                        if (s1, s2) in self.distance:
                            return_mat[i_index, j_index] = self.distance[(s1, s2)]
                        else:
                            norm_dist = self.norm_fn(i, j)
                            return_mat[i_index, j_index] = norm_dist

                            # Store in cache
                            self.distance[(s1, s2)] = norm_dist
                            self.distance[(s2, s1)] = norm_dist

        elif self.align_dist == "ssa":
            # Use gpu for this
            return_mat = np.zeros((len(feats_1_x), len(feats_2_x)))
            for test_index, test_obj in enumerate(feats_1_x):
                test_obj = torch.from_numpy(test_obj)
                test_obj = test_obj.cuda() if self.use_gpu else test_obj
                for ref_index, ref_obj in enumerate(feats_2_x):
                    ref_obj = torch.from_numpy(ref_obj)
                    orig_seq, ref_seq = seqs_1[test_index], seqs_2[ref_index]
                    if (orig_seq, ref_seq) in self.distance:
                        return_mat[test_index, ref_index] = self.distance[
                            (ref_seq, orig_seq)
                        ]
                    else:
                        with torch.no_grad():
                            # Port to GPU
                            ref_obj = ref_obj.cuda() if self.use_gpu else ref_obj
                            attn, dist, norm_dist = ssa_utils.get_ssa_dist(
                                test_obj, ref_obj, self.norm_fn
                            )
                        return_mat[test_index, ref_index] = norm_dist

                        # Store in cache
                        self.distance[(orig_seq, ref_seq)] = norm_dist
                        self.distance[(ref_seq, orig_seq)] = norm_dist

        elif self.align_dist in ["sw", "nw"]:
            # TODO: Make this unique
            # We need to get alignments for all of them
            # Assume that distance is symmetric
            logging.info("Starting to fill distance cache")
            with multiprocessing.ThreadPool(multiprocessing.cpu_count()) as p:
                for item_1, feat_1 in tqdm(zip(seqs_1, feats_1_x)):
                    # Pairwise distance
                    # x is a tuple of item_2, feat_2
                    parallel_fn = lambda x: self.align_dist_fn(
                        self.norm_fn,
                        item_1,
                        x[0],
                        feat_1,
                        x[1],
                        self.msa_align_dict,
                        self.coverage_ar,
                        self.align_dist,
                    )

                    to_featurize = [
                        (item_2, feat_2)
                        for item_2, feat_2 in zip(seqs_2, feats_2_x)
                        if (item_1, item_2) not in self.distance
                    ]

                    results = []
                    if len(to_featurize) > 0:
                        # Should we not close these going forward?
                        results = p.map(parallel_fn, to_featurize)

                    for (item_2, feat_2), dist in zip(to_featurize, results):
                        self.distance[(item_1, item_2)] = dist
                        self.distance[(item_2, item_1)] = dist

            # Now fill out the distance matrix
            return_mat = np.zeros((len(seqs_1), len(seqs_2)))
            logging.info("Starting to fill dist matrix with cache")
            for pos_1, seq_1 in tqdm(enumerate(seqs_1)):
                for pos_2, seq_2 in enumerate(seqs_2):
                    # if np.isnan(self.distance[(seq_1,seq_2)]): print(seq_1, seq_2, "\n")
                    return_mat[pos_1, pos_2] = self.distance[(seq_1, seq_2)]
        elif self.align_dist in ["rssa", "rssa_and", "rssa_random", "rssa_full"]:
            # Use gpu for this
            return_mat = np.zeros((len(feats_1_x), len(feats_2_x)))

            # get the ref positions and embeddings into torchand cuda!
            ref_positions, ref_embedding = self.embed_ref_obj
            ref_embedding = torch.from_numpy(ref_embedding[ref_positions])
            ref_embedding = ref_embedding.cuda() if self.use_gpu else ref_embedding

            for test_index, test_obj in enumerate(feats_1_x):
                test_obj = torch.from_numpy(test_obj)
                test_obj = test_obj.cuda() if self.use_gpu else test_obj

                for train_index, train_obj in enumerate(feats_2_x):
                    train_obj = torch.from_numpy(train_obj)
                    orig_seq, train_seq = seqs_1[test_index], seqs_2[train_index]

                    if (orig_seq, train_seq) in self.distance:
                        return_mat[test_index, train_index] = self.distance[
                            (train_seq, orig_seq)
                        ]
                    else:
                        with torch.no_grad():
                            # Port to GPU
                            train_obj = train_obj.cuda() if self.use_gpu else train_obj

                            use_and_gate = self.align_dist == "rssa_and"
                            cross_attn, dist, norm_dist = ssa_utils.get_rssa_dist(
                                test_obj,
                                train_obj,
                                ref_embedding,
                                self.norm_fn,
                                and_gate=use_and_gate,
                            )

                        return_mat[test_index, train_index] = norm_dist
                        # Store in cache
                        self.distance[(orig_seq, train_seq)] = norm_dist
                        self.distance[(train_seq, orig_seq)] = norm_dist

        else:
            raise ValueError(f"Unexpected align dist {self.align_dist}")

        # Set the return matrix to large int if it's undefined
        return_mat[np.isnan(return_mat)] = LARGE_INT

        # Store these if itwe've added to self.distance
        if start_length_cache != len(self.distance):
            self.dump_cache(**kwargs)

        return return_mat


class L2ProtFeatureDistance(NormFeatureDistance):
    """Implement generic norm function"""

    def __init__(self, **kwargs):
        super(L2ProtFeatureDistance, self).__init__(**kwargs)

        ord_ = 2
        if self.align_dist in ["ssa", "rssa", "rssa_and", "rssa_random", "rssa_full"]:
            self.norm_ar = partial(torch.norm, p=ord_, dim=-1)
        else:
            self.norm_ar = partial(np.linalg.norm, ord=ord_, axis=-1)
        self.norm_fn = self._norm_fn

    def _norm_fn(self, x: np.ndarray, y: np.ndarray):
        """norm_fn"""
        if len(x.shape) == 2 and len(y.shape) == 2:
            diff_mat = x[:, None, :] - y[None, :, :]
            return self.norm_ar(diff_mat)
        elif len(x.shape) == 1 and len(y.shape) == 1:
            return self.norm_ar(x - y)

    def dist(self, *args, **kwargs):
        """Compute Protein distance"""
        return super(L2ProtFeatureDistance, self).dist(
            *args, key="prot_features", **kwargs
        )


class L1ProtFeatureDistance(NormFeatureDistance):
    """Implement generic norm function"""

    def __init__(self, **kwargs):
        super(L1ProtFeatureDistance, self).__init__(**kwargs)
        ord_ = 1
        if self.align_dist in ["ssa", "rssa", "rssa_and", "rssa_random", "rssa_full"]:
            self.norm_ar = partial(torch.norm, p=ord_, dim=-1)
        else:
            self.norm_ar = partial(np.linalg.norm, ord=ord_, axis=-1)
        self.norm_fn = self._norm_fn

    def _norm_fn(self, x: np.ndarray, y: np.ndarray):
        """norm_fn"""
        if len(x.shape) == 2 and len(y.shape) == 2:
            diff_mat = x[:, None, :] - y[None, :, :]
            return self.norm_ar(diff_mat)
        elif len(x.shape) == 1 and len(y.shape) == 1:
            return self.norm_ar(x - y)

    def dist(self, *args, **kwargs):
        """Compute Protein distance"""
        return super(L1ProtFeatureDistance, self).dist(
            *args, key="prot_features", **kwargs
        )


class CosineDistance(NormFeatureDistance):
    """Implement generic norm function"""

    def __init__(self, **kwargs):
        super(CosineDistance, self).__init__(**kwargs)
        self.norm_fn = self._norm_fn

    def _norm_fn(self, x: np.ndarray, y: np.ndarray):
        """norm_fn"""
        eps = 1e-9
        if len(x.shape) == 2 and len(y.shape) == 2:

            diff_mat = (x[:, None, :] * y[None, :, :]).sum(-1)
            x_norm = (x ** 2).sum(-1) ** (1 / 2)
            y_norm = (y ** 2).sum(-1) ** (1 / 2)
            square_mat = x_norm[:, None] * y_norm[None, :]
            diff_mat = diff_mat / (square_mat + eps)
            ret = 1 - diff_mat
            ret[ret < 0] = 0
            return ret

        elif len(x.shape) == 1 and len(y.shape) == 1:
            diff = (x * y).sum(-1)
            x_norm = (x ** 2).sum(-1) ** (1 / 2)
            y_norm = (y ** 2).sum(-1) ** (1 / 2)
            square_mat = x_norm * y_norm
            diff_mat = diff_mat / (square_mat + eps)
            return max(1 - diff_mat, 0)

    def dist(self, *args, **kwargs):
        """Compute Protein distance"""
        return super(CosineDistance, self).dist(*args, key="prot_features", **kwargs)


############ SUBSTRATES ##############
class NormSubDist(BaseDistance):
    """Implement generic norm function"""

    def __init__(self, norm_fn=None, **kwargs):
        super(NormSubDist, self).__init__(**kwargs)
        self.norm_fn = norm_fn

    def create_cache_str(self):
        """Create a cache string about the parameters"""
        return None

    def extract_feats(self, feat_df: pd.DataFrame, key: str):
        """Extract features of the key"""
        feats = None
        if key in feat_df:
            feats = np.stack(feat_df[key].values, axis=0)
            self.check_shape(feats)

        if feats is None:
            raise ValueError()

        return feats

    def pairwise_fn(self, item_1, item_2):
        """Pairwise distance function"""
        item_1 = np.array(item_1)
        item_2 = np.array(item_2)

        # Return the first item of this
        return self.norm_fn(item_1, item_2)[0, 1]

    def rank_sparse_dist(
        self,
        ref_dataset: TorchDataset,
        test_dataset: TorchDataset,
        top_n=8,
        key="rxn_features",
        **kwargs,
    ) -> np.ndarray:
        """See super class.

        Return:
            np.ndarray: Distance matrix with shape len(new_dataset) x
                len(old_datset)
        """

        feats_1 = test_dataset.get_feature_df()  # .values
        feats_2 = ref_dataset.get_feature_df()  # .values

        feats_1_x = self.extract_feats(feats_1, key)
        feats_2_x = self.extract_feats(feats_2, key)
        ret = super(TanimotoDistance, self).rank_sparse_dist(
            feats_1_x, feats_2_x, top_n=top_n, **kwargs
        )
        return ret

    def dist(
        self,
        ref_dataset: TorchDataset,
        test_dataset: TorchDataset,
        key="rxn_features",
        **kwargs,
    ) -> np.ndarray:
        """Return a distance matrix between x1 and x2

        Return:
            np.ndarray: Distance matrix with shape len(new_dataset) x
                len(old_datset)
        """
        feats_1 = test_dataset.get_feature_df()  # .values
        feats_2 = ref_dataset.get_feature_df()  # .values

        feats_1_x = self.extract_feats(feats_1, key)
        feats_2_x = self.extract_feats(feats_2, key)
        if self.norm_fn == None:
            raise ValueError()

        # Calculate tanimoto distance with binary fingerprint

        return self.norm_fn(feats_1_x, feats_2_x)


class TanimotoDistance(NormSubDist):
    """Implement generic norm function"""

    def __init__(self, **kwargs):
        self.norm_fn = lambda x, y: self._norm(x, y)
        super(TanimotoDistance, self).__init__(norm_fn=self.norm_fn, **kwargs)

    def _norm(self, feats_1_x, feats_2_x):
        """Given feature matrix x and feature matrix y, compute tanimoto norm"""
        arb_max = 10000

        # Calculate tanimoto distance with binary fingerprint
        intersect_mat = feats_1_x[:, None, :] & feats_2_x[None, :, :]
        union_mat = feats_1_x[:, None, :] | feats_2_x[None, :, :]

        intersection = intersect_mat.sum(-1)
        union = union_mat.sum(-1)

        ### I took the reciprocal here so instead of tanimoto sim, it became
        # distance. Could have just made negative but
        # sklearn doesn't accept negative distance matrices
        output = union / intersection
        output[output > arb_max] = arb_max
        return output


class L1SubDist(NormSubDist):
    """Implement generic norm function"""

    def __init__(self, **kwargs):
        self.norm_fn = lambda x, y: self._norm(x, y)
        super(L1SubDist, self).__init__(norm_fn=self.norm_fn, **kwargs)

    def _norm(self, feats_1_x, feats_2_x):
        """Given feature matrix x and feature matrix y, compute tanimoto norm"""
        # Calculate tanimoto distance with binary fingerprint
        res_mat = np.linalg.norm(
            feats_1_x[:, None, :] - feats_2_x[None, :, :], ord=1, axis=-1
        )
        return res_mat
