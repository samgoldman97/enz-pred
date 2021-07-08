"""Helper functions for the learning module

"""
import argparse

from itertools import chain
import logging

from enzpred.dataset import dataloader, splitter
from enzpred.features import build_features
from enzpred.features import feature_selection
from enzpred.models import dense_models
from enzpred.evaluation import metrics
from enzpred.utils import parse_utils


def get_parser() -> argparse.ArgumentParser:
    """get_parser.

    Args:

    Returns:
        argparse.ArgumentParser

    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out", type=str, help="Name of outprefix", default="results/out"
    )

    parser.add_argument("--seed", type=int, default=None, help="Rnd seed.")

    parser.add_argument(
        "--dataset-type",
        type=str,
        help="Name of dataset",
        choices=dataloader.DATASET_TYPES,
        default=dataloader.DATASET_TYPES[0],
    )

    parser.add_argument(
        "--chem-featurizer",
        type=str,
        help="Chem featurizer",
        choices=build_features.CHEM_FEATURIZER_TYPES,
    )

    parser.add_argument(
        "--prot-featurizer",
        type=str,
        help="Protein featurizer",
        choices=build_features.PROT_FEATURIZER_TYPES,
    )

    parser.add_argument(
        "--debug-mode", help="Debugging mode", action="store_true", default=False
    )

    parser.add_argument(
        "--export-predictions",
        help="If true, export the dataset splits with predictions",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--gpu", help="If true, use GPU", action="store_true", default=False
    )

    parser.add_argument(
        "--regression",
        help="If true, use a regression dataset",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--model-params-file",
        type=str,
        help="Name of json file to load args from. Used to load optuna results from prev run ",
        default=None,
    )

    parser.add_argument("--save-outputs", action="store_true", default=False)

    parser = add_optuna_args_general(parser)

    for sub_args, sub_kwargs in chain(
        dataloader.DATASET_ARGS,
        build_features.FEATURIZER_ARGS,
        feature_selection.FEATURE_SELECTION_ARGS,
        metrics.METRIC_ARGS,
    ):
        parser.add_argument(*sub_args, **sub_kwargs)

    return parser


def add_optuna_args_general(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """add_optuna_args.

    Add optuna arguments

    Args:
        parser (argparse.ArgumentParser): parser

    Returns:
        argparse.ArgumentParser:
    """

    parser.add_argument(
        "--run-optuna",
        action="store_true",
        help="If true, run hyperparameter optimization",
        default=False,
    )

    # parser.add_argument("--optuna-objective",
    #                    action="store",
    #                    help="If run-hyperopt, optimize for this objective",
    #                    default='auc-roc',
    #                    choices= metrics.METRIC_OPTIONS)
    parser.add_argument(
        "--optuna-trials",
        action="store",
        help="Number of hyperopt trials",
        default=10,
        type=int,
    )

    return parser


def add_optuna_args_dense(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """add_optuna_args_dense.

    Add optuna arguments

    Args:
        parser (argparse.ArgumentParser): parser

    Returns:
        argparse.ArgumentParser:
    """

    parser.add_argument(
        "--optuna-folds",
        action="store",
        help="Number of k folds for optuna CV trial",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--optuna-grid-sample",
        action="store_true",
        help="If true, sample from a grid",
        default=False,
    )

    parser.add_argument(
        "--optuna-global",
        action="store_true",
        help=(
            "If true, run optuna ONCE globally for the"
            "entire dataset rather than in the CV"
        ),
        default=False,
    )

    return parser


def get_dense_parser() -> argparse.ArgumentParser:
    """get_dense_parser..

    Get dense arguments. Builds off of the base parser defined above

    Returns:
        argparse.ArgumentParser:
    """

    # Get base parser
    parser = get_parser()

    parser.add_argument(
        "--splitter-name",
        type=str,
        help="Name of splitter to use",
        choices=splitter.DENSE_SPLITTER_NAMES,
        default="random",
    )
    parser.add_argument(
        "--eval-grouping",
        type=str,
        help="Which direction to group items for evaluation",
        choices=["SEQ", "SUBSTRATES", "ALL"],
        default="ALL",
    )
    parser.add_argument(
        "--scale-prot",
        action="store_true",
        default=False,
        help="If true, scale features",
    )
    parser.add_argument(
        "--scale-chem",
        action="store_true",
        default=False,
        help="If true, scale features",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model type to choose",
        default=dense_models.MODEL_TYPES[0],
        choices=dense_models.MODEL_TYPES,
    )
    parser.add_argument(
        "--ignore-train",
        help="If true, ignore evaluation on the training set",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--pivot-task",
        type=str,
        help=(
            "If this is set to SEQ or SUBSTRATES, reformat"
            "data such that the pivot task is turned into a"
            "multiclass prediction"
        ),
        default=None,
        choices=["SEQ", "SUBSTRATES"],
    )
    parser.add_argument(
        "--frac-train-mask",
        type=float,
        help=("Mask a portion of the training set"),
        default=0.0,
    )
    add_optuna_args_dense(parser)

    for sub_args, sub_kwargs in chain(
        splitter.DENSE_SPLITTER_ARGS, dense_models.MODEL_ARGS
    ):
        parser.add_argument(*sub_args, **sub_kwargs)

    return parser


def get_embed_args() -> argparse.ArgumentParser:
    """get_embed_args..

    Get embed arguments. Builds off of the dense parser defined above

    Returns:
        argparse.ArgumentParser:
    """

    # Get base parser
    parser = get_dense_parser()
    parser.add_argument("--embed-file", help="Name of file containnig seqs to embed")
    parser.add_argument(
        "--saved-model", help="Name of file containing pickled saved model"
    )

    return parser


def modify_parser(args: dict) -> dict:
    """modifying parser"""

    # Baseline model args
    if args["model"] == "random" or args["model"] == "constant":
        logging.info(
            """Changing prot featurizer to cat and chem 
                     featurizer to identity for baseline models"""
        )
        args["chem_featurizer"] = "smiles"

    if args["splitter_name"] == "predetermined" and args["split_indices_file"] is None:
        raise ValueError(
            "For predetermined splitter, must have arg split-indices-file set."
        )
    return args


def modify_parser_dense(args: dict) -> dict:
    """modifying parser"""
    # Base
    args = modify_parser(args)

    # Datset
    args["dataset_type"] = "HTSLoader"

    # KNN
    if args["model"] == "knn":
        if args["seq_dist_type"] in ("l2-dist", "l2-dist") and not args["align_dist"]:
            logging.info("Exiting if pool output strategy is not None")
            assert args["pool_prot_strategy"] is not None
        elif args["seq_dist_type"] in ("l2-dist", "l2-dist") and args["align_dist"]:
            logging.info("Setting pool output prot to FALSE for align-dist")
            args["pool_prot_strategy"] = None

    # Change this to set pool num
    if args["pool_prot_strategy"] in ["hard", "randhard"]:

        ref_seq, pool_residues = parse_utils.parse_ssa_reference(args["ssa_ref_file"])
        args["pool_num"] = len(pool_residues)

    if args["frac_train_mask"] < 0.0 or args["frac_train_mask"] > 1.0:
        raise ValueError("frac train mask must be between 0 and 1")

    return args
