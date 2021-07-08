""" Module to train a model on splits of a dense grid

  Train on small, dense CSV grids 

"""
import copy
import sys
import logging
import time
import random
import json
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
from imp import reload
import optuna
from typing import Optional, Callable
import pickle

from sklearn import preprocessing
from enzpred.features import build_features, feature_selection

from enzpred.models import dense_models
from enzpred.utils import file_utils
from enzpred.dataset import dataloader, splitter
from enzpred import parsing
from enzpred.evaluation import metrics


def run_optuna(
    kwargs: dict,
    train: dataloader.HTSData,
    prot_featurizer: build_features.GenericFeaturizer,
    chem_featurizer: build_features.GenericFeaturizer,
):
    """run_optuna.

    Run optuna on dense dataset. Use the training set and kfold cross
    validation.

    Args:
        kwargs (dict): kwargs
        train (dataloader.HTSData): train
        prot_featurizer (build_features.GenericFeaturizer): prot_featurizer
        chem_featurizer (build_features.GenericFeaturizer): chem_featurizer
    """

    # Always MINIMIZE objective

    if kwargs["optuna_grid_sample"]:
        if kwargs["model"] == "knn":
            grid = {"n_neighbors": np.arange(1, 11).tolist()}
        elif kwargs["model"] == "linear":
            grid = {"logalpha": np.arange(-3, 5).tolist()}
        else:
            raise ValueError("model grid not defined")

        sampler = optuna.samplers.GridSampler(grid)
        # Get proper length
        n_trials = 1
        for i in grid.values():
            n_trials = n_trials * len(i)
    else:
        sampler = None
        n_trials = kwargs.get("optuna_trials")

    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(), direction="minimize", sampler=sampler
    )

    optuna_objective = get_optuna_fn(
        train=train,
        prot_featurizer_obj=prot_featurizer,
        chem_featurizer_obj=chem_featurizer,
        **kwargs,
    )

    study.optimize(optuna_objective, n_trials=n_trials)

    logging.info(f"Best params from optuna: {json.dumps(study.best_params, indent=2)}")

    optuna_out_name = f"{kwargs.get('out')}_optuna_params.json"

    # Update the params (and convert to correct types)
    update_params = study.best_params
    if "batch_size" in update_params:
        update_params["batch_size"] = int(update_params["batch_size"])

    if "logalpha" in update_params:
        update_params["alpha"] = np.power(10.0, update_params["logalpha"])

    # Remove the ensemble num from the update because it was set to 1 manually
    update_params.pop("deep_ensemble_num", None)
    update_params.pop("num_tasks", None)

    json.dump(update_params, open(optuna_out_name, "w"), indent=2)

    # kwargs is updated here
    kwargs.update(update_params)

    # Dump these model args again
    # Save all arguments
    out_args = kwargs.get("out") + "_args.json"

    # TODO: Save all optuna parameters, not just the last one
    file_utils.dump_json(kwargs, out_args)

    logging.info(f"Updated params: {json.dumps(kwargs, indent=1)}")
    optuna_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    optuna_df.to_csv(kwargs.get("out") + "_optuna_results.csv")


def get_optuna_fn(
    train: dataloader.BaseDataset,
    prot_featurizer_obj: build_features.GenericFeaturizer,
    chem_featurizer_obj: build_features.GenericFeaturizer,
    optuna_folds: int = 3,
    **kwargs,
) -> Callable:
    """get_optuna_fn.

    This function should execute hyperparameter optimization.

    Args:
        train (dataloader.BaseDataset): train
        prot_featurizer_obj (build_features.GenericFeaturizer): prot_featurizer
        chem_featurizer_obj (build_features.GenericFeaturizer): chem_featurizer
        optuna_folds (int): Number of folds for optuna hyperparam selection
        kwargs:
    Return:
        Hyperopt experiment function
    """

    model_type = kwargs.get("model")

    # Use neg auc roc
    optuna_objective = "rmse" if kwargs.get("regression") else "neg-avg-pr"

    # kwargs.get("optuna_objective")
    # optuna_objective = "neg-spearman" if kwargs.get("regression") else "neg-auc-roc"  #kwargs.get("optuna_objective")

    metric_fn = metrics.get_metrics(**kwargs)[optuna_objective]

    def optuna_objective(trial):
        """optuna function"""
        kwargs_copy = kwargs.copy()

        # Only optimize one at a time
        kwargs_copy["deep_ensemble_num"] = 1

        # Prune
        trial.set_user_attr("prune", False)

        if kwargs_copy["model"] == "ffn":
            kwargs_copy["model_dropout"] = trial.suggest_uniform(
                "model_dropout", 0, 0.2
            )
            kwargs_copy["weight_decay"] = trial.suggest_uniform("weight_decay", 0, 0.01)
            kwargs_copy["hidden_size"] = trial.suggest_int("hidden_size", 10, 90, 20)
            kwargs_copy["layers"] = trial.suggest_int("layers", 1, 2)
            kwargs_copy["learning_rate"] = trial.suggest_loguniform(
                "learning_rate", 0.00001, 0.001
            )
            # kwargs_copy['batch_size'] = trial.suggest_int('batch_size', 16, 64, 16)
            # kwargs_copy['use_scheduler'] = trial.suggest_categorical('no_scheduler', [True, False])

        elif kwargs_copy["model"] == "ffnsingle":
            kwargs_copy["model_dropout"] = trial.suggest_uniform(
                "model_dropout", 0, 0.2
            )
            kwargs_copy["weight_decay"] = trial.suggest_uniform("weight_decay", 0, 0.01)
            kwargs_copy["hidden_size"] = trial.suggest_int("hidden_size", 10, 100, 20)
            kwargs_copy["layers"] = trial.suggest_int("layers", 1, 2)
            kwargs_copy["learning_rate"] = trial.suggest_loguniform(
                "learning_rate", 0.00001, 0.001
            )
            # kwargs_copy['batch_size'] = trial.suggest_int('batch_size', 16, 64, 16)
            # kwargs_copy['use_scheduler'] = trial.suggest_categorical('no_scheduler', [True, False])

        elif kwargs_copy["model"] == "ffndot":
            kwargs_copy["model_dropout"] = trial.suggest_uniform(
                "model_dropout", 0, 0.2
            )
            kwargs_copy["weight_decay"] = trial.suggest_uniform("weight_decay", 0, 0.01)
            kwargs_copy["hidden_size"] = trial.suggest_int("hidden_size", 10, 100, 20)
            kwargs_copy["layers"] = trial.suggest_int("layers", 1, 2)
            kwargs_copy["learning_rate"] = trial.suggest_loguniform(
                "learning_rate", 0.00001, 0.001
            )

        elif kwargs_copy["model"] == "gp":
            pass

        elif kwargs_copy["model"] == "rf":
            kwargs_copy["max_depth"] = trial.suggest_int("max_depth", 1, 5)
            kwargs_copy["n_estimators"] = trial.suggest_int("n_estimators", 1, 1000)

        elif kwargs_copy["model"] == "knn":
            kwargs_copy["n_neighbors"] = trial.suggest_int("n_neighbors", 1, 10)

        elif kwargs_copy["model"] == "linear":
            # kwargs_copy['alpha'] = trial.suggest_loguniform('alpha', 1e-1, 1e4)
            log_alpha = trial.suggest_int("logalpha", -3, 4)
            kwargs_copy["alpha"] = np.power(10.0, log_alpha)
            # kwargs_copy['no_class_weight'] = trial.suggest_categorical("no_class_weight", [True, False])

        else:
            raise ValueError(
                f"Hyperopt not implemented for model {kwargs_copy['model']}"
            )

        # Because we have kfold, train, val , test size don't matter
        k_splitter = splitter.get_splitter_dense(
            splitter_name=kwargs["splitter_name"],
            train_size=0.8,
            val_size=0.1,
            test_size=0.1,
            num_folds=optuna_folds,
            num_kfold_trials=1,  # 20, # recently added
        )
        # max_imbalance=kwargs.get("max_imbalance"))
        test_labels = []
        test_preds = []
        test_eval_groups = []

        ### Non CPI Optuna
        if kwargs["pivot_task"] is not None:
            for fold, (split_name, train_k, val_k, test_k) in enumerate(
                k_splitter.get_splits(train, **kwargs_copy)
            ):

                # Randomly subset down num tasks to 10 max for speed
                # Only use a subset of 10 IF we are in an FFN single model
                # This change was made to accomodate the single ffn models that
                # were single task but acted on each column...
                if kwargs_copy["model"] == "ffnsingle":
                    train_k.random_label_subset(max_k=10)
                    val_k.random_label_subset(train_k.get_label_names())
                    test_k.random_label_subset(train_k.get_label_names())
                    kwargs_copy["num_tasks"] = train_k.num_tasks()

                # Run featurizer over the train subset to make this a little harder
                prot_featurizer_temp, chem_featurizer_temp = copy.copy(
                    prot_featurizer_obj
                ), copy.copy(chem_featurizer_obj)
                in_features = select_features(
                    prot_featurizer_temp,
                    chem_featurizer_temp,
                    train_k,
                    val_k,
                    test_k,
                    kwargs_copy,
                )

                # np.random.seed(kwargs_copy.get('seed'))
                # random.seed(kwargs_copy.get('seed'))

                model = dense_models.get_model(
                    model_type, in_features=in_features, **kwargs_copy
                )

                model.train(train_k, val_k, trial=trial, **kwargs_copy)
                test_predictions, _ = model.predict(test_k, **kwargs_copy)

                test_eval_groups.extend(
                    [split_name for _ in np.arange(len(test_predictions))]
                )
                test_preds.append(test_predictions)  # .reshape(-1,1))
                test_labels.append(test_k.get_labels())  # .reshape(-1,1))

            # Only compute metric once for ALL predictions
            test_preds = np.vstack(test_preds)
            test_labels = np.vstack(test_labels)
            test_eval_groups = np.array(test_eval_groups)

            metric_res = []
            for eval_group in np.unique(test_eval_groups):
                test_labels_temp = test_labels[test_eval_groups == eval_group]
                test_preds_temp = test_preds[test_eval_groups == eval_group]

                for col in range(test_preds.shape[1]):
                    metric_output = metric_fn(
                        test_labels_temp[:, col], test_preds_temp[:, col]
                    )
                    if metric_output is None:
                        raise ValueError()
                    metric_res.append(metric_output)

            metric_output = np.mean(metric_res)

        ### CPI Optuna
        else:
            grouping_list = []
            for fold, (split_name, train_k, val_k, test_k) in enumerate(
                k_splitter.get_splits(train, **kwargs_copy)
            ):

                # Run featurizer over the train subset to make this a little harder
                prot_featurizer_temp, chem_featurizer_temp = copy.copy(
                    prot_featurizer_obj
                ), copy.copy(chem_featurizer_obj)
                in_features = select_features(
                    prot_featurizer_temp,
                    chem_featurizer_temp,
                    train_k,
                    val_k,
                    test_k,
                    kwargs_copy,
                )

                np.random.seed(kwargs_copy.get("seed"))
                random.seed(kwargs_copy.get("seed"))

                model = dense_models.get_model(
                    model_type, in_features=in_features, **kwargs_copy
                )

                model.train(train_k, val_k, trial=trial, **kwargs_copy)
                test_predictions, _ = model.predict(test_k, **kwargs_copy)

                test_preds.append(test_predictions)
                test_labels.append(test_k.get_labels())

                eval_labels = test_k.data[kwargs["eval_grouping"]].values.tolist()
                grouping_list.extend(eval_labels)

            # Only compute metric once for ALL predictions
            test_preds = np.vstack(test_preds)
            test_labels = np.vstack(test_labels)

            # Find all eval groups
            grouping_list = np.array(grouping_list)
            uniq_groups = np.unique(np.sort(grouping_list))

            metric_res = []
            for group in uniq_groups:
                bool_select = grouping_list == group
                test_preds_temp = test_preds[bool_select]
                test_labels_temp = test_labels[bool_select]

                # Skipping column if it's too imbalanced
                to_skip = dataloader.skip_col(test_labels_temp, **kwargs)
                if to_skip:
                    continue

                metric_output = metric_fn(test_labels_temp, test_preds_temp)
                if metric_output is None:
                    raise ValueError()

                metric_res.append(metric_output)

            metric_output = np.mean(metric_res)

        return metric_output

    return optuna_objective


def save_model(model, featurizer_dict, kwargs):
    """Save model"""
    model.export_state(kwargs.get("out") + "_model_state.p", **kwargs)
    # For fit featurizers, we should export them
    feat_dict = {}
    for featurizer_name, featurizer in featurizer_dict.items():
        if featurizer is not None and featurizer.is_fit:
            feat_dict[featurizer_name] = featurizer
    pickle.dump(feat_dict, open(kwargs.get("out") + "_fit_featurizers.p", "wb"))


def select_features(
    prot_featurizer_obj: build_features.GenericFeaturizer,
    chem_featurizer_obj: build_features.GenericFeaturizer,
    train: dataloader.BaseDataset,
    val: dataloader.BaseDataset,
    test: dataloader.BaseDataset,
    kwargs: dict,
):
    """select_features.

    Select features and change outdimension.
    Return the dict of in_features for use in rest of pipeline


    Args:
        prot_feautrizer_obj (GenericFeaturizer) : prot_featurizer_obj
        chem_featurizer_obj (GenericFeaturizer) : chem_featurizer_obj
        train (dataloader.BaseDataset): train
        val (dataloader.BaseDataset): val
        test (dataloader.BaseDataset): test
        kwargs (dict): kwargs
    """

    featurizers = [prot_featurizer_obj, chem_featurizer_obj]
    featurizer_names = ["prot_featurizer", "chem_featurizer"]

    in_features = {
        name: j.out_dim() for j, name in zip(featurizers, featurizer_names) if j
    }
    # Need to augment
    # If we concat an rdkit featurizer
    if kwargs.get("concat_rdkit", False):
        in_features[
            "chem_featurizer_global"
        ] = chem_featurizer.rdkit_featurizer.out_dim()

    selectors = []
    selector_names = [kwargs.get("prot_selector"), kwargs.get("chem_selector")]
    col_names = ["prot", "chem"]
    for featurizer_name, featurizer, col, selector_name in zip(
        featurizer_names, featurizers, col_names, selector_names
    ):
        if featurizer is not None and featurizer.out_dim()[0] is not None:
            selector = feature_selection.get_feature_selection(selector_name, **kwargs)
            selectors.append(selector)
            new_outdim = train.select_features(col, selector, train=True, **kwargs)
            val.select_features(col, selector, train=False, **kwargs)
            test.select_features(col, selector, train=False, **kwargs)

            # Update outdim
            if new_outdim is not None:
                featurizer.set_out_dim(new_outdim)
                logging.info(f"Setting featurizer {featurizer} outdim to {new_outdim}")
                in_features[featurizer_name] = featurizer.out_dim()
        elif featurizer is not None and featurizer.out_dim()[0] is None:
            logging.warning(
                f"""Trying to run feature selection over non fixed dim featurizer {featurizer}"""
            )
        else:
            logging.info(f"No feature selection for featurizer {featurizer_name}")

    return in_features


def scale_features(
    prot_scaler: Optional[preprocessing.StandardScaler],
    chem_scaler: Optional[preprocessing.StandardScaler],
    train: dataloader.BaseDataset,
    val: dataloader.BaseDataset,
    test: dataloader.BaseDataset,
    kwargs: dict,
):
    """scale_features.

    Args:
        prot_scaler (Optional[preprocessing.StandardScaler]): prot_scaler
        chem_scaler (Optional[preprocessing.StandardScaler]): chem_scaler
        train (dataloader.BaseDataset): train
        val (dataloader.BaseDataset): val
        test (dataloader.BaseDataset): test
        kwargs (dict): kwargs
    """
    col_names = ["prot", "chem"]
    scalers = [prot_scaler, chem_scaler]

    # Fit on both train and val
    # Consider removing eventually
    joint_train_val = dataloader.HTSData.fuse_hts(train, val, **kwargs)
    for scaler, col in zip(scalers, col_names):
        if scaler is not None:
            joint_train_val.scale_features(col, scaler, train=True, **kwargs)
            train.scale_features(col, scaler, train=False, **kwargs)
            val.scale_features(col, scaler, train=False, **kwargs)
            test.scale_features(col, scaler, train=False, **kwargs)


def scale_targets(dataset, scalers: Optional[list] = None, kwargs: dict = {}):
    """scale_features.

    Args:
        dataset (dataloader.BaseDataset): train
        scalers (Optional[list]): List of scalers to use; if None, create them
            and fit scalers!
        kwargs (dict): kwargs
    Return:
        list[featurizers]: List of featurizers for each label
    """

    targ_cols = dataset.get_label_names()
    train = False

    # If scalers not passed, define and fit them
    if scalers is None:
        scalers = [preprocessing.StandardScaler() for i in targ_cols]
        train = True

    for col, scaler in zip(dataset.get_label_names(), scalers):
        dataset.scale_targets(col, scaler, train=train)

    return scalers


def run_training(kwargs: dict):
    """run_training.

    Args:
        kwargs (dict): kwargs
    """

    # Build featurizers
    with file_utils.Stage("BUILD FEATURIZERS"):
        chem_featurizer = build_features.get_chem_featurizer(**kwargs)
        prot_featurizer = build_features.get_prot_featurizer(**kwargs)

    # Build dataloader
    with file_utils.Stage("BUILDING DATASET"):

        # Get regression dataset
        data = pd.read_csv(kwargs.get("hts_csv_file"), index_col=0)

        ######## DEBUG
        # valid_prots = list(pd.unique(data['SEQ']))[:10]
        # data = data[np.array([i in valid_prots for i in data["SEQ"]])]
        ## NOW TRUNCATE LEN OF PROTEINS
        # data["SEQ"] = [i[:50] for i in data["SEQ"]]

        if (
            kwargs["debug_mode"]
            and "phosphatase" in kwargs["hts_csv_file"]
            and kwargs["splitter_name"] == "single-sub"
        ):
            truth_val = data["SUBSTRATES"] == "CC(=O)OP(=O)(O)O"
            data = data[truth_val].reset_index(drop=True)

        ## Filter substrates before they get pivoted if not in a category
        data = dataloader.filter_substrates(data, **kwargs)

        # Pivot the task
        # pivot_task options: SEQ, SUBSTRATES, default None
        pivot_task = kwargs.get("pivot_task", None)
        if pivot_task is not None:
            data = dataloader.pivot_task_df(data, skip_cols=True, **kwargs)

        full_dataset = dataloader.get_dataset(data=data, **kwargs)

        # Set num task for building models later
        kwargs["num_tasks"] = full_dataset.num_tasks()

    # Featurize data in data loaders
    with file_utils.Stage("FEATURIZING DATA"):

        # embed_ref_obj = full_dataset.get_ref_obj(prot_featurizer_ = prot_featurizer, **kwargs)
        embed_ref_obj = None  # this will break some things, particularly RSSA
        # in distance, oh well
        full_dataset.featurize_data(
            chem_featurizer_obj=chem_featurizer,
            prot_featurizer_obj=prot_featurizer,
            **kwargs,
        )

    ## TODO: Implement a global split option
    if kwargs["optuna_global"] and kwargs["run_optuna"]:

        with file_utils.Stage("Global optuna tuning"):
            global_copy = dataloader.HTSData.fuse_hts(full_dataset, **kwargs)

            if kwargs["scale_prot"] or kwargs["scale_chem"]:
                # Create copies of featurizers because this is an inner loop
                scale_prot = (
                    prot_featurizer
                    and kwargs["scale_prot"]
                    and prot_featurizer.out_dim()[0] is not None
                )
                scale_chem = (
                    chem_featurizer
                    and kwargs["scale_chem"]
                    and chem_featurizer.out_dim()[0] is not None
                )

                prot_scaler = preprocessing.StandardScaler() if scale_prot else None
                chem_scaler = preprocessing.StandardScaler() if scale_chem else None

                if prot_scaler:
                    global_copy.scale_features(
                        "prot", prot_scaler, train=True, **kwargs
                    )
                if chem_scaler:
                    global_copy.scale_features(
                        "chem", chem_scaler, train=True, **kwargs
                    )

            # Scale targets
            if kwargs.get("regression"):
                scale_targets(global_copy, scalers=None, kwargs=kwargs)

            # run optuna
            run_optuna(kwargs, global_copy, prot_featurizer, chem_featurizer)

    with file_utils.Stage("RUNNING SPLITS"):
        data_splitter = splitter.get_splitter_dense(**kwargs)
        # data_outputs = []
        # metric_names = set()
        model_predictions = []
        for fold_count, (split_name, train, val, test) in enumerate(
            data_splitter.get_splits(full_dataset, **kwargs)
        ):

            with file_utils.Stage("FEATURE SCALING"):
                if kwargs["scale_prot"] or kwargs["scale_chem"]:
                    # Create copies of featurizers because this is an inner loop
                    scale_prot = (
                        prot_featurizer
                        and kwargs["scale_prot"]
                        and prot_featurizer.out_dim()[0] is not None
                    )
                    scale_chem = (
                        chem_featurizer
                        and kwargs["scale_chem"]
                        and chem_featurizer.out_dim()[0] is not None
                    )

                    prot_scaler = preprocessing.StandardScaler() if scale_prot else None
                    chem_scaler = preprocessing.StandardScaler() if scale_chem else None

                    ### AHH! It's variance in how we build our scaler..
                    scale_features(prot_scaler, chem_scaler, train, val, test, kwargs)
                else:
                    logging.info("Not scaling features")

            with file_utils.Stage("SCALING TARGETS"):
                # Scale train and val
                if kwargs.get("regression"):
                    joint_train_val = dataloader.HTSData.fuse_hts(train, val, **kwargs)

                    # Scale train + val together
                    scaler_set = scale_targets(
                        joint_train_val, scalers=None, kwargs=kwargs
                    )
                    scaler_set = scale_targets(train, scalers=scaler_set, kwargs=kwargs)
                    scaler_set = scale_targets(val, scalers=scaler_set, kwargs=kwargs)

            with file_utils.Stage("HYPERPARAM OPTIMIZATION"):
                """Run optuna selection on the data split, just to get _model_ hyperparams"""
                if kwargs["run_optuna"] and not kwargs["optuna_global"]:
                    ## Optuna settings here.
                    joint_train_val = dataloader.HTSData.fuse_hts(train, val, **kwargs)

                    run_optuna(
                        kwargs, joint_train_val, prot_featurizer, chem_featurizer
                    )
                elif not kwargs["run_optuna"]:
                    logging.info(f"No hyperparam tuning for this model")

            # Select features based on train
            with file_utils.Stage("FEATURE SELECTION"):
                # Create copies of featurizers because this is an inner loop
                prot_featurizer_temp, chem_featurizer_temp = copy.copy(
                    prot_featurizer
                ), copy.copy(chem_featurizer)
                in_features = select_features(
                    prot_featurizer_temp, chem_featurizer_temp, train, val, test, kwargs
                )

            # Build model
            with file_utils.Stage("BUILD MODEL"):
                model = dense_models.get_model(
                    kwargs.get("model"),
                    in_features=in_features,
                    embed_ref_obj=embed_ref_obj,
                    **kwargs,
                )

            # Train model
            with file_utils.Stage("TRAINING"):

                # Run experiments to mask part of the training mask
                if kwargs["frac_train_mask"] > 0.0:
                    frac_na = kwargs["frac_train_mask"]
                    na_mask = np.random.choice(
                        a=[0, 1],
                        p=[1 - frac_na, frac_na],
                        size=train.get_labels().shape,
                    ).astype(bool)
                    new_targs = train.get_labels().astype(float)
                    new_targs[na_mask] = np.nan
                    train.data[train.get_label_names()] = new_targs

                model.train(train, val, **kwargs)

            # Evaluation
            with file_utils.Stage("EVALUATION"):
                metrics_store = defaultdict(
                    lambda: dict()
                )  # lambda : defaultdict(lambda : dict()))

                data_list = [train]
                names_list = ["train"]

                if len(val) > 0:
                    data_list.append(val)
                    names_list.append("val")
                if len(test) > 0:
                    data_list.append(test)
                    names_list.append("test")

                if kwargs.get("ignore_train", False):
                    data_list, names_list = [test], ["test"]

                # For each dataset split
                for data_, name_ in zip(data_list, names_list):
                    # For ALL the data
                    pred_, aux_ = model.predict(data_, return_pairwise=False, **kwargs)

                    if len(pred_.shape) == 1:
                        pred_ = pred_.reshape(-1, 1)

                    data_labels = data_.get_labels()
                    label_names = data_.get_label_names()

                    ### Unscale targets
                    if kwargs.get("regression"):
                        # Inverse scale predictions
                        for scaler, column in zip(
                            scaler_set, range(data_labels.shape[1])
                        ):
                            col_preds = pred_[:, column].reshape(-1, 1)

                            # Reset data labels
                            pred_[:, column] = scaler.inverse_transform(
                                col_preds
                            ).flatten()

                        # Inverse scale targets too
                        if name_ in ["train", "val"]:
                            for scaler, column in zip(
                                scaler_set, range(data_labels.shape[1])
                            ):
                                data_col = data_labels[:, column].reshape(-1, 1)

                                # Reset data labels
                                data_labels[:, column] = scaler.inverse_transform(
                                    data_col
                                ).flatten()

                    # Loop over all preidctions
                    # Compute labels based on the task
                    for output_entry, true_vals, (index, row) in zip(
                        pred_, data_labels, data_.data.iterrows()
                    ):
                        for pred_value, true_value, label in zip(
                            output_entry, true_vals, label_names
                        ):
                            # true_value = row[label]
                            predict_entry = {
                                "SUBSTRATES": row["SUBSTRATES"],
                                "SEQ": row["SEQ"],
                                "label": label,
                                "val": true_value,
                                "pred": pred_value,
                                "split": name_,
                                "split_name": split_name,
                            }
                            # Unpivot the label if it's pivoted
                            # Then modify the label such that it matches
                            # the desired eval grouping
                            update_info = {}
                            update_info = dataloader.unpivot_label(
                                label, pivot_task=kwargs.get("pivot_task", None)
                            )
                            predict_entry.update(update_info)
                            eval_grouping = kwargs.get("eval_grouping")
                            if eval_grouping != "ALL":
                                eval_cat = predict_entry.get(eval_grouping)
                                cur_label = predict_entry.get("label")
                                new_label = f"{cur_label}_{eval_cat}"
                                predict_entry["label"] = new_label

                            model_predictions.append(predict_entry)

    with file_utils.Stage("COMPUTE METRICS"):
        """After getting outputs of the model, let's process the data"""
        # Loop over the data frame that has results
        df = pd.DataFrame(model_predictions)

        data_outputs = []
        metric_names = set()
        # Loop over train, val, test predictions
        for train_val_test_split, train_test_df in df.groupby("split"):

            # Loop over different folds of the data
            for fold_name, fold_df in train_test_df.groupby("split_name"):
                entry = {}
                entry.update(kwargs)
                entry.update({"dataset_split": train_val_test_split})
                entry.update({"split_name": fold_name})

                # Loop over all the tasks (labels)
                # Each entry should have values for every task
                # Design: each label has its own value, but there's overall a
                # summary that is an average across the different tasks...
                metrics_store = dict()
                # Try each different metric function on this label
                for metric_name, metric_fn in metrics.get_metrics(**kwargs).items():
                    metric_avg = []
                    for label, sub_df_label in fold_df.groupby("label"):

                        predictions = sub_df_label["pred"].values
                        values = sub_df_label["val"].values

                        # If we're creating alternative groupings,
                        # Then we should not evaluate this on columns that
                        # should be skipped according to other splits
                        if eval_grouping != "ALL":
                            to_skip = dataloader.skip_col(values, **kwargs)
                            if to_skip:
                                continue

                        # Only compute where it isn't na
                        not_na = ~np.isnan(values)
                        metric_name_ = f"{metric_name}_{label}"
                        entry[metric_name_] = metric_fn(
                            values[not_na], predictions[not_na]
                        )
                        metric_avg.append(entry[metric_name_])

                    # Compute an average over all labels/tasks!
                    metric_names.add(metric_name)
                    metric_avg = np.array(metric_avg)
                    is_none = np.array([i is None for i in metric_avg])
                    if np.any(is_none):
                        logging.info(
                            f"WARNING: Found None values for {metric_name} in fold {fold_name}"
                        )
                    entry[metric_name] = np.mean(metric_avg[~is_none])

                # In addition to grouping over label and computing all metrics,
                # let's compute some metrics ACROSS all labels
                # We can pivot two df's; one for pred, one for val, and just
                # subtract them

                # pred_frame = train_test_df.pivot_table(values="pred",
                #                                       columns="label",
                #                                       index="SEQ").values
                # targ_frame = train_test_df.pivot_table(values="val",
                #                                       columns="label",
                #                                       index="SEQ").values

                ## across whole df
                # avg_hamming = np.abs(pred_frame - targ_frame).mean()
                # avg_hamming_row_first = np.abs(pred_frame -
                #                               targ_frame).mean(1).mean(0)

                data_outputs.append(entry)

    with file_utils.Stage("EXPORT RESULTS"):
        # Export data
        outfile = kwargs.get("out") + "_output.csv"

        # Change setting for export!
        pd.set_option("display.max_columns", None)

        # model.export_model_stats(kwargs.get('out'))
        df = pd.DataFrame(data_outputs)

        summary_metrics = df.groupby("dataset_split")[list(metric_names)].mean()
        summary_metrics = summary_metrics.reindex(
            sorted(summary_metrics.columns), axis=1
        )
        logging.info(f"Final results averaged over {fold_count + 1} folds: ")

        # Average across tasks
        summary_metrics = summary_metrics.groupby(
            by=lambda x: x, axis=1  # lambda x: x.rsplit("_", 1)[0],
        ).mean()
        logging.info(f"\n {summary_metrics}")
        df.to_csv(outfile)
        datapoints_out = f"{kwargs.get('out')}_preds.json"

        # Convert away from np float types
        for j in model_predictions:
            j["pred"] = float(j["pred"])
            j["val"] = float(j["val"])

        file_utils.dump_json(model_predictions, datapoints_out, pretty_print=False)

        export_save = f"{kwargs.get('out')}_exported"
        model.export_model_stats(export_save)

    with file_utils.Stage("SAVE MODEL"):
        # Save the last model
        featurizer_dict = {
            "chem_featurizer": chem_featurizer,
            "prot_featurizer": prot_featurizer,
        }
        save_model(model, featurizer_dict, kwargs)

        # Dump args again...
        # Save all arguments
        out_args = kwargs.get("out") + "_args.json"
        file_utils.dump_json(kwargs, out_args)


def main():
    """Main method"""
    start_time = time.time()
    parser = parsing.get_dense_parser()
    kwargs = parser.parse_args()

    kwargs = kwargs.__dict__
    if kwargs["model_params_file"]:
        logging.info(f"Loaded model params from {kwargs['model_params_file']}")
        loaded_params = json.load(open(kwargs["model_params_file"], "r"))
        kwargs.update(loaded_params)

    # Make directory
    file_utils.make_dir(kwargs.get("out"))

    # Logging not configured
    # Import it in case some random module set the logger
    reload(logging)
    if kwargs["debug_mode"]:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO
    logging_level = logging.DEBUG
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(kwargs.get("out") + "_run.log"),
        ],
    )

    # modify all bad or conflicting arguments
    kwargs = parsing.modify_parser_dense(kwargs)

    logging.info(f"Parsed args: {json.dumps(kwargs, indent=2)}")

    np.random.seed(kwargs.get("seed"))
    random.seed(kwargs.get("seed"))
    if kwargs.get("seed") is not None:
        torch.manual_seed(kwargs.get("seed"))

    # Save all arguments
    out_args = kwargs.get("out") + "_args.json"
    file_utils.dump_json(kwargs, out_args)
    run_training(kwargs)
    logging.info(f"Wall time for program: {time.time() - start_time : .2f} seconds")
