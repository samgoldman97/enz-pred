"""torch_models.
Module to hold all pytorch models we want and their training functions
"""

import logging
import time
from typing import Callable, Optional, List, Tuple, Dict, Union
from collections import defaultdict
import copy
from tqdm import tqdm

import torch
from torch import optim
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import optuna
from optuna.trial import Trial

from enzpred.dataset import dataloader

MODEL_ARGS = [
    (
        ["--layers"],
        dict(action="store", type=int, default=3, help="Number of layers ot use"),
    ),
    (
        ["--hidden-size"],
        dict(action="store", type=int, default=50, help="Hidden size to use"),
    ),
    (
        ["--model-dropout"],
        dict(
            action="store", type=float, default=0.1, help="Dropout rate for ffn model."
        ),
    ),
    (
        ["--use-scheduler"],
        dict(action="store_true", default=False, help="If set, use NOAM lr scheduler"),
    ),
    (
        ["--warmup-epochs"],
        dict(
            action="store",
            type=int,
            default=1,
            help="How many warmup epochs for noam scheduler",
        ),
    ),
    (
        ["--kernel-size"],
        dict(
            action="store",
            type=int,
            default=5,
            help="Size of kernels for protein convs",
        ),
    ),
    (
        ["--avg-pool-conv"],
        dict(
            action="store_true",
            default=False,
            help="If true, use avg pool for protein conv",
        ),
    ),
    (
        ["--num-conv-layers"],
        dict(
            action="store",
            type=int,
            default=3,
            help="Number of protein conv layers to apply",
        ),
    ),
    (
        ["--batches-per-eval"],
        dict(
            action="store",
            type=int,
            default=None,
            help="Check val loss after every x batches",
        ),
    ),
    (
        ["--weight-decay"],
        dict(
            action="store", type=float, default=0, help="Weight decay for torch optim"
        ),
    ),
]

MAIN_LOSS = "Reaction outcome loss"
TOTAL_LOSS = "Total loss"


class NoamLR(optim.lr_scheduler._LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.
    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where warmup_steps = warmup_epochs * steps_per_epoch).
    Then the learning rate decreases exponentially from max_lr to final_lr over the
    course of the remaining total_steps - warmup_steps (where total_steps =
    total_epochs * steps_per_epoch). This is roughly based on the learning rate
    schedule from Attention is All You Need, section 5.3 (https://arxiv.org/abs/1706.03762).
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: List[Union[float, int]],
        total_epochs: List[int],
        steps_per_epoch: int,
        init_lr: List[float],
        max_lr: List[float],
        final_lr: List[float],
        **kwargs,
    ):
        """
        Initializes the learning rate scheduler.
        Args:
            optimizer: A PyTorch optimizer.
            warmup_epochs (List[Union[float, int]]): The number of epochs during which to linearly increase the learning rate.
            total_epochs (List[int]): The total number of epochs.
            steps_per_epoch (int): The number of steps (batches) per epoch.
            init_lr (List[float]): The initial learning rate.
            max_lr (List[float]): The maximum learning rate (achieved after warmup_epochs).
            final_lr (List[float]): The final learning rate (achieved after total_epochs).
            kwargs :
        """
        assert (
            len(optimizer.param_groups)
            == len(warmup_epochs)
            == len(total_epochs)
            == len(init_lr)
            == len(max_lr)
            == len(final_lr)
        )

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (
            1 / (self.total_steps - self.warmup_steps)
        )

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """Gets a list of the current learning rates."""
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.
        Args:
            current_step (int): Optionally specify what step to set the learning rate to.
                If None, current_step = self.current_step + 1.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = (
                    self.init_lr[i] + self.current_step * self.linear_increment[i]
                )
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (
                    self.exponential_gamma[i]
                    ** (self.current_step - self.warmup_steps[i])
                )
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]["lr"] = self.lr[i]


def build_lr_scheduler(
    optimizer: optim.Optimizer,
    warmup_epochs: int,
    epochs: int,
    batch_size: int,
    train_data_size: int,
    learning_rate: float,
    **kwargs,
) -> optim.lr_scheduler._LRScheduler:
    """
    Builds a learning rate scheduler.
    Args:
        optimizer: The Optimizer whose learning rate will be scheduled.
        warmup_epochs (int): Warmup epochs
        epochs (int): Num epochs
        batch_size (int): Size of batches in train
        train_data_size: The size of the training dataset.
        learning_rate (float) :
        kwargs: Arguments.
    Return:
        An initialized learning rate scheduler.
    """
    # Learning rate scheduler
    # print(train_data_size, batch_size, train_data_size//batch_size)
    return NoamLR(
        optimizer=optimizer,
        warmup_epochs=[warmup_epochs],
        steps_per_epoch=train_data_size // batch_size,
        total_epochs=[epochs],
        init_lr=[learning_rate / 10],
        max_lr=[learning_rate],
        final_lr=[learning_rate / 100],
        **kwargs,
    )


def get_predictions(
    model_: nn.Module,
    test_dataloader: DataLoader,
    gpu: bool = False,
    regression: bool = False,
    **kwargs,
) -> Tuple[np.array, dict]:
    """get_predictions.

    Args:
        model_ (nn.Module): model_
        test_dataloader (DataLoader): test_dataloader
        gpu (bool): gpu
        regression (bool): If true, use regression formulation
        kwargs:

    Returns:
        Tuple[np.array, dict]:
    """

    model = model_
    full_outputs = []
    # Set model to eval
    model.eval()
    if gpu:
        model = model.cuda()
        model.set_gpu(gpu)

    aux_outputs = defaultdict(lambda: [])
    with torch.no_grad():
        for batch in test_dataloader:

            outputs, aux_preds = model(batch)

            # Reformat auxilary predictions into a more convenient list for
            # data export
            if regression:
                outputs = outputs.cpu().numpy()
            else:
                outputs = torch.nn.functional.sigmoid(outputs).cpu().numpy()

            full_outputs.append(outputs)

    outs = np.concatenate(full_outputs, axis=0)
    return outs, dict(aux_outputs)


def append_loss_to_list(loss_list: Dict[str, List], dict_loss: dict):
    """
    Helper fn to add dict_loss of most recent losses into loss_list.
    """
    for k, v in dict_loss.items():
        loss_list[k].append(v)


def avg_loss_list(loss_list: Dict[str, List]) -> dict:
    """
    Helper fn to add dict_loss of most recent losses into loss_list.
    """
    out = {}
    for k, v in loss_list.items():
        out[k] = np.mean(v)
    return out


def train_model(
    model_: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    gpu: bool,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    warmup_epochs: int,
    use_scheduler: bool,
    trial: Trial = None,
    batches_per_eval: Optional[int] = None,
    weight_decay: float = 0.0,
    regression: bool = False,
    **kwargs,
):
    """train_model.

    Args:
        model_ (nn.Module): model_
        train_dataloader (DataLoader): train_dataloader
        val_dataloader (DataLoader): val_dataloader
        gpu (bool): gpu
        batch_size (int): batch_size
        epochs (int): epochs
        learning_rate (float): learning_rate
        warmup_epochs (int): Num of warm up epochs
        use_scheduler (bool): If true, use a scheduler
        trial (Trial): Optuna trial object ued for early stopping and
            hyperparam opt
        batches_per_eval (Optional[int]) : Number of batches between evaluating full val
            loss
        weight_decay (float): amount of weight decay for optim
        regression (bool) : If true, run regression loss
        kwargs:
    """

    # Define optuna parsing fn
    optuna_step = 0
    if trial and trial.user_attrs["prune"]:
        prune_mod = len(train_dataloader) // trial.user_attrs["prune_freq"]

    def run_optuna(model_, val_dataloader_, gpu_, trial_, optuna_step, **kwargs):
        val_predictions, _ = get_predictions(
            model_, val_dataloader_, gpu_, get_aux_preds=False, **kwargs
        )
        metric_out = trial_.user_attrs["metric_fn"](
            val_dataloader_.dataset.get_labels(), val_predictions
        )
        trial_.report(metric_out, optuna_step)
        if trial_.should_prune():
            raise optuna.TrialPruned()

    model = model_
    train_size = len(train_dataloader) * batch_size

    if gpu:
        model = model.cuda()
        model.set_gpu(gpu)

    opt = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    scheduler = None
    if use_scheduler:
        scheduler = build_lr_scheduler(
            opt, warmup_epochs, epochs, batch_size, train_size, learning_rate, **kwargs
        )
    if regression:
        loss_fn = torch.nn.MSELoss(reduction="none")
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    train_losses = defaultdict(lambda: [])
    val_losses = defaultdict(lambda: [])

    # Calculate val loss before everything starts
    val_losses_0 = get_val_loss(model, loss_fn, val_dataloader, gpu)
    append_loss_to_list(val_losses, val_losses_0)

    logging.info(f"Val loss before train {val_losses_0}")

    best_loss = None
    best_model = None
    start = time.time()
    end = time.time()

    for epoch in range(epochs):
        # print(f"Time for epoch: {time.time() - start}")
        start = time.time()
        # print(f"Time for epoch: {start - end}")

        cur_lr = [obj["lr"] for obj in opt.param_groups]
        logging.info(f"Current learning rate {cur_lr}")
        model.train()
        logging.info(f"Epoch: {epoch}")
        epoch_loss = defaultdict(lambda: [])
        batch_start = 0
        for batch_index, batch in tqdm(enumerate(train_dataloader)):

            batch_loss = {}
            opt.zero_grad()
            y_pred, aux_preds = model(batch)

            y_true = batch["labels"]
            total_loss = torch.zeros(1)

            if gpu:
                y_true = y_true.cuda()
                total_loss = total_loss.cuda()

            # Calculate main loss
            numeric_index = ~torch.isnan(y_true)
            loss_main = loss_fn(y_pred[numeric_index], y_true[numeric_index])
            loss_main = loss_main.mean()

            # TEMP
            batch_loss[MAIN_LOSS] = loss_main.item()
            total_loss += loss_main

            ## Potentially add auxilary losses here
            total_loss.backward()
            opt.step()

            # Add total loss to results
            batch_loss[TOTAL_LOSS] = total_loss.item()

            append_loss_to_list(train_losses, batch_loss)
            append_loss_to_list(epoch_loss, batch_loss)

            # Step the scheduler
            if scheduler:
                scheduler.step()

            # If we have a trial object, if it's supposed to be pruned,
            # if this isn't the first batch, and if this is a multiple
            # Also don't execute last prune trial; save for after epoch
            # logging.info(f"TRIAL {trial}")
            # logging.info(f"TRIAL PRUNE {trial.user_attrs['prune']}")
            # logging.info(f"BATCH INDEX {batch_index}")
            # logging.info(f"PRUNE MOD {prune_mod}")
            # logging.info(f"Len train loader {len(train_dataloader)}")
            if (
                trial
                and trial.user_attrs["prune"]
                and batch_index > 0
                and batch_index % prune_mod == 0
                and batch_index + prune_mod < len(train_dataloader)
            ):
                run_optuna(model, val_dataloader, gpu, trial, optuna_step, **kwargs)
                logging.info(f"Done with Optuna prune step {optuna_step}")
                optuna_step += 1
                model.train()

            # Calculate val loss in batches
            if batches_per_eval is not None and batch_index % batches_per_eval == 0:
                val_loss = get_val_loss(model, loss_fn, val_dataloader, gpu)
                val_losses.append(val_loss)
                model.train()

        val_loss = get_val_loss(model, loss_fn, val_dataloader, gpu)
        append_loss_to_list(val_losses, val_loss)
        logging.info(
            f"After epoch {epoch}, val loss {val_loss} | train loss {avg_loss_list(epoch_loss)}"
        )

        # Select based on TOTAL loss
        if not best_loss or val_loss[TOTAL_LOSS] < best_loss:
            logging.info(f"Found new best model at epoch {epoch}")
            best_loss = val_loss[TOTAL_LOSS]
            best_epoch = epoch
            best_model = copy.deepcopy(model)

        # Run optuna pruning
        if trial and trial.user_attrs["prune"]:
            run_optuna(model, val_dataloader, gpu, trial, optuna_step, **kwargs)
            optuna_step += 1
        end = time.time()

    logging.info(f"Best model found after epoch {best_epoch + 1} of {epochs}.")

    results_dict = {
        "best_model": best_model,
        "train_losses": dict(train_losses),
        "val_losses": dict(val_losses),
    }
    return results_dict


def embed_dataset(model_: nn.Module, dataloader: DataLoader, gpu: bool, **kwargs):
    """embed_dataset.

    For a metric embedder, embed the entire dataset. Return an NP array

    Args:
        model_ (nn.Module): Neural model
        dataloader (DataLoader): Dataloader object
        gpu (bool): If true, use a gpu
    """
    model = model_

    if gpu:
        model = model.cuda()
        model.set_gpu(gpu)

    results = []
    aux_list = []
    with torch.no_grad():
        for batch in dataloader:
            model.eval()
            embeddings, aux = model(batch, get_aux=True)
            embeddings = embeddings.cpu().numpy()
            results.append(embeddings)
            aux_list.append(aux)

    # Collect aux
    pool_weights = [
        i for batch_aux in aux_list for i in batch_aux.get("pool_weight", [])
    ]

    return np.vstack(results), pool_weights


def get_val_loss(
    model_: nn.Module, loss_fn: Callable, val_loader: DataLoader, gpu: bool, **kwargs
) -> float:
    """get_val_loss.

    Args:
        model (nn.Module): model
        loss_fn (Callable): loss_fn
        val_loader (DataLoader): val_loader
        gpu (bool) : if true, use gpu

    Returns:
        float:
    """
    model = model_
    model.eval()

    if gpu:
        model = model.cuda()
        model.set_gpu(gpu)

    with torch.no_grad():
        epoch_loss = defaultdict(lambda: [])
        for batch in val_loader:
            total_loss = 0
            batch_loss = {}
            # Note: always convert to cpu for this because we iterate through
            # dataset
            y_pred, aux_preds = model(batch)
            y_true = batch["labels"]
            if gpu:
                y_true = y_true.cuda()

            # Calculate main loss
            numeric_index = ~torch.isnan(y_true)
            loss_main = loss_fn(y_pred[numeric_index], y_true[numeric_index])

            loss_main = loss_main.mean().item()
            batch_loss[MAIN_LOSS] = loss_main

            total_loss = sum(batch_loss.values())
            batch_loss[TOTAL_LOSS] = total_loss
            append_loss_to_list(epoch_loss, batch_loss)

    avged_loss = avg_loss_list(epoch_loss)
    return avged_loss


def get_dataloader(
    dataset: dataloader.BaseDataset,
    shuffle=False,
    batch_size: int = 32,
    train_mode: bool = True,
    single_batch: bool = False,
    regression: bool = False,
    **kwargs,
) -> DataLoader:
    """get_dataloader.

    Args:
        dataset (dataloader.BaseDataset): dataset
        shuffle (bool) : shuffle
        batch_size (int): Size of batch
        train_mode (bool): If false don't try to include labels
        single_batch (bool): If true, ouptut only one batch
        kwargs:
    Return: DataLoader
    """

    rxn_collate = lambda x: dataloader.rxn_collate(x, train_mode=train_mode)
    # If we want only 1 batch, do this
    if single_batch:
        batch_size = len(dataset)

    if regression or (not train_mode):
        sampler = None
    else:

        label_vec = dataset.get_labels()
        weights = np.ones(label_vec.shape)

        # Set NAN weight to 0
        weights[np.isnan(label_vec)] = 0.0

        zero_bool = label_vec == 0
        one_bool = label_vec == 1

        num_zero = np.sum(zero_bool, 0)
        num_one = np.sum(one_bool, 0)

        # Now rescale weights
        weights[zero_bool] = (weights * 0.5 / num_zero)[zero_bool]
        weights[one_bool] = (weights * 0.5 / num_one)[one_bool]

        # calculate new weights
        # Normalize by total sums
        weights = weights / weights.sum(0)

        # If we have multiple tasks, collapse by taking the mean
        if len(weights.shape) > 1:
            weights = weights.mean(1)

        # Idnetify number of samples by the amount of times we should sample
        # the lowest prob item to see it at least once per epoch
        total_sample_num = int(1 / np.min(weights[weights != 0]))

        # Build sampler
        sampler = WeightedRandomSampler(weights.squeeze(), total_sample_num)

        # Turn shuffle mode false
        shuffle = False

    loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=rxn_collate,
        num_workers=0,
        shuffle=shuffle,
        batch_size=batch_size,
        sampler=sampler,
    )
    return loader


class NNModel(nn.Module):
    """NNModel.
    Parent class to hold neural network models
    """

    def __init__(self, **kwargs):
        """__init__.

        Args:
        """
        self.use_gpu = False
        super(NNModel, self).__init__()

    def set_gpu(self, gpu: bool):
        self.use_gpu = gpu

    def forward(self, batch: dict, **kwargs):
        """Return probability LOGITS"""
        raise NotImplemented

    def log_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.params = sum([np.prod(p.size()) for p in model_parameters])
        logging.info(f"Number of params in model {self.params}")


class FFNNoSharing(NNModel):
    """FFNNoSharing.

    FFN With no weight sharing across tasks

    """

    def __init__(
        self,
        in_features: dict,
        layers: int = 3,
        hidden_size: int = 100,
        model_dropout: float = 0.1,
        regression: bool = False,
        num_tasks: int = 1,
        no_output: bool = False,
        **kwargs,
    ):
        """__init__.
        Args:
        """
        super(FFNNoSharing, self).__init__(**kwargs)

        self.num_tasks = num_tasks

        input_modules = []
        inner_modules = []
        output_modules = []

        # Concat only
        in_feature_list = [i[0] for i in in_features.values()]
        for _ in range(self.num_tasks):

            if None in in_feature_list:
                raise ValueError(
                    "Trying to use linear layer with variable in dimension"
                )
            else:
                in_features = sum(in_feature_list)

            self.input_layer = nn.Linear(
                in_features=in_features, out_features=hidden_size, bias=True
            )
            input_modules.append(self.input_layer)

            ## Middle layers
            sequential_layers = []
            for i in range(layers - 1):
                # Add activation before each middle layer s.t. we can use this for
                # standard logistic regression without any activation
                sequential_layers.append(torch.nn.ReLU())
                new_layer = nn.Linear(
                    in_features=hidden_size, out_features=hidden_size, bias=True
                )
                sequential_layers.append(new_layer)
                sequential_layers.append(nn.Dropout(p=model_dropout))

            self.inner_layers = nn.Sequential(*sequential_layers)
            inner_modules.append(self.inner_layers)

            # Regression or not
            self.regression = regression

            # Have multiple outputs
            self.output_layer = nn.Sequential(
                torch.nn.ReLU(),
                nn.Linear(in_features=hidden_size, out_features=1, bias=True),
            )
            output_modules.append(self.output_layer)

        self.input_modules = nn.ModuleList(input_modules)
        self.inner_modules = nn.ModuleList(inner_modules)
        self.output_modules = nn.ModuleList(output_modules)

        self.log_params()

    def forward(self, batch: dict, **kwargs):
        """Forward pass, return logits"""

        feature_ar = []

        # Concat
        if "rxn_features" in batch:
            feature_ar.append(batch["rxn_features"])

        if "prot_features" in batch:
            feature_ar.append(batch["prot_features"])

        # Concatenate the features then run module
        in_features = torch.cat(feature_ar, 1)
        if self.use_gpu:
            in_features = in_features.cuda()

        outputs = []
        for input_layer, inner_layer, outter_layer in zip(
            self.input_modules, self.inner_modules, self.output_modules
        ):

            output = input_layer(in_features)
            output = inner_layer(output)
            output = outter_layer(output)
            outputs.append(output)

        output = torch.cat(outputs, 1)
        return output, {}


class SimpleFFN(NNModel):
    """FFN."""

    def __init__(
        self,
        in_features: dict,
        layers: int = 3,
        hidden_size: int = 100,
        model_dropout: float = 0.1,
        regression: bool = False,
        num_tasks: int = 1,
        interaction_layer: str = "concat",
        no_output: bool = False,
        **kwargs,
    ):
        """__init__.
        Args:
        """
        super(SimpleFFN, self).__init__(**kwargs)

        self.no_output = no_output
        self.interaction_layer = interaction_layer
        if interaction_layer == "concat":
            in_feature_list = [i[0] for i in in_features.values()]
            if None in in_feature_list:
                raise ValueError(
                    "Trying to use linear layer with variable in dimension"
                )
            else:
                in_features = sum(in_feature_list)
            self.input_layer = nn.Linear(
                in_features=in_features, out_features=hidden_size, bias=True
            )
        elif interaction_layer == "dot":
            self.in_features_prot = in_features.get("prot_featurizer", (0,))[0]
            self.in_features_chem = in_features.get("chem_featurizer", (0,))[0]
            self.input_layer_prot = nn.Linear(
                in_features=self.in_features_prot, out_features=hidden_size, bias=True
            )
            self.input_layer_chem = nn.Linear(
                in_features=self.in_features_chem, out_features=hidden_size, bias=True
            )
        elif interaction_layer == "none":
            self.in_features = in_features.get("joint_feature_len")
            self.input_layer = nn.Linear(
                in_features=self.in_features, out_features=hidden_size, bias=True
            )
        else:
            raise ValueError(f"Unexpected value {interaction_layer}")

        ## Middle layers
        sequential_layers = []
        for i in range(layers - 1):
            # Add activation before each middle layer s.t. we can use this for
            # standard logistic regression without any activation
            sequential_layers.append(torch.nn.ReLU())
            new_layer = nn.Linear(
                in_features=hidden_size, out_features=hidden_size, bias=True
            )
            sequential_layers.append(new_layer)
            sequential_layers.append(nn.Dropout(p=model_dropout))

        self.inner_layers = nn.Sequential(*sequential_layers)

        # Regression or not
        self.regression = regression

        # Have multiple outputs
        self.num_tasks = num_tasks
        self.out_features = self.num_tasks
        self.output_layer = nn.Sequential(
            torch.nn.ReLU(),
            nn.Linear(
                in_features=hidden_size, out_features=self.out_features, bias=True
            ),
        )
        self.log_params()

    def forward(self, batch: dict, **kwargs):
        """Forward pass, return logits"""

        feature_ar = []
        if self.interaction_layer == "concat":
            if "rxn_features" in batch:
                feature_ar.append(batch["rxn_features"])

            if "prot_features" in batch:
                feature_ar.append(batch["prot_features"])

            if self.use_gpu:
                feature_ar = [j.cuda() for j in feature_ar]
            # Concatenate the features then run module
            in_features = torch.cat(feature_ar, 1)

            output = self.input_layer(in_features)

        elif self.interaction_layer == "dot":
            if "prot_features" in batch:
                feature_ar.append(batch["prot_features"])

            if len(feature_ar) <= 0:
                raise RuntimeError("Expected prot or ec features for ffn network")

            # Concatenate the features then run module
            rxn_repr = batch["rxn_features"]

            if self.use_gpu:
                feature_ar = [j.cuda() for j in feature_ar]
                rxn_repr = rxn_repr.cuda()
            protein_repr = torch.cat(feature_ar, 1)

            protein_repr = self.input_layer_prot(protein_repr)
            rxn_repr = self.input_layer_chem(rxn_repr)
            # Take element wise product between representations!
            output = torch.einsum("bd,bd->bd", protein_repr, rxn_repr)
        elif self.interaction_layer == "none":

            features = batch["x"].cuda() if self.use_gpu else batch["x"]
            features = features.float()
            output = self.input_layer(features)

        else:
            raise ValueError(f"Bad interaction layer value: {self.interaction_layer}")

        output = self.inner_layers(output)

        if not self.no_output:
            output = self.output_layer(output)

        return output, {}
