"""
Utility functions for training.
"""

import numpy as np
from sklearn.metrics import jaccard_score
import torch
from torch import nn
from tqdm import tqdm
from torchmetrics import Metric
import wandb

from utils.metrics import calc_target_metrics, calc_concept_metrics
from utils.plotting import compute_and_plot_heatmap


def train_one_epoch_scbm(
    train_loader, model, optimizer, mode, metrics, epoch, config, loss_fn, device
):
    """
    Train the Stochastic Concept Bottleneck Model (SCBM) for one epoch.

    This function trains the SCBM for one epoch using the provided training data loader, model, optimizer, and loss function.
    It supports different training modes and updates the model parameters accordingly. The function also computes and logs
    various metrics during the training process.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        model (torch.nn.Module): The SCBM model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        mode (str): The training mode. Supported modes are:
                    - "j": Joint training of the model.
                    - "c": Training the concept head only.
                    - "t": Training the classifier head only.
        metrics (object): An object to track and compute metrics during training.
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and training settings.
        loss_fn (callable): The loss function used to compute losses.
        device (torch.device): The device to run the computations on.

    Returns:
        None

    Notes:
        - Depending on the training mode, certain parts of the model are set to evaluation mode.
        - The function iterates over the training data, performs forward and backward passes, and updates the model parameters.
        - Metrics are computed and logged at the end of each epoch.
    """

    model.train()
    metrics.reset()

    if (
        config.model.training_mode == "sequential"
        or config.model.training_mode == "independent"
    ):
        if mode == "c":
            model.head.eval()
        elif mode == "t":
            model.encoder.eval()

    for k, batch in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch + 1}", position=0, leave=True)
    ):
        batch_features, target_true = batch["features"].to(device), batch["labels"].to(
            device
        )
        concepts_true = batch["concepts"].to(device)

        # Forward pass
        concepts_mcmc_probs, triang_cov, target_pred_logits = model(
            batch_features, epoch, c_true=concepts_true
        )

        # Backward pass depends on the training mode of the model
        optimizer.zero_grad()

        # Compute the loss
        target_loss, concepts_loss, prec_loss, total_loss = loss_fn(
            concepts_mcmc_probs,
            concepts_true,
            target_pred_logits,
            target_true,
            triang_cov,
        )

        if mode == "j":
            total_loss.backward()
        elif mode == "c":
            (concepts_loss + prec_loss).backward()
        else:
            target_loss.backward()
        optimizer.step()  # perform an update

        # Store predictions
        concepts_pred_probs = concepts_mcmc_probs.mean(-1)
        metrics.update(
            target_loss,
            concepts_loss,
            total_loss,
            target_true,
            target_pred_logits,
            concepts_true,
            concepts_pred_probs,
            prec_loss=prec_loss,
        )

    # Calculate and log metrics
    metrics_dict = metrics.compute()
    wandb.log({f"train/{k}": v for k, v in metrics_dict.items()})
    prints = f"Epoch {epoch + 1}, Train     : "
    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    metrics.reset()
    return


def train_one_epoch_cbm(
    train_loader, model, optimizer, mode, metrics, epoch, config, loss_fn, device
):
    """
    Train a baseline method for one epoch.

    This function trains the CEM/AR/CBM for one epoch using the provided training data loader, model, optimizer, and loss function.
    It supports different training modes and updates the model parameters accordingly. The function also computes and logs
    various metrics during the training process.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        model (torch.nn.Module): The SCBM model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        mode (str): The training mode. Supported modes are:
                    - "j": Joint training of the model.
                    - "c": Training the concept head only.
                    - "t": Training the classifier head only.
        metrics (object): An object to track and compute metrics during training.
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and training settings.
        loss_fn (callable): The loss function used to compute losses.
        device (torch.device): The device to run the computations on.

    Returns:
        None

    Notes:
        - Depending on the training mode, certain parts of the model are set to evaluation mode.
        - The function iterates over the training data, performs forward and backward passes, and updates the model parameters.
        - Metrics are computed and logged at the end of each epoch.
    """

    model.train()
    metrics.reset()

    if config.model.training_mode in ("sequential", "independent"):
        if mode == "c":
            model.head.eval()
        elif mode == "t":
            model.encoder.eval()

    for k, batch in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch + 1}", position=0, leave=True)
    ):
        batch_features, target_true = batch["features"].to(device), batch["labels"].to(
            device
        )
        concepts_true = batch["concepts"].to(device)

        # Forward pass
        if config.model.training_mode == "independent" and mode == "t":
            concepts_pred_probs, target_pred_logits, concepts_hard = model(
                batch_features, epoch, concepts_true
            )
        elif config.model.concept_learning == "autoregressive" and mode == "c":
            concepts_pred_probs, target_pred_logits, concepts_hard = model(
                batch_features, epoch, concepts_train_ar=concepts_true
            )
        else:
            concepts_pred_probs, target_pred_logits, concepts_hard = model(
                batch_features, epoch
            )
        # Backward pass depends on the training mode of the model
        optimizer.zero_grad()
        # Compute the loss
        target_loss, concepts_loss, total_loss = loss_fn(
            concepts_pred_probs, concepts_true, target_pred_logits, target_true
        )

        if mode == "j":
            total_loss.backward()
        elif mode == "c":
            concepts_loss.backward()
        else:
            target_loss.backward()
        optimizer.step()  # perform an update

        # Store predictions
        metrics.update(
            target_loss,
            concepts_loss,
            total_loss,
            target_true,
            target_pred_logits,
            concepts_true,
            concepts_pred_probs,
        )

    # Calculate and log metrics
    metrics_dict = metrics.compute()
    wandb.log({f"train/{k}": v for k, v in metrics_dict.items()})
    prints = f"Epoch {epoch + 1}, Train     : "
    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    metrics.reset()
    return


def validate_one_epoch_scbm(
    loader,
    model,
    metrics,
    epoch,
    config,
    loss_fn,
    device,
    test=False,
    concept_names_graph=None,
):
    """
    Validate the Stochastic Concept Bottleneck Model (SCBM) for one epoch.

    This function evaluates the SCBM for one epoch using the provided data loader, model, and loss function.
    It computes and logs various metrics during the validation process. It also generates
    and plots a heatmap of the learned concept correlation matrix on the final test set.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader for the validation or test data.
        model (torch.nn.Module): The SCBM model to be validated.
        metrics (object): An object to track and compute metrics during validation.
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and validation settings.
        loss_fn (callable): The loss function used to compute losses.
        device (torch.device): The device to run the computations on.
        test (bool, optional): Flag indicating whether this is the final evaluation on the test set. Default is False.
        concept_names_graph (list, optional): List of concept names for plotting the heatmap.
                                              Default is None for which range(n_concepts) is used.

    Returns:
        None

    Notes:
        - The function sets the model to evaluation mode and disables gradient computation.
        - It iterates over the validation data, performs forward passes, and computes the losses.
        - Metrics are computed and logged at the end of the validation epoch.
        - During testing, the function generates and plots a heatmap of the concept correlation matrix.
    """
    model.eval()
    with torch.no_grad():

        for k, batch in enumerate(
            tqdm(loader, desc=f"Epoch {epoch}", position=0, leave=True)
        ):
            batch_features, target_true = batch["features"].to(device), batch[
                "labels"
            ].to(device)
            concepts_true = batch["concepts"].to(device)

            concepts_mcmc_probs, triang_cov, target_pred_logits = model(
                batch_features, epoch, validation=True, c_true=concepts_true
            )
            # Compute covariance matrix of concepts
            cov = torch.matmul(triang_cov, torch.transpose(triang_cov, dim0=1, dim1=2))

            if test and k % (len(loader) // 10) == 0:
                try:
                    corr = (cov[0] / cov[0].diag().sqrt()).transpose(
                        dim0=0, dim1=1
                    ) / cov[0].diag().sqrt()
                    matrix = corr.cpu().numpy()

                    compute_and_plot_heatmap(
                        matrix, concepts_true, concept_names_graph, config
                    )

                except:
                    pass

            target_loss, concepts_loss, prec_loss, total_loss = loss_fn(
                concepts_mcmc_probs,
                concepts_true,
                target_pred_logits,
                target_true,
                triang_cov,
            )

            # Store predictions
            concepts_pred_probs = concepts_mcmc_probs.mean(-1)
            metrics.update(
                target_loss,
                concepts_loss,
                total_loss,
                target_true,
                target_pred_logits,
                concepts_true,
                concepts_pred_probs,
                prec_loss=prec_loss,
            )

    # Calculate and log metrics
    metrics_dict = metrics.compute(validation=True, config=config)

    if not test:
        wandb.log({f"validation/{k}": v for k, v in metrics_dict.items()})
        prints = f"Epoch {epoch}, Validation: "
    else:
        wandb.log({f"test/{k}": v for k, v in metrics_dict.items()})
        prints = f"Test: "
    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    print()
    metrics.reset()
    return


def validate_one_epoch_cbm(
    loader,
    model,
    metrics,
    epoch,
    config,
    loss_fn,
    device,
    test=False,
    concept_names_graph=None,
):
    """
    Validate a baseline method for one epoch.

    This function evaluates the CEM/AR/CBM for one epoch using the provided data loader, model, and loss function.
    It computes and logs various metrics during the validation process.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader for the validation or test data.
        model (torch.nn.Module): The model to be validated.
        metrics (object): An object to track and compute metrics during validation.
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and validation settings.
        loss_fn (callable): The loss function used to compute losses.
        device (torch.device): The device to run the computations on.
        test (bool, optional): Flag indicating whether this is the final evaluation on the test set. Default is False.

    Returns:
        None

    Notes:
        - The function sets the model to evaluation mode and disables gradient computation.
        - It iterates over the validation data, performs forward passes, and computes the losses.
        - Metrics are computed and logged at the end of the validation epoch.
    """
    model.eval()

    with torch.no_grad():
        for k, batch in enumerate(
            tqdm(loader, desc=f"Epoch {epoch}", position=0, leave=True)
        ):
            batch_features, target_true = batch["features"].to(device), batch[
                "labels"
            ].to(device)
            concepts_true = batch["concepts"].to(device)

            concepts_pred_probs, target_pred_logits, concepts_hard = model(
                batch_features, epoch, validation=True
            )
            if config.model.concept_learning == "autoregressive":
                concepts_input = concepts_hard
            elif config.model.concept_learning == "hard":
                concepts_input = concepts_hard
            else:
                concepts_input = concepts_pred_probs
            if config.model.concept_learning == "autoregressive":
                concepts_pred_probs = torch.mean(
                    concepts_pred_probs, dim=-1
                )  # Calculating the metrics on the average probabilities from MCMC

            target_loss, concepts_loss, total_loss = loss_fn(
                concepts_pred_probs, concepts_true, target_pred_logits, target_true
            )

            # Store predictions
            metrics.update(
                target_loss,
                concepts_loss,
                total_loss,
                target_true,
                target_pred_logits,
                concepts_true,
                concepts_pred_probs,
            )

    # Calculate and log metrics
    metrics_dict = metrics.compute(validation=True, config=config)
    if not test:
        wandb.log({f"validation/{k}": v for k, v in metrics_dict.items()})
        prints = f"Epoch {epoch}, Validation: "
    else:
        wandb.log({f"test/{k}": v for k, v in metrics_dict.items()})
        prints = f"Test: "
    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    print()
    metrics.reset()
    return


def create_optimizer(config, model):
    """
    Parse the configuration file and return a optimizer object to update the model parameters.
    """
    assert config.optimizer in [
        "sgd",
        "adam",
    ], "Only SGD and Adam optimizers are available!"

    optim_params = [
        {
            "params": filter(lambda p: p.requires_grad, model.parameters()),
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
        }
    ]

    if config.optimizer == "sgd":
        return torch.optim.SGD(optim_params)
    elif config.optimizer == "adam":
        return torch.optim.Adam(optim_params)


class Custom_Metrics(Metric):
    """
    Custom metrics class for tracking and computing various metrics during training and validation.

    This class extends the PyTorch Metric class and provides methods to update and compute metrics such as
    target loss, concept loss, total loss, accuracy, and Jaccard index for both target and concepts.
    It is being updated for each batch. At the end of each epoch, the compute function is called to compute
    the final metrics and return them as a dictionary.

    Args:
        n_concepts (int): The number of concepts in the model.
        device (torch.device): The device to run the computations on.

    Attributes:
        n_concepts (int): The number of concepts in the model.
        target_loss (torch.Tensor): The accumulated target loss.
        concepts_loss (torch.Tensor): The accumulated concepts loss.
        total_loss (torch.Tensor): The accumulated total loss.
        y_true (list): List of true target labels.
        y_pred_logits (list): List of predicted target logits.
        c_true (list): List of true concept labels.
        c_pred_probs (list): List of predicted concept probabilities.
        batch_features (list): List of batch features.
        cov_norm (torch.Tensor): The accumulated covariance norm.
        n_samples (torch.Tensor): The number of samples processed.
        prec_loss (torch.Tensor): The accumulated precision loss.
    """

    def __init__(self, n_concepts, device):
        super().__init__()
        self.n_concepts = n_concepts
        self.add_state("target_loss", default=torch.tensor(0.0, device=device))
        self.add_state("concepts_loss", default=torch.tensor(0.0, device=device))
        self.add_state("total_loss", default=torch.tensor(0.0, device=device))
        self.add_state("y_true", default=[])
        self.add_state("y_pred_logits", default=[])
        self.add_state("c_true", default=[])
        (
            self.add_state("c_pred_probs", default=[]),
            self.add_state("concepts_input", default=[]),
        ),
        self.add_state("batch_features", default=[])
        self.add_state("cov_norm", default=torch.tensor(0.0, device=device))
        self.add_state(
            "n_samples", default=torch.tensor(0, dtype=torch.int, device=device)
        )
        self.add_state("prec_loss", default=torch.tensor(0.0, device=device))

    def update(
        self,
        target_loss: torch.Tensor,
        concepts_loss: torch.Tensor,
        total_loss: torch.Tensor,
        y_true: torch.Tensor,
        y_pred_logits: torch.Tensor,
        c_true: torch.Tensor,
        c_pred_probs: torch.Tensor,
        cov_norm: torch.Tensor = None,
        prec_loss: torch.Tensor = None,
    ):
        assert c_true.shape == c_pred_probs.shape

        n_samples = y_true.size(0)
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCELoss()
        self.n_samples += n_samples
        self.target_loss += target_loss * n_samples
        self.concepts_loss += concepts_loss * n_samples
        self.total_loss += total_loss * n_samples
        self.y_true.append(y_true)
        self.y_pred_logits.append(y_pred_logits.detach())
        self.c_true.append(c_true)
        self.c_pred_probs.append(c_pred_probs.detach())
        if cov_norm:
            self.cov_norm += cov_norm * n_samples
        if prec_loss:
            self.prec_loss += prec_loss * n_samples

    def compute(self, validation=False, config=None):
        self.y_true = torch.cat(self.y_true, dim=0).cpu()
        self.c_true = torch.cat(self.c_true, dim=0).cpu()
        self.c_pred_probs = torch.cat(self.c_pred_probs, dim=0).cpu()
        self.y_pred_logits = torch.cat(self.y_pred_logits, dim=0).cpu()
        self.c_true = self.c_true.cpu().numpy()
        self.c_pred_probs = self.c_pred_probs.cpu().numpy()
        c_pred = self.c_pred_probs > 0.5
        if self.y_pred_logits.size(1) == 1:
            y_pred_probs = nn.Sigmoid()(self.y_pred_logits.squeeze())
            y_pred = y_pred_probs > 0.5
        else:
            y_pred_probs = nn.Softmax(dim=1)(self.y_pred_logits)
            y_pred = self.y_pred_logits.argmax(dim=-1)

        target_acc = (self.y_true == y_pred).sum() / self.n_samples
        concept_acc = (self.c_true == c_pred).sum() / (self.n_samples * self.n_concepts)
        complete_concept_acc = (
            (self.c_true == c_pred).sum(1) == self.n_concepts
        ).sum() / self.n_samples
        target_jaccard = jaccard_score(self.y_true, y_pred, average="micro")
        concept_jaccard = jaccard_score(self.c_true, c_pred, average="micro")
        metrics = dict(
            {
                "target_loss": self.target_loss / self.n_samples,
                "prec_loss": self.prec_loss / self.n_samples,
                "concepts_loss": self.concepts_loss / self.n_samples,
                "total_loss": self.total_loss / self.n_samples,
                "y_accuracy": target_acc,
                "c_accuracy": concept_acc,
                "complete_c_accuracy": complete_concept_acc,
                "target_jaccard": target_jaccard,
                "concept_jaccard": concept_jaccard,
            }
        )

        if self.cov_norm != 0:
            metrics = metrics | {"covariance_norm": self.cov_norm / self.n_samples}

        if validation is True:
            c_pred_probs = []
            for j in range(self.n_concepts):
                c_pred_probs.append(
                    np.hstack(
                        (
                            np.expand_dims(1 - self.c_pred_probs[:, j], 1),
                            np.expand_dims(self.c_pred_probs[:, j], 1),
                        )
                    )
                )

            y_metrics = calc_target_metrics(
                self.y_true.numpy(), y_pred_probs.numpy(), config.data
            )
            c_metrics, c_metrics_per_concept = calc_concept_metrics(
                self.c_true, c_pred_probs, config.data
            )
            metrics = (
                metrics
                | {f"y_{k}": v for k, v in y_metrics.items()}
                | {f"c_{k}": v for k, v in c_metrics.items()}
            )  # | c_metrics_per_concept # Update dict

        return metrics


def freeze_module(m):
    m.eval()
    for param in m.parameters():
        param.requires_grad = False


def unfreeze_module(m):
    m.train()
    for param in m.parameters():
        param.requires_grad = True
