"""
Utility methods for constructing loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


def create_loss(config):
    """
    Create and return a loss function based on the configuration.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        nn.Module: The loss function.
    """
    if config.model.model == "cbm":
        return CBLoss(
            num_classes=config.data.num_classes,
            reduction="mean",
            alpha=config.model.alpha,
            config=config.model,
        )
    elif config.model.model == "scbm":
        return SCBLoss(
            num_classes=config.data.num_classes,
            alpha=config.model.alpha,
            config=config.model,
        )
    else:
        raise NotImplementedError


class CBLoss(nn.Module):
    """
    Loss function for the Concept Bottleneck Model (CBM).
    """

    def __init__(
        self,
        num_classes: Optional[int] = 2,
        reduction: str = "mean",
        alpha: float = 1,
        config: dict = {},
    ) -> None:
        """
        Initialize the CBLoss.

        Args:
            num_classes (int, optional): Number of target classes.
            reduction (str, optional): Reduction method for the loss.
            alpha (float, optional): Weight in joint training.
            config (dict, optional): Configuration dictionary.
        """
        super(CBLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha if config.training_mode == "joint" else 1.0
        self.reduction = reduction

    def forward(
        self,
        concepts_pred_probs: Tensor,
        concepts_true: Tensor,
        target_pred_logits: Tensor,
        target_true: Tensor,
    ) -> Tensor:
        """
        Compute the loss.

        Args:
            concepts_pred_probs (Tensor): Predicted concept probabilities.
            concepts_true (Tensor): Ground-truth concept values.
            target_pred_logits (Tensor): Predicted target logits.
            target_true (Tensor): Ground-truth target values.

        Returns:
            Tensor: Target loss, concept loss, and total loss.
        """

        concepts_loss = 0

        assert torch.all((concepts_true == 0) | (concepts_true == 1))

        for concept_idx in range(concepts_true.shape[1]):
            c_loss = F.binary_cross_entropy(
                concepts_pred_probs[:, concept_idx],
                concepts_true[:, concept_idx].float(),
                reduction=self.reduction,
            )
            concepts_loss += c_loss
        concepts_loss = self.alpha * concepts_loss

        if self.num_classes == 2:
            # Logits to probs
            target_pred_probs = nn.Sigmoid()(target_pred_logits.squeeze(1))
            target_loss = F.binary_cross_entropy(
                target_pred_probs, target_true.float(), reduction=self.reduction
            )
        else:
            target_loss = F.cross_entropy(
                target_pred_logits, target_true.long(), reduction=self.reduction
            )

        total_loss = target_loss + concepts_loss

        return target_loss, concepts_loss, total_loss


class SCBLoss(nn.Module):
    """
    Loss function for the Stochastic Concept Bottleneck Model (SCBM).
    """

    def __init__(
        self, num_classes: Optional[int] = 2, alpha: float = 1, config: dict = {}
    ) -> None:
        """
        Initialize the SCBLoss.

        Args:
            num_classes (int, optional): Number of target classes.
            alpha (float, optional): Weight for joint training.
            config (dict, optional): Configuration dictionary.
        """
        super(SCBLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha if config.training_mode == "joint" else 1.0
        self.reg_precision = config.reg_precision
        self.reg_weight = config.reg_weight

    def forward(
        self,
        concepts_mcmc_probs: Tensor,
        concepts_true: Tensor,
        target_pred_logits: Tensor,
        target_true: Tensor,
        c_triang_cov: Tensor,
        cov_not_triang=False,
    ) -> Tensor:
        """
        Compute the loss.

        Args:
            concepts_mcmc_probs (Tensor): MCMC matrix of predicted concept probabilities.
            concepts_true (Tensor): Ground-truth concept values.
            target_pred_logits (Tensor): Predicted target logits.
            target_true (Tensor): Ground-truth target values.
            c_triang_cov (Tensor): Cholesky decomposition of the concept covariance matrix.
            cov_not_triang (bool, optional): Flag indicating if the covariance is in its cholesky form or already the covariance.

        Returns:
            Tensor: Target loss, concept loss, precision loss, and total loss.
        """

        assert torch.all((concepts_true == 0) | (concepts_true == 1))
        concepts_true_expanded = concepts_true.unsqueeze(-1).expand_as(
            concepts_mcmc_probs
        )

        bce_loss = F.binary_cross_entropy(
            concepts_mcmc_probs, concepts_true_expanded.float(), reduction="none"
        )  # [B,C,MCMC]
        intermediate_concepts_loss = -torch.sum(bce_loss, dim=1)  # [B,MCMC]
        mcmc_loss = -torch.logsumexp(
            intermediate_concepts_loss, dim=1
        )  # [B], logsumexp for numerical stability due to shift invariance
        concepts_loss = self.alpha * torch.mean(mcmc_loss)

        if self.num_classes == 2:
            # Logits to probs
            target_pred_probs = nn.Sigmoid()(target_pred_logits.squeeze(1))
            target_loss = F.binary_cross_entropy(
                target_pred_probs, target_true.float(), reduction="mean"
            )
        else:
            target_loss = F.cross_entropy(
                target_pred_logits, target_true.long(), reduction="mean"
            )

        # Add precision loss
        if self.reg_precision == "l1":
            if cov_not_triang:
                prec_matrix = torch.inverse(c_triang_cov)
            else:
                c_triang_inv = torch.inverse(c_triang_cov)
                prec_matrix = torch.matmul(
                    torch.transpose(c_triang_inv, dim0=1, dim1=2), c_triang_inv
                )
            prec_loss = prec_matrix.abs().sum(dim=(1, 2)) - prec_matrix.diagonal(
                offset=0, dim1=1, dim2=2
            ).abs().sum(-1)
            if prec_matrix.size(1) > 1:
                prec_loss = prec_loss / (
                    prec_matrix.size(1) * (prec_matrix.size(1) - 1)
                )
            else:  # Univariate case, can happen when intervening
                prec_loss = prec_loss
            prec_loss = self.reg_weight * prec_loss.mean(-1)
        else:
            prec_loss = torch.zeros_like(concepts_loss)

        total_loss = target_loss + concepts_loss + prec_loss

        return target_loss, concepts_loss, prec_loss, total_loss
