"""
Utility functions for data loading.
"""

import os
import torch
from torch.utils.data import DataLoader

from datasets.synthetic_dataset import get_synthetic_datasets
from datasets.CUB_dataset import get_CUB_dataloaders
from datasets.cifar10_dataset import get_CIFAR10_CBM_dataloader
from datasets.cifar100_dataset import get_CIFAR100_CBM_dataloader
from utils.utils import numerical_stability_check


def get_data(config_base, config, gen):
    """
    Parse the configuration file and return the relevant dataset loaders.

    This function parses the provided configuration file and returns the appropriate dataset loaders based on the
    specified dataset type. It also sets the data path based on the hostname or the configuration file if working
    locally and on a cluster. The function supports synthetic datasets, CUB, CIFAR-10, and CIFAR-100 datasets.

    Args:
        config_base (dict): The base configuration dictionary.
        config (dict): The data configuration dictionary containing dataset and data path information.
        gen (object): A generator object to control the randomness of the data loader.

    Returns:
        tuple: A tuple containing the training data loader, validation data loader, and test data loader.
    """
    hostname = os.uname()[1]
    if "biomed" in hostname:
        # Remote Datafolder for our group cluster
        config.data_path = "/cluster/dataset/vogtlab/Projects/CBM/"
    elif "data_path" not in config:
        # Local Datafolder if not already specified in yaml
        config.data_path = "../datasets/"
    elif config.data_path is None:
        config.data_path = "../datasets/"
    else:
        pass

    if config.dataset == "synthetic":
        print("SYNTHETIC DATASET")
        type = None
        if "sim_type" in config:
            type = config.sim_type
            print("SIMULATION TYPE: " + str(type))
            if config.num_classes > 2:
                raise NotImplementedError(
                    "ERROR: Only binary classification is supported for synthetic data."
                )
        trainset, validset, testset = get_synthetic_datasets(
            num_vars=config.num_covariates,
            num_points=config.num_points,
            num_predicates=config.num_concepts,
            train_ratio=0.6,
            val_ratio=0.2,
            type=type,
            seed=config_base.seed,
        )
    elif config.dataset == "CUB":
        print("CUB DATASET")
        trainset, validset, testset = get_CUB_dataloaders(
            config,
        )
    elif config.dataset == "cifar10":
        print("CIFAR-10 DATASET")
        trainset, validset, testset = get_CIFAR10_CBM_dataloader(
            config.data_path,
        )
    elif config.dataset == "cifar100":
        print("CIFAR-100 DATASET")
        trainset, validset, testset = get_CIFAR100_CBM_dataloader(
            config.data_path,
        )
    else:
        NotImplementedError("ERROR: Dataset not supported!")

    config = config_base
    train_loader = DataLoader(
        trainset,
        batch_size=config.model.train_batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True,
        generator=gen,
        drop_last=True,
    )
    val_loader = DataLoader(
        validset,
        batch_size=config.model.val_batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True,
        generator=gen,
    )
    test_loader = DataLoader(
        testset,
        batch_size=config.model.val_batch_size,
        num_workers=config.workers,
        generator=gen,
    )

    return train_loader, val_loader, test_loader


def get_empirical_covariance(dataloader):
    """
    Compute the empirical covariance matrix of the concepts in the given dataloader.

    This function computes the empirical covariance matrix of the concepts in the given dataloader.
    It first concatenates all the concept data into a single tensor, then applies a logit transformation
    to the data to work in the correct space. The covariance matrix is computed from the transformed data
    and brought into a lower triangular form using Cholesky decomposition. In comments, an alternative
    covariance computation is provided that can be used if the dataset is too large to fit into memory.

    Args:
        dataloader (torch.utils.data.DataLoader): A dataloader containing batches of data with a "concepts" key.

    Returns:
        torch.Tensor: The lower triangular form of the empirical covariance matrix.
    """
    data = []
    for batch in dataloader:
        concepts = batch["concepts"]
        data.append(concepts)
    data = torch.cat(data)  # Concatenate all data into a single tensor
    data_logits = torch.logit(0.05 + 0.9 * data)
    covariance = torch.cov(data_logits.transpose(0, 1))

    # Bringing it into lower triangular form
    covariance = numerical_stability_check(covariance, device="cpu")
    lower_triangle = torch.linalg.cholesky(covariance)

    ####### Alternative cov computation if dataset was too large for memory
    # num_samples = 0
    # for i, batch in enumerate(dataloader):
    #     concepts = batch["concepts"]
    #     if i == 0:
    #         logits = torch.logit(0.05 + 0.9 * concepts).sum(0)
    #     else:
    #         logits += torch.logit(0.05 + 0.9 * concepts).sum(0)
    #     num_samples += concepts.shape[0]
    # logits_mean = logits / num_samples

    # for i, batch in enumerate(dataloader):
    #     concepts = batch["concepts"]
    #     temp = (torch.logit(0.05 + 0.9 * concepts) - logits_mean).unsqueeze(-1)
    #     if i == 0:
    #         cov = torch.matmul(temp, temp.transpose(-2, -1)).sum(0)
    #     else:
    #         cov += torch.matmul(temp, temp.transpose(-2, -1)).sum(0)
    # cov = cov / num_samples

    ########
    return lower_triangle


def get_concept_groups(config):
    """
    Retrieve the concept groups based on the dataset specified in the configuration.

    This function retrieves the concept groups based on the dataset specified in the configuration.
    This is used for plotting the heatmap of the correlation matrix with the correct concept names.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        list: A list of concept names.
    """
    if config.dataset == "CUB":
        # Oracle grouping based on concept type for CUB
        with open(
            os.path.join(config.data_path, "CUB/CUB_200_2011/concept_names.txt"),
            "r",
        ) as f:
            concept_names = []
            for line in f:
                concept_names.append(line.replace("\n", "").split("::"))
        concept_names_graph = [": ".join(name) for name in concept_names]

    elif config.dataset == "cifar10":
        # Oracle grouping based on concept type for cifar10
        with open(
            os.path.join(config.data_path, "cifar10/cifar10_filtered.txt"), "r"
        ) as file:
            # Read the contents of the file
            concept_names_graph = [line.strip() for line in file]
    else:
        concept_names_graph = [str(i) for i in range(config.num_concepts)]

    return concept_names_graph
