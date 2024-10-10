"""
CIFAR-10 dataset loader with concept labels. Relies on create_dataset_cifar.py to have generated concept labels.

This module provides a custom DataLoader for the CIFAR-10 dataset, including concept labels for training, validation, and testing.
The dataset is preprocessed with transformations.

Classes:
    CIFAR10_CBM_dataloader: Custom DataLoader for CIFAR-10 with concept labels.

Functions:
    get_CIFAR10_CBM_dataloader: Returns DataLoaders for training, validation, and testing splits.
"""

import torch
from torchvision import datasets, transforms


def get_CIFAR10_CBM_dataloader(datapath):
    datapath = datapath + "cifar10/"
    image_datasets = {
        "train": CIFAR10_CBM_dataloader(
            root=datapath,
            train=True,
            download=False,
        ),
        "val": CIFAR10_CBM_dataloader(
            root=datapath,
            train=False,
            download=False,
        ),
        "test": CIFAR10_CBM_dataloader(
            root=datapath,
            train=False,
            download=False,
        ),
    }

    return image_datasets["train"], image_datasets["val"], image_datasets["test"]


class CIFAR10_CBM_dataloader(datasets.CIFAR10):

    def __init__(self, *args, **kwargs):
        super(CIFAR10_CBM_dataloader, self).__init__(*args, **kwargs)

        if kwargs["train"]:
            self.transform = transforms.Compose(
                [
                    transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
                    transforms.Resize(size=(224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),  # implicitly divides by 255
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            self.concepts = (
                torch.load(kwargs["root"] + f"cifar10_train_concept_labels.pt") * 1
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),  # implicitly divides by 255
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            self.concepts = (
                torch.load(kwargs["root"] + f"cifar10_test_concept_labels.pt") * 1
            )

    def __getitem__(self, idx):
        X, target = super().__getitem__(idx)

        return {
            "img_code": idx,
            "labels": target,
            "features": X,
            "concepts": self.concepts[idx],
        }
