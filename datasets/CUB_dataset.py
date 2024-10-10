"""
CUB dataset loader with concept labels. 

This module provides a custom DataLoader for the CUB dataset, including concept labels for training, validation, and testing.
The dataset is preprocessed with transformations.

Classes:
    CUB_DatasetGenerator: Custom DataLoader for the CUB dataset.

Functions:
    train_test_split_CUB: Perform train-validation-test split for the CUB dataset according the predefined photographer-specific partitions.
    get_CUB_dataloaders: Get DataLoaders for the CUB dataset.
"""

"""
CIFAR-100 dataset loader Relies on create_dataset_cifar.py to have generated concept labels.

This module provides a custom DataLoader for the CIFAR-100 dataset, including concept labels for training, validation, and testing.
The dataset is preprocessed with transformations.

Classes:
    CIFAR100_CBM_dataloader: Custom DataLoader for CIFAR-100 with concept labels.

Functions:
    get_CIFAR100_CBM_dataloader: Returns DataLoaders for training, validation, and testing splits.
"""


import os
import pickle
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CUB_DatasetGenerator(Dataset):
    """CUB Dataset object"""

    def __init__(self, data_pkl, transform=None):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = data_pkl
        self.transform = transform
        self.img_index = [0, 1, 2, 3]  # Placeholder

    def __getitem__(self, index):
        # Gets an element of the dataset
        img_data = self.data[index]
        img_path = img_data["img_path"]
        imageData = Image.open(img_path).convert("RGB")
        # imageData = imageData.resize((224, 224))
        image_label = img_data["class_label"]

        image_attr = np.array(img_data["attribute_label"])

        if self.transform != None:
            imageData = self.transform(imageData)

        # Return a tuple of images, labels, and protected attributes
        return {
            "img_code": index,
            "labels": image_label,
            "features": imageData,
            "concepts": image_attr,
        }

    def __len__(self):
        return len(self.data)


def train_test_split_CUB(root_dir):
    """Performs train-validation-test split for the CUB dataset"""

    # Using pre-determined split as to have different photographers in train & test
    data_train = []
    data_val = []
    data_test = []
    data_train.extend(
        pickle.load(
            open(
                os.path.join(
                    root_dir, "CUB/CUB_processed/class_attr_data_10/train.pkl"
                ),
                "rb",
            )
        )
    )
    data_val.extend(
        pickle.load(
            open(
                os.path.join(root_dir, "CUB/CUB_processed/class_attr_data_10/val.pkl"),
                "rb",
            )
        )
    )
    data_test.extend(
        pickle.load(
            open(
                os.path.join(root_dir, "CUB/CUB_processed/class_attr_data_10/test.pkl"),
                "rb",
            )
        )
    )
    for dataset in [data_train, data_val, data_test]:
        for i in range(len(dataset)):
            parts = dataset[i]["img_path"].split("/")
            index = parts.index("images")
            end_path = "/".join(parts[index:])

            dataset[i]["img_path"] = os.path.join(
                root_dir, "CUB/CUB_200_2011/", end_path
            )

    return data_train, data_val, data_test


def get_CUB_dataloaders(config):
    """Returns a dictionary of data loaders for the CUB dataset, for the training, validation, and test sets."""

    train_imgs, val_imgs, test_imgs = train_test_split_CUB(
        root_dir=config.data_path,
    )

    # Following the transformations from CBM paper
    resol = 299
    # resized_resol = int(resol * 256 / 224)
    train_transform = transforms.Compose(
        [
            transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.Resize(size=(224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # implicitly divides by 255
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.CenterCrop(resol),
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),  # implicitly divides by 255
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Datasets
    image_datasets = {
        "train": CUB_DatasetGenerator(train_imgs, transform=train_transform),
        "val": CUB_DatasetGenerator(val_imgs, transform=test_transform),
        "test": CUB_DatasetGenerator(test_imgs, transform=test_transform),
    }

    return (
        image_datasets["train"],
        image_datasets["val"],
        image_datasets["test"],
    )
