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

import ctypes
import os
import pickle
from PIL import Image
import numpy as np
import multiprocessing as mp

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CUB_DatasetGenerator(Dataset): 
    """CUB Dataset object with caching"""
 
    def __init__(self, data_pkl, transform=None, cache=False): 
        """ 
        Arguments: 
        data_pkl: list of data dictionaries containing img_path, class_label, attribute_label
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing 
        cache: Whether to cache the dataset in shared system RAM.
        """ 
        self.data = data_pkl 
        self.transform = transform 
        self.cache = cache

        num_samples = len(data_pkl) 
        
        # Maximum possible dimensions from CUB 
        max_height = 500 
        max_width = 500 
        data_dims = (3, max_height, max_width) 
        dimension = int(np.prod(data_dims)) 
        
        if self.cache:
            # Create shared array for image data (padded to max size)
            shared_array_base = mp.Array(
                ctypes.c_uint8, num_samples * dimension 
            ) 
            shared_array = np.ctypeslib.as_array(shared_array_base.get_obj()) 
            shared_array = shared_array.reshape(num_samples, *data_dims) 
            self.image_cache = torch.from_numpy(shared_array)
            
            # Create shared array for image dimensions and validity
            # Format: [height, width] per image, initialized to [-1, -1] (invalid)
            dims_array_base = mp.Array(
                ctypes.c_int, num_samples * 2  # 2 values per image: height, width
            )
            dims_array = np.ctypeslib.as_array(dims_array_base.get_obj())
            dims_array = dims_array.reshape(num_samples, 2)
            self.dims_cache = torch.from_numpy(dims_array)
            self.dims_cache.fill_(-1)  # Initialize to -1 to indicate not cached
            
            # CUB has 112 binary attributes - we need 112 bits = 14 bytes per sample
            # We'll use 15 bytes (120 bits) for easier alignment and future expansion
            attr_size = len(data_pkl[0]["attribute_label"])
            self.num_attributes = attr_size
            bytes_per_sample = (attr_size + 7) // 8  # Round up to nearest byte

            attr_array_base = mp.Array(
                ctypes.c_uint8, num_samples * bytes_per_sample
            )
            attr_array = np.ctypeslib.as_array(attr_array_base.get_obj())
            attr_array = attr_array.reshape(num_samples, bytes_per_sample)
            self.attr_cache = torch.from_numpy(attr_array)
            self.attr_cache.fill_(0)  # Initialize to 0
            
            # Create shared array for class labels
            label_array_base = mp.Array(
                ctypes.c_int, num_samples
            )
            label_array = np.ctypeslib.as_array(label_array_base.get_obj())
            self.label_cache = torch.from_numpy(label_array)
            self.label_cache.fill_(-1)  # Initialize to -1 to indicate not cached

    def _pack_attributes(self, attributes):
        """
        Pack binary attributes into bytes.
        
        Args:
            attributes: numpy array of binary values (0 or 1)
        
        Returns:
            Packed byte array
        """
        # Ensure attributes are binary
        attributes = np.array(attributes, dtype=np.uint8)
        attributes = np.clip(attributes, 0, 1)  # Ensure binary
        
        # Calculate number of bytes needed
        n_bytes = (len(attributes) + 7) // 8
        packed = np.zeros(n_bytes, dtype=np.uint8)
        
        # Pack bits into bytes
        for i, attr in enumerate(attributes):
            if attr:
                byte_idx = i // 8
                bit_idx = i % 8
                packed[byte_idx] |= (1 << bit_idx)
        
        return packed
    
    def _unpack_attributes(self, packed_bytes):
        """
        Unpack bytes into binary attributes.
        
        Args:
            packed_bytes: byte array
        
        Returns:
            numpy array of binary values (0 or 1)
        """
        attributes = np.zeros(self.num_attributes, dtype=np.float64)
        
        for i in range(self.num_attributes):
            byte_idx = i // 8
            bit_idx = i % 8
            if byte_idx < len(packed_bytes):
                bit_value = (packed_bytes[byte_idx] >> bit_idx) & 1
                # Store as float64 to match original
                attributes[i] = float(bit_value)
        
        # Return as float32 for consistency with model expectations
        return attributes.astype(np.float32)

    def _is_cached(self, index):
        """Check if an image is already cached by looking at dimensions"""
        if self.cache:
            return self.dims_cache[index][0] != -1 and self.dims_cache[index][1] != -1
        return False

    def _cache_image(self, index, image_pil, image_attr, image_label):
        """Cache an image and its metadata in the shared arrays"""
        # Convert PIL image to numpy array
        img_array = np.array(image_pil)
        h, w = img_array.shape[:2]
        
        # Store dimensions
        self.dims_cache[index] = torch.tensor([h, w])
        
        # Pad image to maximum size if necessary
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)  # Convert to RGB
        
        # Convert to CHW format and pad
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC -> CHW
        
        # Pad to maximum size
        padded_img = torch.zeros((3, 500, 500), dtype=torch.uint8)
        padded_img[:, :h, :w] = img_tensor
        
        # Store in cache
        self.image_cache[index] = padded_img
        
        # Pack and store attributes
        packed_attrs = self._pack_attributes(image_attr)
        self.attr_cache[index] = torch.from_numpy(packed_attrs)
        
        self.label_cache[index] = image_label

    def _get_cached_image(self, index):
        """Retrieve a cached image with its original dimensions"""
        h, w = self.dims_cache[index]
        h, w = int(h), int(w)
        
        # Extract the image without padding
        img_tensor = self.image_cache[index][:, :h, :w]  # CHW format
        
        # Convert back to PIL Image (HWC format)
        img_array = img_tensor.permute(1, 2, 0).numpy()  # CHW -> HWC
        image_pil = Image.fromarray(img_array)
        
        # Unpack attributes
        packed_attrs = self.attr_cache[index].numpy()
        image_attr = self._unpack_attributes(packed_attrs)
        
        image_label = int(self.label_cache[index])
        
        return image_pil, image_attr, image_label

    def __getitem__(self, index): 
        # Check if already cached
        if self._is_cached(index):
            image_data, image_attr, image_label = self._get_cached_image(index)
        else: 
            img_data = self.data[index] 
            img_path = img_data["img_path"] 
            image_data = Image.open(img_path).convert("RGB") 
            image_label = img_data["class_label"] 
            image_attr = np.array(img_data["attribute_label"], dtype=np.float32)
            
            if self.cache:
                self._cache_image(index, image_data, image_attr, image_label)
        
        if self.transform is not None: 
            image_data = self.transform(image_data) 
 
        # Return a tuple of images, labels, and protected attributes 
        return { 
            "img_code": index, 
            "labels": image_label, 
            "features": image_data, 
            "concepts": image_attr,  # This is now float32 array as expected
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
        "train": CUB_DatasetGenerator(train_imgs, transform=train_transform, cache=True),
        "val": CUB_DatasetGenerator(val_imgs, transform=test_transform, cache=True),
        "test": CUB_DatasetGenerator(test_imgs, transform=test_transform, cache=False),
    }

    return (
        image_datasets["train"],
        image_datasets["val"],
        image_datasets["test"],
    )
