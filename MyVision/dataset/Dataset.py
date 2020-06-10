import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import cv2

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import albumentations
import torchvision.transforms as transforms


import operator

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DatasetUtils:
    """
    This class contains utilities for making a Pytorch Dataset.  
    """

    def __init__(
        self, train_df, image_path_column, target_column, train_tfms, valid_tfms
    ):
        self.train_df = train_df
        self.image_path_column = image_path_column
        self.target_column = target_column
        self.train_tfms = train_tfms
        self.valid_tfms = valid_tfms

    def splitter(self, valid_size=0.25):
        train_images, valid_images, train_labels, valid_labels = train_test_split(
            self.train_df[self.image_path_column],
            self.train_df[self.target_column],
            test_size=valid_size,
            random_state=42,
        )
        return (
            train_images.values,
            train_labels.values,
            valid_images.values,
            valid_labels.values,
        )

    def make_dataset(
        self, resize, train_idx=None, val_idx=None, valid_size=0.25, is_CV=False
    ):
        if is_CV:
            train_dataset = CVDataset(
                df=self.train_df,
                indices=train_idx,
                image_paths=self.image_path_column,
                target_cols=self.target_column,
                transform=self.train_tfms,
                resize=resize,
            )

            valid_dataset = CVDataset(
                df=self.train_df,
                indices=val_idx,
                image_paths=self.image_path_column,
                target_cols=self.target_column,
                transform=self.valid_tfms,
                resize=resize,
            )
        else:

            (
                train_image_paths,
                train_labels,
                valid_image_paths,
                valid_labels,
            ) = self.splitter(valid_size=valid_size)

            train_dataset = SimpleDataset(
                train_image_paths, train_labels, transform=self.train_tfms
            )

            valid_dataset = SimpleDataset(
                valid_image_paths, valid_labels, transform=self.valid_tfms
            )

        return train_dataset, valid_dataset


class SimpleDataset(Dataset):
    def __init__(self, image_paths, targets, transform=None, resize=224):
        self.image_paths = image_paths
        self.targets = targets
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.resize = resize
        self.default_aug = albumentations.Compose(
            [
                albumentations.Normalize(
                    mean, std, max_pixel_value=255.0, always_apply=True
                ),
                albumentations.Resize(self.resize, self.resize),
            ]
        )
        self.transform = transform

    def __getitem__(self, index: int):
        image = Image.open(self.image_paths[index])
        label = self.targets[index]

        if self.transform:
            image = self.transform(image)

        augmented_image = self.default_aug(image=np.array(image))

        image = np.transpose(augmented_image["image"], (2, 0, 1)).astype(np.float32)

        return (
            torch.tensor(image, dtype=torch.float),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self):
        return len(self.image_paths)


class CVDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        indices: np.ndarray,
        image_paths,
        target_cols,
        transform=None,
        resize=224,
    ):
        self.df = df
        self.indices = indices
        self.transform = transform
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.resize = resize
        self.default_aug = albumentations.Compose(
            [
                albumentations.Normalize(
                    mean, std, max_pixel_value=255.0, always_apply=True
                ),
                albumentations.Resize(self.resize, self.resize),
            ]
        )
        self.image_paths = image_paths
        self.target_cols = target_cols

    def __getitem__(self, idx: int):
        image_ids = operator.itemgetter(*self.indices)(self.df[self.image_paths])
        labels = operator.itemgetter(*self.indices)(self.df[self.target_cols])

        image = Image.open(image_ids[idx])
        label = torch.tensor(labels[idx], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        augmented_image = self.default_aug(image=np.array(image))

        image = np.transpose(augmented_image["image"], (2, 0, 1)).astype(np.float32)

        return (
            torch.tensor(image, dtype=torch.float),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self):
        return len(self.indices)
