import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import albumentations as A

import operator

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DatasetUtils:
    """
    This class contains utilities for making a Pytorch Dataset.
    It takes in: 
        a pd.Dataframe, 
        column containing image paths, 
        column having labels,
        train transforms,
        validation transforms
    & makes two datasets one for Training and the other for Validation.   
    """
    def __init__(
        self, train_df, image_path_column, target_column, train_tfms, valid_tfms
    ):
        self.train_df = train_df
        self.image_path_column = image_path_column
        self.target_column = target_column
        self.train_tfms = train_tfms
        self.val_tfms = valid_tfms

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

    @staticmethod
    def make_dataset(
        train_df,
        image_path_column,
        label_column,
        train_idx,
        val_idx,
        train_transforms=self.train_tfms,
        valid_transforms=self.valid_tfms,
        valid_size=0.25,
        is_CV=None,
    ):
        if is_CV:
            train_dataset = CVDataset(
                train_df,
                train_idx,
                image_path_column,
                label_column,
                transform=train_transforms,
            )

            valid_dataset = CVDataset(
                train_df,
                val_idx,
                image_path_column,
                label_column,
                transform=valid_transforms,
            )
        else:

            train_image_paths, train_labels, valid_image_paths, valid_labels = splitter(
                valid_size=valid_size
            )

            train_dataset = Dataset.SimpleDataset(
                train_image_paths, train_labels, transform=train_transforms
            )

            valid_dataset = Dataset.SimpleDataset(
                valid_image_paths, valid_labels, transform=valid_transforms
            )

        return train_dataset, valid_dataset


class SimpleDataset(Dataset):
    def __init__(self, image_paths, targets, transform=None):
        self.image_paths = image_paths
        self.targets = targets
        self.defaut_tfms = A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __getitem__(self, index: int):
        image = Image.open(self.image_paths[index])
        target = self.targets[index]

        if self.transform:
            image = self.transform(image)

        image = self.defaut_tfms(image)

        target = torch.tensor(target)

        return image, target

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
    ):
        self.df = df
        self.indices = indices
        self.transform = transform
        self.defaut_tfms = A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.image_paths = image_paths
        self.target_cols = target_cols

    def __getitem__(self, idx: int):
        image_ids = operator.itemgetter(*self.indices)(self.df[[self.image_paths]])
        labels = operator.itemgetter(*self.indices)(self.df[[self.target_cols]])

        image = Image.open(image_ids[idx])
        label = torch.tensor(labels[idx])

        if self.transform:
            image = self.transform(image)

        image = self.defaut_tfms(image)

        return image, label

    def __len__(self):
        return len(self.indices)
