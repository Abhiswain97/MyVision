import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile

import numpy as np
import pandas as pd

import operator

ImageFile.LOAD_TRUNCATED_IMAGES = True


def make_dataset(
    is_CV,
    train_df,
    train_idx,
    val_idx,
    image_path_column,
    image_label_column,
    train_tfms,
    valid_tfms
):
    if is_CV:
        train_dataset = CVDataset(
            train_df, train_idx, image_path_column, image_label_column, transform=train_tfms
        )

        valid_dataset = CVDataset(
            train_df, val_idx, image_path_column, image_label_column, transform=valid_tfms
        )
    else:
        train_dataset = Dataset.SimpleDataset(
            image_path_column, image_label_column, transform=train_tfms
        )

        valid_dataset = Dataset.SimpleDataset(
            image_path_column, image_label_column, transform=valid_tfms
        )

    return train_dataset, valid_dataset


class SimpleDataset(Dataset):
    def __init__(self, image_paths, targets, transform=None):
        self.image_paths = image_paths
        self.targets = targets

    def __getitem__(self, index: int):
        image = Image.open(self.image_paths[index])
        target = self.targets[index]

        if self.transform:
            image = self.transform(image)

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
        self.image_paths = image_paths
        self.target_cols = target_cols

    def __getitem__(self, idx: int):
        image_ids = operator.itemgetter(*self.indices)(self.df[[self.image_paths]])
        labels = operator.itemgetter(*self.indices)(self.df[[self.target_cols]])

        image = Image.open(image_ids[idx])
        label = torch.tensor(labels[idx])

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.indices)
