import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import albumentations

import operator

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DatasetUtils(object):
    """
    This class contains utilities for making a PyTorch Dataset.
    """

    @staticmethod
    def splitter(train_df=None, image_paths=None, targets=None, valid_size=0.25):

        if isinstance(train_df, pd.DataFrame):
            train_images, valid_images, train_labels, valid_labels = train_test_split(
                train_df[image_paths],
                train_df[targets],
                test_size=valid_size,
                random_state=42,
            )

            return (
                train_images.values.tolist(),
                train_labels.values.tolist(),
                valid_images.values.tolist(),
                valid_labels.values.tolist(),
            )

        else:
            train_images, valid_images, train_labels, valid_labels = train_test_split(
                image_paths, targets, test_size=valid_size, random_state=42,
            )

            return (
                train_images,
                train_labels,
                valid_images,
                valid_labels,
            )

    @staticmethod
    def make_dataset(
        train_df=None,
        image_paths=None,
        targets=None,
        train_tfms=None,
        valid_tfms=None,
        train_idx=None,
        val_idx=None,
        valid_size=0.25,
        is_CV=False,
        split=True,
    ):
        if is_CV:
            train_dataset = CVDataset(
                df=train_df,
                indices=train_idx,
                image_paths=image_paths,
                targets=targets,
                transform=train_tfms,
            )

            valid_dataset = CVDataset(
                df=train_df,
                indices=val_idx,
                image_paths=image_paths,
                targets=targets,
                transform=valid_tfms,
            )
        else:

            if split:

                (
                    train_image_paths,
                    train_labels,
                    valid_image_paths,
                    valid_labels,
                ) = DatasetUtils.splitter(
                    train_df=train_df,
                    image_paths=image_paths,
                    targets=targets,
                    valid_size=valid_size,
                )

                train_dataset = SimpleDataset(
                    image_paths=train_image_paths,
                    targets=train_labels,
                    transform=train_tfms,
                )

                valid_dataset = SimpleDataset(
                    image_paths=valid_image_paths,
                    targets=valid_labels,
                    transform=valid_tfms,
                )

                return train_dataset, valid_dataset

            else:

                train_dataset = SimpleDataset(
                    image_paths=image_paths, targets=targets, transform=train_tfms
                )

                return train_dataset


class SimpleDataset(Dataset):
    def __init__(self, image_paths, targets, transform=None):
        self.image_paths = image_paths
        self.targets = targets
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.default_aug = albumentations.Compose(
            [
                albumentations.Normalize(
                    mean, std, max_pixel_value=255.0, always_apply=True
                ),
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
            torch.tensor(image),
            torch.tensor(label),
        )

    def __len__(self):
        return len(self.image_paths)


class CVDataset(Dataset):
    """
    This class provides utilities for doing cross-validation, using a PyTorch Dataset.
    """

    def __init__(self, df, indices, image_paths, targets, transform=None):
        self.df = df
        self.indices = indices
        self.transform = transform
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.default_aug = albumentations.Compose(
            [
                albumentations.Normalize(
                    mean, std, max_pixel_value=255.0, always_apply=True
                )
            ]
        )
        self.image_paths = image_paths
        self.targets = targets

    def __getitem__(self, idx):

        if isinstance(self.df, pd.DataFrame):
            image_ids = operator.itemgetter(*self.indices)(self.df[self.image_paths])
            labels = operator.itemgetter(*self.indices)(self.df[self.target_cols])
        else:
            image_ids = operator.itemgetter(*self.indices)(self.image_paths)
            labels = operator.itemgetter(*self.indices)(self.targets)

        image = Image.open(image_ids[idx])
        label = labels[idx]

        if self.transform:
            image = self.transform(image)

        augmented_image = self.default_aug(image=np.array(image))

        image = np.transpose(augmented_image["image"], (2, 0, 1)).astype(np.float32)

        return (
            torch.tensor(image),
            torch.tensor(label),
        )

    def __len__(self):
        return len(self.indices)
