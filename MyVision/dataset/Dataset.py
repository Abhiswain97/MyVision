import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import albumentations
from albumentations.pytorch import ToTensorV2

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
                train_images.values,
                train_labels.values,
                valid_images.values,
                valid_labels.values,
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
        self.default_aug = albumentations.Compose(
            [
                albumentations.Normalize(),
                ToTensorV2()
            ]
        ) 
        self.transform = transform

    def __getitem__(self, index: int):
        image = Image.open(self.image_paths[index])
        label = self.targets[index]

        if self.transform:
            image = self.transform(image)

        augmented_image = self.default_aug(image=np.array(image))

        return (
            torch.tensor(augmented_image['image']),
            torch.tensor(label),
        )

    def __len__(self):
        return len(self.image_paths)


class PascalVOCDataset(Dataset):
    def __init__(self, df, image_paths, boxes, targets, transform=None):
        self.df = df
        self.transform = transform
        self.image_paths = image_paths
        self.boxes = boxes
        self.targets = targets
        self.default_aug = albumentations.Normalize()
        
    def __getitem__(self, idx):
        image = Image.open(self.df[self.image_paths][idx])
        box = self.df[self.boxes].astype(np.float32)[idx]
        class_ = self.df[self.targets].astype(np.float32)[idx]
        
        if self.transform:
            image = self.transform(image)

        augmented_image = self.default_aug(image=np.array(image))
    
        box = torch.as_tensor(box, dtype=torch.float32)
        class_ = torch.as_tensor(class_, dtype=torch.float32)
        
        return {
            "image": image,
            "box": box,
            "class": class_
        }
    
    def __len__(self):
        return len(self.df)


class CVDataset(Dataset):
    """
    This class provides utilities for doing cross-validation, using a PyTorch Dataset.
    """

    def __init__(self, df, indices, image_paths, targets, transform=None):
        self.df = df
        self.indices = indices
        self.transform = transform
        self.image_paths = image_paths
        self.targets = targets
        self.default_aug = albumentations.Compose(
            [
                albumentations.Normalize(),
                ToTensorV2()
            ]
        )

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

        return (
            torch.tensor(augmented_image['image']),
            torch.tensor(label),
        )

    def __len__(self):
        return len(self.indices)
