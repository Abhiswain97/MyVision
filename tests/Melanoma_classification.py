import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from itertools import chain

from MyVision.dataset import Dataset
from MyVision.dataloader import DataLoader
from MyVision.engine import Engine
from MyVision.model import CNN

import torch.nn as nn
import torch


def main(args):
    train_df = pd.read_csv(args.csv_file)

    model = CNN.NeuralNet(output_features=1)
    loss = nn.BCELoss()

    train_tfms = A.Compose(
        [
            A.RandomResizedCrop(224, 224),
            A.Flip(always_apply=True),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_tfms = A.Compose(
        [
            A.RandomResizedCrop(224, 224),
            A.Flip(always_apply=False),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    if args.folds:

        best_loss = 100
        for fold, (train_idx, val_idx) in enumerate(
            StratifiedKFold(n_splits=args.folds, random_state=42).split(
                train_df[args.image_path_column], train_df[args.image_label_column]
            )
        ):
            for epoch in args.epochs:

                for phase in ["train", "valid"]:

                    print(f"[FOLD {fold}] [EPOCH {epoch}] [PHASE {phase}]")
                    (train_dataset, valid_dataset,) = Dataset.make_dataset(
                        is_CV=True,
                        train_df=train_df,
                        train_idx=train_idx,
                        val_idx=val_idx,
                        image_path_column=args.image_path_column,
                        image_label_column=args.image_label_column,
                        train_tfms=train_tfms,
                        val_tfms=val_tfms,
                    )

                    train_loader = DataLoader.SimpleDataLoader(
                        train_dataset, batch_size=args.batch_size
                    )
                    val_loader = DataLoader.SimpleDataLoader(
                        valid_dataset, batch_size=args.batch_size
                    )

                    engine = Engine.Engine(
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=None,
                        device=args.device,
                        loss=loss,
                        model=model,
                    )

                    if phase == "train":
                        engine.train()

                    if phase == "valid":
                        preds, avg_loss = engine.valid()

                        if avg_loss < best_loss:
                            torch.save(args.model.state_dict(), args.checkpoint_path)
    else:

        for epoch in args.epochs:

            for phase in ["train", "valid"]:
                print(f"[EPOCH {epoch}] [PHASE {phase}]")

                (train_dataset, valid_dataset,) = Dataset.make_dataset(
                    is_CV=True,
                    train_df=train_df,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    image_path_column=args.image_path_column,
                    image_label_column=args.image_label_column,
                )

                engine = Engine.Engine(
                    train_loader=args.train_loader,
                    val_loader=args.val_loader,
                    test_loader=None,
                    device=args.device,
                    loss=loss,
                    model=model,
                    train_tfms=train_tfms,
                    val_tfms=val_tfms,
                )

                if phase == "train":
                    engine.train()

                elif phase == "valid":
                    preds, avg_loss = engine.valid()

                    if avg_loss < best_loss:
                        torch.save(args.model.state_dict(), args.checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--model", help="Name of the model", default="resnet-18")
    parser.add_argument("--batch_size", help="Batch size", default=16, required=True)
    parser.add_argument("--lr", help="Learning rate", default=0.01)
    parser.add_argument("--momentum", help="Momentum for optimizer", default=0.09)
    parser.add_argument("--weight_decay", help="Weight decay", default=0.01)
    parser.add_argument(
        "--folds",
        help="No of folds for cross-validation(When specified will automatically use CVDataset)",
        default=3,
    )
    parser.add_argument("--epochs", help="Training epochs", default=1)
    parser.add_argument(
        "--device", help="Device name", choices=["cuda", "cpu"], default="cuda"
    )
    parser.add_argument("--csv_file", help="Features and label csv file", required=True)
    parser.add_argument(
        "--image_path_column", help="column containing image paths", required=True
    )
    parser.add_argument("--image_label_column", help="Target column", required=True)
    parser.add_argument(
        "--checkpoint_path", help="Path to save model", required=True, default="."
    )

    arguments = parser.parse_args()
    main(arguments)
