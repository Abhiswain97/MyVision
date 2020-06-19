from tqdm import tqdm
import torch
import numpy as np

from tabulate import tabulate

from ..utils import Meters, LabelUtils, csvlogger, ModelUtils
from .. import metrics

import os
import sys
import time
import logging
import csv
from itertools import chain
from abc import ABC, abstractmethod


class Trainer(object):
    # if not os.path.exists("logs"):
    #     os.mkdir("logs")
    #
    # # creating log file
    # logger = logging.getLogger("Engine-logs")
    # hdlr = logging.FileHandler(f"logs\\{timestampEND}.log", mode="w")
    # formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    # hdlr.setFormatter(formatter)
    # logger.addHandler(hdlr)
    # logger.setLevel(logging.INFO)

    @staticmethod
    def train(
        train_loader,
        device,
        criterion,
        optimizer,
        model,
        accumulation_steps=1,
        use_mixup=True,
    ):
        losses = Meters.AverageMeter("Loss", ":.4e")

        model.train()

        tl = tqdm(train_loader, file=sys.stdout)
        for batch_idx, (images, targets) in enumerate(tl, 1):

            optimizer.zero_grad()

            if use_mixup:
                images, targets_a, targets_b, lam = ModelUtils.MixUp.mixup_data(
                    images, targets, 1.0, device=device
                )
            else:
                images = images.to(device)
                targets = targets.to(device)

            outputs = model(images)

            if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                if is_mixup:
                    loss = ModelUtils.MixUp.mixup_criterion(
                        criterion,
                        outputs,
                        targets_a.unsqueeze(1),
                        targets_b.unsqueeze(1),
                        lam,
                    )
                else:
                    loss = criterion(outputs, targets.unsqueeze(1).float())
            else:
                if is_mixup:
                    loss = ModelUtils.MixUp.mixup_criterion(
                        criterion,
                        outputs,
                        targets_a,
                        targets_b,
                        lam,
                    )
                else:
                    loss = criterion(outputs, targets)

            loss.backward()

            if batch_idx % accumulation_steps == 0:
                optimizer.step()

            losses.update(val=loss.item(), n=images.size(0))

        return losses.avg

    @staticmethod
    def validate(val_loader, device, criterion, optimizer, model):
        losses = Meters.AverageMeter("Loss", ":.4e")

        model.eval()

        predictions = []
        gts = []

        vl = tqdm(val_loader, file=sys.stdout)
        for batch_idx, (images, targets) in enumerate(vl, 1):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)

            if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                loss = criterion(outputs, targets.unsqueeze(1).float())
            else:
                loss = criterion(outputs, targets)

            predictions = chain(predictions, outputs.detach().cpu().numpy())
            gts = chain(gts, targets.detach().cpu().numpy())

            losses.update(val=loss.item(), n=images.size(0))

        return np.array(list(predictions)), np.array(list(gts)), losses.avg

    @staticmethod
    def fit(
        train_loader,
        val_loader,
        device,
        criterion,
        optimizer,
        model,
        epochs,
        metric_name="accuracy",
        lr_scheduler=None,
        checkpoint_path="models",
        accumulation_steps=1,
        csv_logger=True,
    ):

        best_loss = 1
        table_list = []

        train_start_time = time.time()

        csv_logs = csvlogger.CSVLogger() if csv_logger else None

        train_metrics = []

        for epoch in range(epochs):

            start_time = time.time()

            train_loss = Trainer.train(
                train_loader, device, criterion, optimizer, model, accumulation_steps
            )

            with torch.no_grad():
                preds, gts, valid_loss = Trainer.validate(
                    val_loader, device, criterion, optimizer, model
                )

            if valid_loss < best_loss:

                if not os.path.exists(checkpoint_path):
                    os.mkdir(checkpoint_path)

                timestampTime = time.strftime("%H%M%S")
                timestampDate = time.strftime("%d%m%Y")
                timestampEND = timestampDate + "-" + timestampTime

                print(f"[SAVING] to {checkpoint_path}\\model-[{timestampEND}].pt")
                torch.save(
                    {
                        "model": model,
                        "state_dict": model.state_dict(),
                        "best_loss": best_loss,
                        "best_epoch": epoch,
                    },
                    f"{checkpoint_path}\\model-[{timestampEND}].pt",
                )

                best_loss = valid_loss

            pred_labels = LabelUtils.probs_to_labels(preds)

            defined_metrics = metrics.ClassificationMetrics().metrics

            given_metric = metrics.ClassificationMetrics()

            if metric_name == "auc":

                score = (
                    given_metric(
                        metric=metric_name, y_true=gts, y_pred=None, y_proba=preds
                    )
                    if metric_name in defined_metrics
                    else None
                )
            else:
                score = (
                    given_metric(metric=metric_name, y_true=gts, y_pred=pred_labels)
                    if metric_name in defined_metrics
                    else None
                )

            # log the metrics to a csv files
            if csv_logger:
                train_metrics.append(
                    [
                        epoch + 1,
                        round(train_loss, 3),
                        round(valid_loss, 3),
                        round(score, 3),
                    ],
                )
                csv_logs(metrics=train_metrics, metric_name=metric_name)

            if score:
                table_list.append(
                    (
                        epoch + 1,
                        round(train_loss, 3),
                        round(valid_loss, 3),
                        round(score, 3),
                    )
                )
            else:
                table_list.append(
                    (epoch + 1, round(train_loss, 3), round(valid_loss, 3))
                )

            print(
                tabulate(
                    table_list,
                    headers=("Epoch", "Train loss", "Validation loss", metric_name),
                    tablefmt="pretty",
                )
            )

            if lr_scheduler:
                lr_scheduler.step(score)

            print(f"Epoch completed in: {(time.time() - start_time) / 60} mins")

        print(f"Training completed in {(time.time() - train_start_time) / 60} mins")


class CustomTrainer(ABC):
    """
    Abstract base class for defining your own custom training and validation steps,
    basically your own `Trainer`.
    """

    def __init__(
        self,
        train_loader,
        val_loader,
        device,
        criterion,
        optimizer,
        model,
        epochs,
        metric_name,
        lr_scheduler=None,
        checkpoint_path="models",
        accumulation_steps=1,
    ):
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.model = model
        self.lr_scheduler = lr_scheduler
        self.metric_name = metric_name
        self.checkpoint_path = checkpoint_path
        self.accumulation_steps = accumulation_steps

    @staticmethod
    def train():
        """
        This method defines the custom training step you define.

        :returns:
            a dictionary containing the:
                train_loss: Average training loss after one epoch.
        """

        return {"avg_train_loss": train_loss}

    @staticmethod
    def validate():
        """
        This method defines the custom training step you define.

        :returns:
            a dictionary containing the folowing:

                preds: the predicted outputs from the model.
                targets: ground truths(original targets).
                avg_val_loss: Average validation loss
        """
        return {"preds": preds, "targets": gts, "avg_val_loss": valid_loss}

    @staticmethod
    def fit(self, epochs, metric=None, checkpoint_path="models"):

        best_loss = 1
        table_list = []

        for epoch in range(epochs):

            train_loss = self.train().values()

            preds, gts, valid_loss = self.validate().values()

            if valid_loss < best_loss:

                if not os.path.exists(checkpoint_path):
                    os.mkdir(checkpoint_path)

                print(f"[SAVING] to {checkpoint_path}\\best_model-({epoch}).pth.tar")
                torch.save(
                    self.model.state_dict(),
                    f"{checkpoint_path}\\best_model-({epoch}).pth.tar",
                )

            pred_labels = LabelUtils.probs_to_labels(preds)

            score = metric(pred_labels, preds) if metric else None

            if score:
                table_list.append((epoch, train_loss, valid_loss, score))
            else:
                table_list.append((epoch, train_loss, valid_loss))

            print(
                tabulate(
                    table_list,
                    headers=("Epoch", "Train loss", "Validation loss", metric),
                )
            )

            if self.lr_scheduler:
                if metric is not None:
                    self.lr_scheduler.step(score)
                else:
                    self.lr_scheduler.step(valid_loss)
