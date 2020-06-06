from tqdm import tqdm
import torch
import numpy as np

from tabulate import tabulate

from ..utils import Meters
from MyVision import metrics

import os
import time
from itertools import chain
import abc


class Trainer:
    def __init__(
        self,
        train_loader,
        val_loader,
        test_loader,
        device,
        loss,
        optimizer,
        model,
        lr_scheduler,
        accumulation_steps=1,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = loss
        self.optimizer = optimizer
        self.model = model
        self.lr_scheduler = lr_scheduler
        self.accumulation_steps = accumulation_steps

    def train(self):
        losses = Meters.AverageMeter("Loss", ":.4e")

        self.model.train()

        tl = tqdm(self.train_loader)
        for batch_idx, (images, targets) in enumerate(tl, 1):

            self.optimizer.zero_grad()

            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(images)

            loss = self.criterion(outputs, targets)

            loss.backward()

            if batch_idx % self.accumulation_steps == 0:
                self.optimizer.step()

            losses.update(val=loss.item(), n=images.size(0))

        return losses.avg

    def validate(self):
        losses = Meters.AverageMeter("Loss", ":.4e")

        self.model.eval()

        predictions = []
        gts = []

        vl = tqdm(self.val_loader)
        for batch_idx, (images, targets) in enumerate(vl, 1):
            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(images)

            loss = self.criterion(outputs, targets)

            predictions = chain(predictions, outputs.detach().cpu().numpy())
            gts = chain(gts, targets.detach().cpu().numpy())

            losses.update(val=loss.item(), n=images.size(0))

        return (np.array(list(predictions)), np.array(list(gts)), losses.avg)

    def fit(self, epochs, metric):

        best_loss = 1
        table_list = []

        for epoch in range(epochs):

            train_loss = self.train()

            preds, gts, valid_loss = self.validate()

            if valid_loss < best_loss:

                if not os.path.exists("models"):
                    os.mkdir("models")
                print("[SAVING].....")
                torch.save(
                    self.model.state_dict(), f"models\\best_model-({epoch}).pth.tar"
                )

            if len(np.unique(preds)) > 2:

                multiclass_metrics = ["accuracy"]

                preds = [np.argmax(p) for p in preds]

                score = metrics.ClassificationMetrics()(
                    metric, y_true=gts, y_pred=preds, y_proba=None
                )
            else:
                binary_metrics = ["auc", "f1", "recall", "precision"]

                preds = [1 if p >= 0.5 else 0 for p in preds]

                score = metrics.ClassificationMetrics()(
                    metric, y_true=gts, y_pred=preds, y_proba=None
                )

            table_list.append((epoch, train_loss, valid_loss, score))

            print(
                tabulate(
                    table_list,
                    headers=("Epoch", "Train loss", "Validation loss", metric),
                )
            )

            if self.lr_scheduler:
                self.lr_scheduler.step(score)
