from tqdm import tqdm
import torch
from tabulate import tabulate

from ..utils import Meters
from MyVision import metrics

import os
import time
from itertools import chain


class Engine:
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
        self.loss = loss
        self.optimizer = optimizer
        self.model = model
        self.lr_scheduler = lr_scheduler
        self.accumulation_steps = accumulation_steps

    def train(self):

        losses = Meters.AverageMeter("Loss", ":.4e")

        self.model.train()

        tl = tqdm(self.train_loader, 1)
        for batch_idx, (images, targets) in tl:

            self.optimizer.zero_grad()

            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(images)

            loss = self.loss(outputs, targets)
            loss.backward()

            losses.update(val=loss.item(), n=images.size(0))

            if batch_idx % self.accumulation_steps == 0:
                self.optimizer.step()

        return losses.avg

    @torch.no_grad()
    def valid(self):
        losses = Meters.AverageMeter("Loss", ":.4e")

        self.model.eval()

        predictions, gts = []

        vl = tqdm(self.val_loader, 1)
        for batch_idx, (images, targets) in vl:
            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(images)

            loss = self.loss(outputs, targets)

            losses.update(val=loss.item(), n=images.size(0))

            predictions = chain(predictions, outputs.cpu().numpy())
            gts = chain(gts, targets.detach().cpu().numpy())

        return list(predictions), list(gts), losses.avg

    @torch.no_grad()
    def test(self):

        self.model.eval()

        predictions = []

        tl = tqdm(self.test_loader, 1)
        for batch_idx, images in tl:
            images = images.to(self.device)

            outputs = self.model(images)

            predictions = chain(predictions, outputs.cpu().numpy())

        return list(predictions)

    def fit(self, epochs, metric):

        best_loss = 1
        table_list = []

        for epoch in range(epochs):

            train_loss = self.train()

            preds, gts, valid_loss = self.valid()

            if valid_loss < best_loss:

                if not os.path.exists(models):
                    os.mkdir(models)
                print('[SAVING].....')
                torch.save(self.model.state_dict(), f'models\\best_model-({epoch}).pth.tar')

            score = metrics.ClassificationMetrics()(metric, y_true=gts, y_pred=None, y_proba=preds)
            table_list.append((epoch, train_loss, valid_loss, score))

            print(tabulate(table_list, headers=('Epoch', 'Train loss', 'Valid loss', metric)))

            self.lr_scheduler.step(score)
