from tqdm import tqdm
import torch
from ..utils import Meters
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
        accumulation_steps=1,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.loss = loss
        self.optimizer = optimizer
        self.model = model
        self.accumulation_steps = accumulation_steps

    def train(self):
        batch_time = Meters.AverageMeter("Time", ":6.3f")
        losses = Meters.AverageMeter("Loss", ":.4e")

        self.model.train()

        end = time.time()

        tl = tqdm(self.train_loader, 1)
        for batch_idx, (images, targets) in tl:

            self.optimizer.zero_grad()

            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(images)

            loss = self.loss(outputs, targets)
            loss.backward()

            losses.update(val=loss.item(), n=images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % self.accumulation_steps == 0:
                self.optimizer.step()

        return losses.avg

    def valid(self):
        losses = Meters.AverageMeter("Loss", ":.4e")

        self.model.eval()

        predictions = []
        vl = tqdm(self.val_loader, 1)
        for batch_idx, (images, targets) in vl:
            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(images)

            loss = self.loss(outputs, targets)

            losses.update(val=loss.item(), n=images.size(0))

            predictions = chain(predictions, outputs.cpu().numpy())

        return list(predictions), losses.avg

    def test(self):
        losses = Meters.AverageMeter("Loss", ":.4e")

        self.model.eval()

        tl = tqdm(self.test_loader, 1)
        for batch_idx, images in tl:
            images = images.to(self.device)

            outputs = self.model(images)

            loss = self.loss(outputs, targets)
            losses.update(val=loss.item(), n=images.size(0))
