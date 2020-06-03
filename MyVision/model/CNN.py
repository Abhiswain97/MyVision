import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class NeuralNet(nn.Module):
    def __init__(self, output_features):
        super(NeuralNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        model_head_features = self.model.fc.in_features
        self.model.fc = nn.Linear(
            in_features=model_head_features, out_features=output_features
        )

    def forward(self, x):
        return self.model(x)
