import torch.nn as nn
import torchvision.models as models


# TODO: this will be useful for CAM
class ResNet(nn.Module):
    def __init__(self, weights=models.ResNet50_Weights.DEFAULT, n_classes: int=37):
        super().__init__()
        self.backbone = models.resnet50(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)
