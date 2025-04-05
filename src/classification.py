import warnings
import torch.nn as nn
import torchvision.models as models


class ResNet50(nn.Module):
    def __init__(self, weights=models.ResNet50_Weights.DEFAULT, n_classes: int=37):
        super().__init__()
        self.backbone = models.resnet50(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)


class ResNet101(nn.Module):
    def __init__(self, weights=models.ResNet101_Weights.DEFAULT, n_classes: int = 37):
        super(ResNet101, self).__init__()
        self.backbone = models.resnet101(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, n_classes)
        
    def forward(self, x):
        return self.backbone(x)


class DenseNet121(nn.Module):
    def __init__(self, weights=models.DenseNet121_Weights.DEFAULT, n_classes: int = 37):
        super(DenseNet121, self).__init__()
        self.backbone = models.densenet121(weights=weights)
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, n_classes)
        
    def forward(self, x):
        return self.backbone(x)


def select_classifier(model_name, n_classes):
    if model_name == "ResNet50":
        return ResNet50(n_classes=n_classes)
    elif model_name == "ResNet101":
        return ResNet101(n_classes=n_classes)
    elif model_name == "DenseNet121":
        return DenseNet121(n_classes=n_classes)
    else:
        warnings.warn("Incorrect classifier name or model not implemented. Please check the model name in classification.py.")
        return None