# TODO: idea 1 - use the xlms boxes and turn them into a mask (so we get a weak traget) - will have to refine/create the data class...

# TODO: idea 2 - use the class ids (to do class activation maps). still unsure how. need more info. idea: train a classifier on the id and generate a psuedo mask


import torch
import torch.nn as nn
import torchvision.models.segmentation as models

##############################################################################################
# I took the first baseline from: https://arc-celt.github.io/pet-segmentation/ (thanks Carl) #
##############################################################################################


# DoubleConv Block
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# UNet model (encoder-decoder architecture) # TODO: variation of the UNet? See UNet++
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))
        dec4 = self.dec4(torch.cat([self.up4(bottleneck), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.up3(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))
        return torch.sigmoid(self.out_conv(dec1))


class DeepLabV3(nn.Module):
    def __init__(self, weights=models.DeepLabV3_ResNet50_Weights.DEFAULT):
        super().__init__()
        self.deeplab = models.deeplabv3_resnet50(weights=weights)
        self.deeplab.classifier[4] = nn.Conv2d(256, 1, kernel_size=1) # small change here: we output one channel ie the mask

    def forward(self, x):
        out = self.deeplab(x)['out']
        return torch.sigmoid(out) # sigmoid bc we want probabs
    

# Fully convolutional model 
class FCN(nn.Module):
    def __init__(self, weights=models.FCN_ResNet50_Weights.DEFAULT):
        super().__init__()
        self.fcn = models.fcn_resnet50(weights=weights)
        self.fcn.classifier[4] = nn.Conv2d(512, 1, kernel_size=1) # same change here: output one channel ie the mask
    
    def forward(self, x):
        out = self.fcn(x)['out']
        return torch.sigmoid(out) # probas
    

class PetClassifier(nn.Module): # TODO: simplify the main.py file to call the classifier (CE loss)
    def __init__(self, num_classes=37):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)