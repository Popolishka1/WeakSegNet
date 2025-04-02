import torch
import torch.nn.functional as F
import numpy as np

class CAMGenerator:
    def __init__(self, classifier):
        self.classifier = classifier
        self.feature_maps = None
        self.hook = self.classifier.backbone.layer4.register_forward_hook(
            self.hook_feature
        )
    
    def hook_feature(self, module, input, output):
        self.feature_maps = output.detach()
    
    def __del__(self):
        self.hook.remove()

def generate_cam(cam_generator, input_image, target_class=None):
    """Generate Class Activation Map for a given image."""
    # Forward pass
    logits = cam_generator.classifier(input_image.unsqueeze(0))
    if target_class is None:
        target_class = logits.argmax(dim=1).item()
    
    # Get weights from final FC layer
    weights = cam_generator.classifier.backbone.fc.weight[target_class]
    
    # Compute CAM
    features = cam_generator.feature_maps.squeeze()
    cam = torch.matmul(weights, features.view(weights.shape[0], -1))
    
    # Reshape and normalize
    h, w = features.shape[1], features.shape[2]
    cam = cam.view(h, w)
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / cam.max() if cam.max() > 0 else cam
    
    # Upsample to input size
    cam = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=input_image.shape[1:],
        mode='bilinear',
        align_corners=False
    ).squeeze()
    
    return cam, target_class

def cam_to_binary_mask(cam, cam_threshold: float=0.5):
    """Convert CAM to binary mask using threshold."""
    eps = 1e-8
    cam_normalized = (cam - cam.min()) / (cam.max() - cam.min() + eps)
    binary_mask = (cam_normalized > cam_threshold).float()
    return binary_mask.unsqueeze(0)  # expected size[1, H, W] TODO: check it

def generate_pseudo_masks(classifier, dataloader, cam_threshold=0.5, device='cpu'):
    """Generate pseudo masks for an entire dataset using CAM."""
    classifier.eval()
    cam_generator = CAMGenerator(classifier)
    pseudo_masks = []
    
    with torch.no_grad():
        for images, _, _ in dataloader:
            images = images.to(device)
            for img in images:
                cam, _ = generate_cam(cam_generator, img)
                pseudo_mask = cam_to_binary_mask(cam, cam_threshold)
                pseudo_masks.append(pseudo_mask.cpu())
    
    return pseudo_masks