import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def generate_cam_mask(classifier, image_tensor, target_class, threshold=0.2, use_cuda=True):
    target_layers = [classifier.backbone.layer4[-1]]
    
    cam = GradCAM(model=classifier, target_layers=target_layers, use_cuda=use_cuda)
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]

    binary_mask = (grayscale_cam > threshold).astype(np.uint8)
    return binary_mask


def generate_weak_mask_from_cam(classifier, image_pil, target_class, target_transform, threshold=0.2, device="cuda"):

    image_tensor = target_transform(image_pil).unsqueeze(0).to(device)  

    binary_mask = generate_cam_mask(classifier, image_tensor, target_class, threshold=threshold, use_cuda=(device=="cuda"))

    mask_pil = Image.fromarray(binary_mask * 255) 
    weak_mask_tensor = target_transform(mask_pil)
    return weak_mask_tensor
