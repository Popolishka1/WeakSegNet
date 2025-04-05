import torch
import torch.nn as nn
import torchvision.models as models

# TODO (all): find a way to refine cams to get better pseudo masks (dense CRF for instance)

class CAMGenerator:
    def __init__(self, classifier):
        self.classifier = classifier
        self.feature_maps = None
        
        # Identify architecture type (backbone)
        backbone = getattr(classifier, "backbone", None)
        if backbone is None:
            raise ValueError("Classifier must have a `backbone` attribute. Please check classfication.py.")

        # Register hook and get the FC weights (be careful with the architecture!!!)
        if isinstance(backbone, models.ResNet):
            # Works for ResNet50 and ResNet101
            self.hook = backbone.layer4.register_forward_hook(self.hook_feature)
            self.fc_weights = backbone.fc.weight

        elif isinstance(backbone, models.DenseNet):
            self.hook = backbone.features.register_forward_hook(self.hook_feature)
            self.fc_weights = backbone.classifier.weight

        else:
            raise NotImplementedError(f"Backbone architecture {type(backbone)} not supported for CAM. Please check classification.py.")


    def hook_feature(self, module, input, output):
        self.feature_maps = output.detach()


    def __del__(self):
        if hasattr(self, "hook"):
            self.hook.remove()


def generate_cam(cam_generator, input_image, target_class=None):
    """Generate CAM for a given image."""
    # Forward pass
    logits = cam_generator.classifier(input_image.unsqueeze(0))
    if target_class is None:
        target_class = logits.argmax(dim=1).item()
    
    # Get weights from final FC layer
    weights = cam_generator.fc_weights[target_class]
    
    # Get feature maps from the hook
    features = cam_generator.feature_maps.squeeze(0)

    # Compute CAM
    cam = torch.matmul(weights, features.view(features.shape[0], -1))
    cam = cam.view(features.shape[1], features.shape[2])

    # Normalize CAM
    cam = nn.functional.relu(cam)
    eps = 1e-7
    cam = (cam - cam.min()) / (cam.max() - cam.min() + eps)

    # Resize to match input image
    cam = nn.functional.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=input_image.shape[1:],
        mode='bilinear',
        align_corners=False
    ).squeeze()

    return cam, target_class

def generate_grad_cam(cam_generator, input_image, target_class=None):
    """Generate Class Activation Map for a given image using Grad-CAM."""
    
    model = cam_generator.classifier
    model.eval()

    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output
        output.retain_grad()

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    # Register hooks on the target conv layer
    handle_fwd = model.backbone.layer4.register_forward_hook(forward_hook)
    handle_bwd = model.backbone.layer4.register_full_backward_hook(backward_hook)

    input_tensor = input_image.unsqueeze(0).requires_grad_()

    # Forward
    logits = model(input_tensor)
    if target_class is None:
        target_class = logits.argmax(dim=1).item()

    # Backward
    model.zero_grad()
    logits[0, target_class].backward()

    # Clean up hooks and get data
    handle_fwd.remove()
    handle_bwd.remove()
    fmap = activations["value"].squeeze(0)
    grad = gradients["value"].squeeze(0)   

    # Compute alpha_k^c (mean gradient per channel)
    weights = grad.mean(dim=(1, 2)).clamp(min=0)

    # Compute Grad-CAM
    cam = torch.sum(weights[:, None, None] * fmap, dim=0)
    cam = nn.functional.relu(cam)

    # Reshape and normalize
    cam = cam - cam.min()
    cam = cam / cam.max() if cam.max() > 0 else cam
    
    # Upsample to input size
    cam = nn.functional.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=input_image.shape[1:],
        mode='bilinear',
        align_corners=False
    ).squeeze()
    
    return cam, target_class

def generate_grad_cam_pp(cam_generator, input_image, target_class=None):
    """Generate Class Activation Map for a given image using Grad-CAM++."""
    
    model = cam_generator.classifier
    model.eval()

    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output
        output.retain_grad()

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    # Register hooks on the target conv layer
    handle_fwd = model.backbone.layer4.register_forward_hook(forward_hook)
    handle_bwd = model.backbone.layer4.register_full_backward_hook(backward_hook)

    input_tensor = input_image.unsqueeze(0).requires_grad_()

    # Forward
    logits = model(input_tensor)
    if target_class is None:
        target_class = logits.argmax(dim=1).item()

    # Backward
    model.zero_grad()
    logits[0, target_class].backward()

    # Clean up hooks and get data
    handle_fwd.remove()
    handle_bwd.remove()
    fmap = activations["value"].squeeze(0)
    grad = gradients["value"].squeeze(0)   

    # Compute alpha_k^c (mean gradient per channel)
    eps = 1e-8
    grad_relu = grad.clamp(min=0)
    sum_grad_fmap = (grad_relu * fmap).sum(dim=(1, 2), keepdim=True)
    alpha = grad_relu.pow(2) / (2 * grad_relu.pow(2) + sum_grad_fmap + eps)
    weights = (alpha * grad_relu).sum(dim=(1, 2))

    # Compute Grad-CAM++
    cam = torch.sum(weights[:, None, None] * fmap, dim=0)
    cam = nn.functional.relu(cam)

    # Reshape and normalize
    cam = cam - cam.min()
    cam = cam / cam.max() if cam.max() > 0 else cam
    
    # Upsample to input size
    cam = nn.functional.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=input_image.shape[1:],
        mode='bilinear',
        align_corners=False
    ).squeeze()
    
    return cam, target_class

def cam_to_binary_mask(cam, cam_threshold: float=0.5):
    """
    Convert CAM to binary mask using threshold.
    CAM already normalized in generate_cam function.
    """
    binary_mask = (cam > cam_threshold).float()
    return binary_mask.unsqueeze(0)


def generate_pseudo_masks(classifier, dataloader, cam_threshold=0.5, device='cpu', model='CAM'):
    """Generate pseudo masks for an entire dataset using CAM."""
    classifier.eval()
    cam_generator = CAMGenerator(classifier)
    pseudo_masks = []
    if model=='CAM':
        generate_cam_func = generate_cam
    elif model=='Grad-CAM':
        generate_cam_func = generate_grad_cam
    elif model=='Grad-CAM++':
        generate_cam_func = generate_grad_cam_pp
    else:
        raise ValueError("Invalid model type. Choose 'CAM' 'Grad-CAM' or 'Grad-CAM++.")
    
    for images, _, _ in dataloader:
        images = images.to(device)
        for img in images:
            if model=='CAM':
                with torch.no_grad():
                    cam, _ = generate_cam_func(cam_generator, img)
            else:
                with torch.enable_grad():
                    cam, _ = generate_cam_func(cam_generator, img)
            
            with torch.no_grad():
                pseudo_mask = cam_to_binary_mask(cam, cam_threshold)
                pseudo_masks.append(pseudo_mask.cpu())
    
    return pseudo_masks