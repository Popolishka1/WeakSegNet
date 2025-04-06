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
            raise ValueError("Classifier must have a backbone attribute. Please check classfication.py.")

        self.backbone = backbone

        # Register hook and get the FC weights (be careful with the architecture!!!)
        if isinstance(backbone, models.ResNet):
            # Works for ResNet50 and ResNet101
            self.target_layer = backbone.layer4
            self.hook = backbone.layer4.register_forward_hook(self.hook_feature)
            self.fc_weights = backbone.fc.weight

        elif isinstance(backbone, models.DenseNet):
            # Works for DenseNet121
            self.target_layer = backbone.features
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

    model = cam_generator.classifier
    model.eval()

    # Prepare input
    input_tensor = input_image.unsqueeze(0)

    # Forward pass
    logits = model(input_tensor)
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
    cam = cam - cam.min()
    cam = cam / cam.max() if cam.max() > 0 else cam

    # Resize to match input image
    cam = nn.functional.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=input_image.shape[1:],
        mode='bilinear',
        align_corners=False
    ).squeeze()

    return cam, target_class


def generate_grad_cam_common(cam_generator, input_image, target_class=None, variant="Grad-CAM"):
    """Generate CAM for a given image using Grad-CAM or Grad-CAM++."""

    assert variant in ["Grad-CAM", "Grad-CAM++"], "Variant must be 'Grad-CAM' or 'Grad-CAM++'"

    model = cam_generator.classifier
    model.eval()

    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output
        output.retain_grad()

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    # Register hooks on the stored target layer
    handle_fwd = cam_generator.target_layer.register_forward_hook(forward_hook)
    handle_bwd = cam_generator.target_layer.register_full_backward_hook(backward_hook)

    # Prepare input
    input_tensor = input_image.unsqueeze(0).requires_grad_()

    # Forward pass
    logits = model(input_tensor)
    if target_class is None:
        target_class = logits.argmax(dim=1).item()

    # Backward pass
    model.zero_grad()
    logits[0, target_class].backward()

    # Remove hooks
    handle_fwd.remove()
    handle_bwd.remove()

    # Get activations and gradients
    fmap = activations["value"].squeeze(0)
    grad = gradients["value"].squeeze(0)

    if variant == "Grad-CAM":
        weights = grad.mean(dim=(1, 2)).clamp(min=0) # compute weights

    elif variant == "Grad-CAM++":
        eps = 1e-8
        grad_relu = grad.clamp(min=0)
        sum_grad_fmap = (grad_relu * fmap).sum(dim=(1, 2), keepdim=True)
        alpha = grad_relu.pow(2) / (2 * grad_relu.pow(2) + sum_grad_fmap + eps)
        weights = (alpha * grad_relu).sum(dim=(1, 2)) # compute weights

    # Compute CAM
    cam = torch.sum(weights[:, None, None] * fmap, dim=0)

    # Normalize CAM
    cam = nn.functional.relu(cam)
    cam = cam - cam.min()
    cam = cam / cam.max() if cam.max() > 0 else cam

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
    CAM already normalized in the chosen cam generation function.
    """
    binary_mask = (cam > cam_threshold).float()
    return binary_mask.unsqueeze(0)


def get_cam_generator(classifier, cam_type="CAM"):
    """Return the CAM generator and generation function based on the model type."""
    classifier.eval()
    cam_generator = CAMGenerator(classifier)

    if cam_type == "CAM":
        generate_cam_func = generate_cam
        
    elif cam_type in ["Grad-CAM", "Grad-CAM++"]:
        generate_cam_func = lambda cam_gen, img, target_class=None: generate_grad_cam_common(cam_gen, img, target_class, variant=cam_type)
    else:
        raise ValueError("Invalid model type. Choose CAM', 'Grad-CAM', or 'Grad-CAM++'.")

    return cam_generator, generate_cam_func, cam_type


def generate_pseudo_masks_(dataloader, cam_generator, generate_cam_func, cam_threshold, device):
    """Internal helper to loop through dataloader and generate CAM-based pseudo masks."""
    pseudo_masks = []
    for images, _, _ in dataloader:
        images = images.to(device)
        for img in images:
            cam, _ = generate_cam_func(cam_generator, img)
            pseudo_mask = cam_to_binary_mask(cam, cam_threshold)
            pseudo_masks.append(pseudo_mask.cpu())
    return pseudo_masks


def generate_pseudo_masks(dataloader, classifier, cam_type='CAM', cam_threshold=0.5, device='cpu'):
    """Generate pseudo masks using a provided CAM generator and method."""

    cam_generator, generate_cam_func, model_type = get_cam_generator(classifier=classifier, cam_type=cam_type)

    if model_type == 'CAM':
        with torch.no_grad():
            return generate_pseudo_masks_(dataloader=dataloader,
                                          cam_generator=cam_generator,
                                          generate_cam_func=generate_cam_func, 
                                          cam_threshold=cam_threshold,
                                          device=device
                                          )
    else:
        with torch.enable_grad():
            return generate_pseudo_masks_(dataloader=dataloader,
                                          cam_generator=cam_generator,
                                          generate_cam_func=generate_cam_func,
                                          cam_threshold=cam_threshold,
                                          device=device
                                          )