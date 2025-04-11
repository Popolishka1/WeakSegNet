import torch
import torch.nn as nn
import torchvision.models as models

# TODO (INPROGRESS): find a way to refine cams to get better pseudo masks (dense CRF for instance)
import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as crf_utils

def apply_dense_crf(img_tensor, cam):
    """
    Refine the CAM using Dense CRF.

    Args:
        img_tensor: torch.Tensor, image tensor (C, H, W) in [0,1]
        cam: torch.Tensor, CAM probability map (H, W) in [0,1]
    
    Returns:
        refined_cam: torch.Tensor, refined probability map (H, W)
    """
    # Convert the input image to numpy array and ensure it's C-contiguous
    img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
    img_uint8 = np.ascontiguousarray((img_np * 255).astype(np.uint8))
    
    # Get the CAM as a numpy array (H, W)
    prob_map = cam.cpu().numpy()
    H, W = prob_map.shape

    # Set up the CRF model
    d = dcrf.DenseCRF2D(W, H, 2)
    
    # Create unary energy from the probability map for two classes (background and foreground)
    probs = np.stack([1 - prob_map, prob_map], axis=0)
    U = crf_utils.unary_from_softmax(probs)
    d.setUnaryEnergy(U.astype(np.float32))

    # Add pairwise Gaussian smoothing term (spatial proximity)
    d.addPairwiseGaussian(sxy=3, compat=3)
    # Add pairwise bilateral term (color and spatial consistency)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img_uint8, compat=10)

    # Run inference for 5 iterations
    Q = d.inference(5)
    refined_prob = np.array(Q)[1].reshape((H, W))
    
    # Return the refined probability as a torch tensor
    refined_cam = torch.tensor(refined_prob, dtype=torch.float32)
    return refined_cam


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
        generate_cam_func = lambda cam_generator, input_image, target_class=None: generate_grad_cam_common(cam_generator, input_image, target_class, variant=cam_type)
    else:
        raise ValueError("Invalid model type. Choose CAM', 'Grad-CAM', or 'Grad-CAM++'.")

    return cam_generator, generate_cam_func, cam_type



def generate_pseudo_masks_(dataloader, cam_generator, generate_cam_func, cam_threshold, device, use_dense_crf=False):
    """Internal helper to loop through dataloader and generate CAM-based pseudo masks."""
    
    if use_dense_crf:
        print("Refining pseudo masks with Dense CRF...")

    pseudo_masks = []
    for images, _, _ in dataloader:
        images = images.to(device)
        for img in images:
            cam, _ = generate_cam_func(cam_generator, img)
            if use_dense_crf:
                cam = apply_dense_crf(img, cam)
            pseudo_mask = cam_to_binary_mask(cam, cam_threshold)
            pseudo_masks.append(pseudo_mask.cpu())
    return pseudo_masks


# IN PROGRESS: might be better to use DenseCRF to refine the masks
def generate_pseudo_masks(dataloader, classifier, cam_type='CAM', cam_threshold=0.5, device='cpu', use_dense_crf=False):
    cam_generator, generate_cam_func, model_type = get_cam_generator(classifier=classifier, cam_type=cam_type)

    if model_type == 'CAM':
        with torch.no_grad():
            return generate_pseudo_masks_(dataloader=dataloader,
                                          cam_generator=cam_generator,
                                          generate_cam_func=generate_cam_func, 
                                          cam_threshold=cam_threshold,
                                          device=device,
                                          use_dense_crf=use_dense_crf)
    else:
        with torch.enable_grad():
            return generate_pseudo_masks_(dataloader=dataloader,
                                          cam_generator=cam_generator,
                                          generate_cam_func=generate_cam_func,
                                          cam_threshold=cam_threshold,
                                          device=device,
                                          use_dense_crf=use_dense_crf)
