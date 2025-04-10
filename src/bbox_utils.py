import os
import torch
import cv2
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.segmentation import slic
from skimage.util import img_as_float

def dice_score(output, gt_mask, threshold=0.5, eps=1e-7):
    """
    One of the possible metric that we can use: dice score between predicted masks and ground truth.
    """
    if not isinstance(output, torch.Tensor):
        output = torch.from_numpy(output)
    if not isinstance(gt_mask, torch.Tensor):
        gt_mask = torch.from_numpy(gt_mask)

    # If the tensors are 2D (e.g., H x W), unsqueeze to add batch and channel dimensions.
    if output.ndim == 2:
        output = output.unsqueeze(0).unsqueeze(0)
    if gt_mask.ndim == 2:
        gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
    predicted_mask = (output > threshold).float()
    gt_mask = gt_mask.float()

    intersection = (predicted_mask * gt_mask).sum()
    union = predicted_mask.sum() + gt_mask.sum()
    dice = (2 * intersection + eps) / (union + eps)
    return dice.item()

def iou_score(output, gt_mask, threshold=0.5, eps=1e-7):
    if not isinstance(output, torch.Tensor):
        output = torch.from_numpy(output)
    if not isinstance(gt_mask, torch.Tensor):
        gt_mask = torch.from_numpy(gt_mask)

    # If the tensors are 2D (e.g., H x W), unsqueeze to add batch and channel dimensions.
    if output.ndim == 2:
        output = output.unsqueeze(0).unsqueeze(0)
    if gt_mask.ndim == 2:
        gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
    predicted_mask = (output > threshold).float()
    gt_mask = gt_mask.float()

    intersection = (predicted_mask * gt_mask).sum()
    union = predicted_mask.sum() + gt_mask.sum() - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.item()

def pixel_accuracy(output, gt_mask, threshold=0.5):
    if not isinstance(output, torch.Tensor):
        output = torch.from_numpy(output)
    if not isinstance(gt_mask, torch.Tensor):
        gt_mask = torch.from_numpy(gt_mask)

    # If the tensors are 2D (e.g., H x W), unsqueeze to add batch and channel dimensions.
    if output.ndim == 2:
        output = output.unsqueeze(0).unsqueeze(0)
    if gt_mask.ndim == 2:
        gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
    predicted_mask = (output > threshold).float()
    gt_mask = gt_mask.float()

    correct = (predicted_mask == gt_mask).float()

    total = torch.numel(gt_mask)
    accuracy = correct.sum() / total
    return accuracy.item()


def precision_recall(output, gt_mask, threshold=0.5, eps=1e-7):
    if not isinstance(output, torch.Tensor):
        output = torch.from_numpy(output)
    if not isinstance(gt_mask, torch.Tensor):
        gt_mask = torch.from_numpy(gt_mask)

    # If the tensors are 2D (e.g., H x W), unsqueeze to add batch and channel dimensions.
    if output.ndim == 2:
        output = output.unsqueeze(0).unsqueeze(0)
    if gt_mask.ndim == 2:
        gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
    predicted_mask = (output > threshold).float()
    gt_mask = gt_mask.float()

    tp = (predicted_mask * gt_mask).sum()
    predicted_pitive = predicted_mask.sum()
    actual_pitive = gt_mask.sum()

    precision = (tp + eps) / (predicted_pitive + eps)
    recall = (tp + eps) / (actual_pitive + eps)

    return precision.item(), recall.item()

def eval_pseudo_masks_dice(pseudo_masks, masks):
    return sum([dice_score(pseudo_mask, true_mask.squeeze()) for pseudo_mask, true_mask in zip(pseudo_masks, masks)])

def eval_pseudo_masks_pixel_accuracy(pseudo_masks, masks):
    return sum([pixel_accuracy(pseudo_mask, true_mask.squeeze()) for pseudo_mask, true_mask in zip(pseudo_masks, masks)])

def eval_pseudo_masks_iou(pseudo_masks, masks):
    return sum([iou_score(pseudo_mask, true_mask.squeeze()) for pseudo_mask, true_mask in zip(pseudo_masks, masks)])

def eval_pseudo_masks_precision_recall(pseudo_masks, masks):
    precisions_recalls = [precision_recall(pseudo_mask, true_mask.squeeze()) for pseudo_mask, true_mask in zip(pseudo_masks, masks)]
    precisions = [precision_recall[0] for precision_recall in precisions_recalls]
    recalls = [precision_recall[1] for precision_recall in precisions_recalls]
    return sum(precisions), sum(recalls)

def evaluate(pseudo_masks, mask_batch, batch_size):
    dice = eval_pseudo_masks_dice(pseudo_masks, mask_batch) / batch_size
    accuracy_score = eval_pseudo_masks_pixel_accuracy(pseudo_masks, mask_batch) / batch_size
    iou = eval_pseudo_masks_iou(pseudo_masks, mask_batch) / batch_size
    precision, recall = eval_pseudo_masks_precision_recall(pseudo_masks, mask_batch) 
    return dice, accuracy_score, iou, precision / batch_size, recall / batch_size

def generate_pseudo_mask(bbox, image, variant="GrabCut"):
    x_min = bbox[0]
    y_min = bbox[1]
    x_max = bbox[2]
    y_max = bbox[3]
    image_uint8 = (image * 255).clip(0, 255).astype(np.uint8)

    if variant == "GrabCut":
       
        image_uint8 = (image * 255).clip(0, 255).astype(np.uint8)

        # Initialize the mask for GrabCut
        grabcut_mask = np.zer(image_uint8.shape[:2], np.uint8)

        # Initialize background and foreground models (required by GrabCut)
        bgdModel = np.zer((1, 65), np.float64)
        fgdModel = np.zer((1, 65), np.float64)

        # Apply GrabCut with the bounding box initialization
        cv2.grabCut(image_uint8, grabcut_mask, bbox, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        # Convert GrabCut output to binary mask: 0 and 2 are background, 1 and 3 are foreground
        mask2 = np.where((grabcut_mask == 2) | (grabcut_mask == 0), 0, 1).astype('uint8')

        result = image_uint8 * mask2[:, :, np.newaxis]

        pseudo_mask = (mask2 * 255).astype(np.uint8)
    
    elif variant == "Super-Pixel":
        
        image_float = img_as_float(image)

        # Compute superpixels; adjust n_segments and compactness as needed
        segments = slic(image_float, n_segments=200, compactness=10, sigma=1, start_label=0)

        # Create an empty pseudo mask with the same size as segments
        pseudo_mask = np.zer_like(segments, dtype=np.uint8)

        # Iterate over each unique superpixel label
        for label in np.unique(segments):
            sp_mask = segments == label
            coords = np.column_stack(np.nonzero(sp_mask))
            y_center, x_center = coords.mean(axis=0)  # (y, x) centroid of the superpixel
            # If the centroid is within the bounding box, mark the whole superpixel as foreground
            if x_center >= x_min and x_center <= x_max and y_center >= y_min and y_center <= y_max:
                pseudo_mask[sp_mask] = 1

    return pseudo_mask

def generate_pseudo_masks(bboxs, images, variant="GrabCut"):
    return [generate_pseudo_mask(bbox, inverse_normalize(image).permute(1, 2, 0).cpu().numpy(), variant=variant) for bbox, image in zip(bboxs, images)]

def generate_bbox(cam, image):
    
    image_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
    
    # Create a binary mask by thresholding the CAM
    threshold = 0.5  # Adjust this threshold as needed (between 0 and 1)
    binary_mask = cam > threshold
    # Compute bounding box from the binary mask using numpy
    coords = np.column_stack(np.where(binary_mask))
    if coords.size != 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        return (x_min, y_min, x_max, y_max)  # (left, top, right, bottom)
    else:
        return None
    

def generate_bboxs(cams, images):
    return [generate_bbox(cam, inverse_normalize(image).permute(1, 2, 0).cpu().numpy()) for cam, image in zip(cams, images)]

def generate_cam(image):
    image_uint8 = (image * 255).clip(0, 255).astype(np.uint8)

    # Load a pre-trained model (ResNet50 in this example) and set to eval mode.
    model = models.resnet50(pretrained=True)
    model.eval()

    # Choe the target convolutional layer (typically the last conv layer)
    target_layer = model.layer4[-1]

    # Global variables to store activations and gradients
    activations = None
    gradients = None

    # Hook to capture forward activations
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()

    # Hook to capture gradients during backpropagation
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()

    # Register hooks on the target layer
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    # Define image preprocessing pipeline
    preprocess = transforms.Compe([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    pil_img = Image.fromarray(image_uint8)

    # Preprocess the image to get a tensor with shape [1, C, H, W]
    input_tensor = preprocess(pil_img).unsqueeze(0)
    # Forward pass
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()  # Get predicted class index

    # Backward pass: compute gradients for the predicted class score
    model.zero_grad()
    output[0, pred_class].backward()
    # Compute Grad-CAM
    # Global average pooling over gradients for each channel
    weights = torch.mean(gradients, dim=(2, 3))  # Shape: [batch, channels]

    # Combine the activations using these weights
    cam = torch.zer(activations.shape[2:], dtype=torch.float32)
    for i, w in enumerate(weights[0]):
        cam += w * activations[0, i, :, :]

    # Apply ReLU and normalize the CAM
    cam = F.relu(cam)
    cam = cam - cam.min()
    if cam.max() != 0:
        cam = cam / cam.max()

    # Upsample the CAM to the original image size
    cam = cam.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
    cam = F.interpolate(cam, size=(pil_img.size[1], pil_img.size[0]), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    return cam

def generate_cams(images):
    return [generate_cam(inverse_normalize(image).permute(1, 2, 0).cpu().numpy()) for image in images]

def visualise_results(image, bbox, mask, pseudo_mask, variant="GrabCut"):
    image = inverse_normalize(image).permute(1, 2, 0).cpu().numpy()
    mask = mask.squeeze().cpu().numpy()
    x_min = bbox[0]
    y_min = bbox[1]
    x_max = bbox[2]
    y_max = bbox[3]
    plt.figure(figsize=(15, 5))

    # 1. Plot image with bounding box
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                            linewidth=2, edgecolor='red', facecolor='none')
    plt.gca().add_patch(rect)
    plt.title("Image with Bounding Box")
    plt.axis('off')

    # 2. Plot the ground truth mask
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis('off')

    # Plot the pseudo mask
    plt.subplot(1, 3, 3)
    plt.imshow(pseudo_mask, cmap="gray")
    plt.title(f"Pseudo Mask {variant}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def save_pseudo_masks(pseudo_masks, batch_idx, output_dir):
    for idx, pseudo_mask in enumerate(pseudo_masks):
        # Convert a torch tensor to a NumPy array if needed
        if isinstance(pseudo_mask, torch.Tensor):
            pseudo_mask_np = pseudo_mask.cpu().numpy()
        else:
            pseudo_mask_np = pseudo_mask

        # Squeeze dimensions if necessary (for example, from shape [1, H, W] to [H, W])
        pseudo_mask_np = np.squeeze(pseudo_mask_np)
        
        # If the mask values are floats in [0, 1], convert them to 0-255 and to unsigned 8-bit integers
        if pseudo_mask_np.max() <= 1.0:
            pseudo_mask_np = (pseudo_mask_np * 255).astype(np.uint8)
        else:
            pseudo_mask_np = pseudo_mask_np.astype(np.uint8)
        
        # Construct a filename with the .png extension
        file_name = os.path.join(output_dir, f"pseudo_mask_batch{batch_idx}_idx{idx}.png")
        # Save the image using Pillow in PNG format
        Image.fromarray(pseudo_mask_np).save(file_name)
    print(f'Images saved to file')
