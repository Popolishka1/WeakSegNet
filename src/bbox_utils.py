import torch
import re
import cv2
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from src.dataset import inverse_normalize, data_loading, load_data_wrapper, PseudoMaskDataset
import os
from src.utils import parse_args

##########################################
##         EVALUATION FUNCTIONS         ##
##########################################

def dice_score(output, gt_mask, threshold=0.5, eps=1e-7):
    if not isinstance(output, torch.Tensor):
        output = torch.from_numpy(output)
    if not isinstance(gt_mask, torch.Tensor):
        gt_mask = torch.from_numpy(gt_mask)

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
    if output.ndim == 2:
        output = output.unsqueeze(0).unsqueeze(0)
    if gt_mask.ndim == 2:
        gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
    predicted_mask = (output > threshold).float()
    gt_mask = gt_mask.float()

    tp = (predicted_mask * gt_mask).sum()
    predicted_positive = predicted_mask.sum()
    actual_positive = gt_mask.sum()

    precision = (tp + eps) / (predicted_positive + eps)
    recall = (tp + eps) / (actual_positive + eps)

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

##########################################
##      GENERATING MASKS FUNCTIONS      ##
##########################################
def generate_pseudo_mask(bbox, image, variant="GrabCut"):
    x_min, y_min, x_max, y_max = bbox
    image_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
    
    if variant == "GrabCut":
        grabcut_mask = np.zeros(image_uint8.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(image_uint8, grabcut_mask, (x_min, y_min, x_max - x_min, y_max - y_min), bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((grabcut_mask == 2) | (grabcut_mask == 0), 0, 1).astype('uint8')
        pseudo_mask = (mask2 * 255).astype(np.uint8)
    
    elif variant == "Super-Pixel":
        image_cv = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
        image_cv = cv2.GaussianBlur(image_cv, (5, 5), sigmaX=1, sigmaY=1)
        h, w = image_cv.shape[:2]
        region_size = int(np.sqrt((h * w) / 200))
        slic = cv2.ximgproc.createSuperpixelSLIC(image_cv, cv2.ximgproc.SLIC, region_size=region_size, ruler=10)
        slic.iterate(10)
        segments = slic.getLabels()
        pseudo_mask = np.zeros_like(segments, dtype=np.uint8)
        for label in np.unique(segments):
            sp_mask = (segments == label)
            coords = np.column_stack(np.where(sp_mask))
            if len(coords) == 0:
                continue
            y_center, x_center = coords.mean(axis=0)
            if (x_center >= x_min and x_center <= x_max and y_center >= y_min and y_center <= y_max):
                pseudo_mask[sp_mask] = 1
        pseudo_mask = (pseudo_mask * 255).astype(np.uint8)
    
    return pseudo_mask

def generate_pseudo_masks(bboxs, images, variant="GrabCut"):
    return [generate_pseudo_mask(bbox, inverse_normalize(image).permute(1, 2, 0).cpu().numpy(), variant=variant) for bbox, image in zip(bboxs, images)]

def generate_bbox(cam, image):
    
    image_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
    
    # Create a binary mask by thresholding the CAM
    threshold = 0.5  # Adjust this threshold as needed (between 0 and 1)
    binary_mask = cam > threshold
    coords = np.column_stack(np.where(binary_mask))
    if coords.size != 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        return (x_min, y_min, x_max, y_max)  
    else:
        return None
    
def generate_bboxs(cams, images):
    return [generate_bbox(cam, inverse_normalize(image).permute(1, 2, 0).cpu().numpy()) for cam, image in zip(cams, images)]

def generate_cam(image):
    image_uint8 = (image * 255).clip(0, 255).astype(np.uint8)

    # Load a pre-trained model
    model = models.resnet50(pretrained=True)
    model.eval()

    # Choose the target convolutional layer (typically the last conv layer)
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
    preprocess = transforms.Compose([
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
    cam = torch.zeros(activations.shape[2:], dtype=torch.float32)
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

##########################################
##  VISUALISE AND SAVE MASKS FUNCTIONS  ##
##########################################

def visualise_results(image, bbox, mask, pseudo_mask, variant="GrabCut"):
    image_np = inverse_normalize(image).permute(1, 2, 0).cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()
    pseudo_mask_np = pseudo_mask.squeeze().cpu().numpy() if isinstance(pseudo_mask, torch.Tensor) else pseudo_mask

    img_pil = Image.fromarray((image_np * 255).clip(0, 255).astype(np.uint8))
    mask_pil = Image.fromarray((mask_np * 255).clip(0, 255).astype(np.uint8))
    pseudo_mask_pil = Image.fromarray((pseudo_mask_np * 255).clip(0, 255).astype(np.uint8))

    draw = ImageDraw.Draw(img_pil)
    x_min, y_min, x_max, y_max = bbox
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)

    total_width = img_pil.width * 3
    max_height = max(img_pil.height, mask_pil.height, pseudo_mask_pil.height)
    composite = Image.new("RGB", (total_width, max_height + 30), "white")  # +30 for text
    
    fonts_folder = os.path.join(os.environ["WINDIR"], "Fonts") if os.name == 'nt' else "/usr/share/fonts"
    try:
        font = ImageFont.truetype(os.path.join(fonts_folder, "arial.ttf"), 14)
    except:
        font = ImageFont.load_default()

    # Image with bounding box
    composite.paste(img_pil, (0, 0))
    draw = ImageDraw.Draw(composite)
    draw.text((img_pil.width//2 - 40, img_pil.height + 5), 
             "Image with BBox", fill="black", font=font)

    # Ground truth mask
    composite.paste(mask_pil, (img_pil.width, 0))
    draw.text((img_pil.width + mask_pil.width//2 - 50, mask_pil.height + 5), 
             "Ground Truth Mask", fill="black", font=font)

    # Pseudo mask
    composite.paste(pseudo_mask_pil, (img_pil.width*2, 0))
    draw.text((img_pil.width*2 + pseudo_mask_pil.width//2 - 40, pseudo_mask_pil.height + 5), 
             f"{variant} Mask", fill="black", font=font)

    composite.show()

def save_pseudo_masks(pseudo_masks, batch_idx, output_dir):
    for idx, pseudo_mask in enumerate(pseudo_masks):
        if isinstance(pseudo_mask, torch.Tensor):
            pseudo_mask_np = pseudo_mask.cpu().numpy()
        else:
            pseudo_mask_np = pseudo_mask

        pseudo_mask_np = np.squeeze(pseudo_mask_np)
        
        if pseudo_mask_np.max() <= 1.0:
            pseudo_mask_np = (pseudo_mask_np * 255).astype(np.uint8)
        else:
            pseudo_mask_np = pseudo_mask_np.astype(np.uint8)
        
        file_name = os.path.join(output_dir, f"pseudo_mask_batch{batch_idx}_idx{idx}.png")
        Image.fromarray(pseudo_mask_np).save(file_name)
    print(f'Images saved to file')


##########################################
##       TRAINING MODEL FUNCTIONS       ##
##########################################

def natural_keys(text):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', text)]

def load_pseudo_mask(data_dir):
    pseudo_masks = []
    files = sorted(os.listdir(data_dir), key=natural_keys)
    for file in files:
        file_path = os.path.join(data_dir, file)
        print("Loading:", file)
        im_frame = np.array(Image.open(file_path))
        
        if im_frame.ndim == 2:
            im_frame = np.expand_dims(im_frame, axis=0)
            
        im_frame = im_frame.astype(np.float32)
        
        if im_frame.max() > 1.0:
            im_frame = im_frame / 255.0
        
        mask_tensor = torch.tensor(im_frame, dtype=torch.float32)
        pseudo_masks.append(mask_tensor)
    return pseudo_masks

def mix_pseudo_masks_exp(data_dir1="./grab_cut_thres_0.3_grad_cam", data_dir2="./super_pixel/"):
    # Define the directories for your two sets of masks
    # Load the pseudo masks from each directory
    pseudo_masks1 = load_pseudo_mask(data_dir1)
    pseudo_masks2 = load_pseudo_mask(data_dir2)

    # Ensure that both folders have the same number of masks
    if len(pseudo_masks1) != len(pseudo_masks2):
        raise ValueError("Folders contain a different number of masks; please check alignment.")

    # Define a threshold for binarisation (adjust as needed)
    threshold = 0.5

    combined_masks = []
    for mask1, mask2 in zip(pseudo_masks1, pseudo_masks2):
        # Binarise each mask based on the threshold
        binary_mask1 = mask1 > threshold
        binary_mask2 = mask2 > threshold
        
        # Compute the element-wise logical AND to keep only the overlapping (common) regions
        overlapping_region = torch.logical_and(binary_mask1, binary_mask2)
        
        # Optionally, convert back to float format if needed (0.0 for background, 1.0 for overlapping region)
        combined_mask = overlapping_region.float()
        
        combined_masks.append(combined_mask)

    batch_size = 64
    batched_combined_masks = []

    # Loop over combined_masks in steps of batch_size
    for i in range(0, len(combined_masks), batch_size):
        # Take a slice of combined_masks of length up to batch_size
        batch = combined_masks[i : i + batch_size]
        
        # Optionally, you can stack the batch of masks into a tensor.
        # This assumes that each mask has the same shape.
        batch_tensor = torch.stack(batch, dim=0)
        
        batched_combined_masks.append(batch_tensor)
    return batched_combined_masks
    # Now batched_combined_masks is a list, where each element is a tensor of shape (batch_size, ...)


##########################################
##     GENERATE DATASET FUNCTIONS     ##
##########################################

def generate_pseudo_mask_dataset_with_bbox(output_dir="saved_pseudo_masks", batch_size=64, train_loader, variant="GrabCut"):
    
    config = parse_args(expriment_name="BBOX")
     = load_data_wrapper(config=config)    

    total_dice = 0.0
    total_accuracy = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    n_batches = len(train_loader)
     
    os.makedirs(output_dir, exist_ok=True)
    for batch_idx, (image_batch, mask_batch, info_batch) in enumerate(train_loader, start=1):
        print(f"Batch number: {batch_idx}")
        cams = generate_cams(image_batch)
        bboxs = generate_bboxs(cams, image_batch)
        pseudo_masks = generate_pseudo_masks(bboxs, image_batch, variant=variant)
        dice, accuracy_score, iou, precision, recall = evaluate(pseudo_masks, mask_batch, batch_size)
        print(f'Dice score is {dice} || Pixel score is {accuracy_score} || IOU score is {iou} || Precision score is {precision} || Recall score is {recall}')
        total_dice += dice
        total_accuracy += accuracy_score
        total_iou += iou
        total_precision += precision
        total_recall += recall
        visualise_results(image_batch[0], bboxs[0], mask_batch[0], pseudo_masks[0], variant=variant)
        save_pseudo_masks(pseudo_masks, batch_idx, output_dir)

    print(f'Average dice score per image is {total_dice / n_batches}')
    print(f'Average accuracy score per image is {total_accuracy / n_batches}')
    print(f'Average IOU score per image is {total_iou / n_batches}')
    print(f'Average precision score per image is {total_precision / n_batches}')
    print(f'Average recall score per image is {total_recall / n_batches}')
