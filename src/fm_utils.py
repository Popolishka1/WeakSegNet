import numpy as np
import torch
import argparse
import numpy as np
from src.cam_utils import get_cam_generator, generate_pseudo_masks
from src.dataset import load_data_wrapper, PseudoMaskDataset

class SAMWrapper:
    def __init__(self, predictor, device):
        self.predictor = predictor
        self.device = device

    def __call__(self, images):
        """
        Processes a batch of images using the SAM2 predictor.
        Expects images as a batch of tensors (C, H, W). Internally loops per image.
        """
        outputs = []
        for image in images:
            # Convert tensor (C,H,W) to a NumPy image (H,W,C) and scale to 0-255.
            image_np = image.permute(1, 2, 0).cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            self.predictor.set_image(image_np)
            
            # Define a weak prompt: use the image center dynamically.
            H, W, _ = image_np.shape
            input_point = np.array([[W // 2, H // 2]])  # Note the order: (x, y)
            input_label = np.array([1])
            
            masks, _, _ = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )
            
            outputs.append(torch.from_numpy(masks[0]).unsqueeze(0).float().to(self.device))
        return torch.stack(outputs, dim=0)

    def eval(self):
        return self


class SAMWrapper:
    def __init__(self, predictor, device):
        self.predictor = predictor
        self.device = device

    def __call__(self, images,  multi = True):
        """
        Processes a batch of images using the SAM2 predictor.
        Expects images as a batch of tensors (C, H, W). Internally loops per image.
        """
        outputs = []
        for image in images:
            # Convert tensor (C,H,W) to a NumPy image (H,W,C) and scale to 0-255.
            image_np = image.permute(1, 2, 0).cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            self.predictor.set_image(image_np)
            
            # Define a weak prompt: use the image center dynamically.
            H, W, _ = image_np.shape
            input_point = np.array([[W // 2, H // 2]])  # Note the order: (x, y)
            input_label = np.array([1])
            

            # Get multiple mask hypotheses from SAM
            masks, _, _ = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=multi
            )
            
            # Select the largest mask to address undermasking
            mask_sizes = [mask.sum() for mask in masks]
            best_mask_idx = np.argmax(mask_sizes)
            
            outputs.append(torch.from_numpy(masks[best_mask_idx]).unsqueeze(0).float().to(self.device))
        return torch.stack(outputs, dim=0)

    def eval(self):
        return self



class SAMWrapperWithCAM:
    def __init__(self, predictor, pseudo_masks, num_points=3, device="cuda"):
        self.predictor = predictor
        self.pseudo_masks = pseudo_masks
        self.num_points = num_points
        self.device = device

    def __call__(self, images, multi = True):
        outputs = []
        
        # Loop through batch
        for i, image in enumerate(images):
            # Convert tensor to NumPy image format for SAM
            image_np = image.permute(1, 2, 0).cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)

            # Set image for SAM predictor
            self.predictor.set_image(image_np)
            
            # Get the corresponding CAM
            cam = self.pseudo_masks[i].squeeze().cpu().numpy()
            
            # Find peak points in CAM instead of just top-k values
            input_points = self._find_peak_points(cam)
            input_labels = np.ones(len(input_points))

            # Get multiple mask hypotheses from SAM
            masks, _, _ = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=multi
                #box=boxes
            )
            
            # Select the largest mask to address undermasking
            mask_sizes = [mask.sum() for mask in masks]
            best_mask_idx = np.argmax(mask_sizes)
            
            outputs.append(torch.from_numpy(masks[best_mask_idx]).unsqueeze(0).float().to(self.device))

        return torch.stack(outputs, dim=0)
    
    def _find_peak_points(self, cam):
        """Find peak points in CAM using simple local maximum detection"""
        # Apply basic threshold to avoid low activations
        threshold = 0.3 * cam.max()
        binary = cam > threshold
        
        # Simple peak detection (center of mass if no distinct peaks)
        if not np.any(binary):
            # If nothing above threshold, use single highest point
            y_max, x_max = np.unravel_index(np.argmax(cam), cam.shape)
            return np.array([[x_max, y_max]])
        
        # Get at least one point (highest activation)
        points = []
        y_max, x_max = np.unravel_index(np.argmax(cam), cam.shape)
        points.append([x_max, y_max])
        
        # Get additional peaks if requested
        if self.num_points > 1:
            # Simple approach: get top-k points that are not too close to existing points
            min_distance = 10  # Minimum distance between peaks
            flat_indices = np.argsort(cam.flatten())[::-1]  # Descending order
            
            for idx in flat_indices:
                if len(points) >= self.num_points:
                    break
                    
                y, x = np.unravel_index(idx, cam.shape)
                
                # Check if this point is far enough from existing points
                too_close = False
                for px, py in points:
                    if np.sqrt((x - px)**2 + (y - py)**2) < min_distance:
                        too_close = True
                        break
                
                if not too_close:
                    points.append([x, y])
        
        return np.array(points)

    def eval(self):
        return self



def shift_image_batch(image_batch, max_shift=0.4, random_shift=True, enabled=True, shift_x=None, shift_y=None):
    """
    Apply off-center shifts to a batch of images.
    
    Args:
        image_batch: Tensor of shape (B, C, H, W)
        max_shift: Maximum shift as a fraction of image dimensions
        random_shift: If True, use random shifts; if False, use provided shifts
        enabled: Whether shifting is enabled
        shift_x: Specific horizontal shift to apply (if random_shift=False)
        shift_y: Specific vertical shift to apply (if random_shift=False)
    
    Returns:
        Shifted image batch
    """
    if not enabled:
        return image_batch
    
    batch_size, _, height, width = image_batch.shape
    shifted_batch = torch.zeros_like(image_batch)
    
    for i in range(batch_size):
        if random_shift:
            # Random shifts between -max_shift and +max_shift
            shift_x = 2 * max_shift * (np.random.rand() - 0.5)
            shift_y = 2 * max_shift * (np.random.rand() - 0.5)
        
        # Convert shifts to pixels
        pixel_shift_x = int(shift_x * width)
        pixel_shift_y = int(shift_y * height)
        
        # Apply shift with padding
        img = image_batch[i]
        shifted_img = torch.zeros_like(img)
        
        # Source coordinates (where to copy from)
        src_x_start = max(0, -pixel_shift_x)
        src_x_end = min(width, width - pixel_shift_x)
        src_y_start = max(0, -pixel_shift_y)
        src_y_end = min(height, height - pixel_shift_y)
        
        # Target coordinates (where to copy to)
        tgt_x_start = max(0, pixel_shift_x)
        tgt_x_end = min(width, width + pixel_shift_x)
        tgt_y_start = max(0, pixel_shift_y)
        tgt_y_end = min(height, height + pixel_shift_y)
        
        # Copy the valid region
        shifted_img[:, tgt_y_start:tgt_y_end, tgt_x_start:tgt_x_end] = img[:, src_y_start:src_y_end, src_x_start:src_x_end]
        shifted_batch[i] = shifted_img
    
    return shifted_batch


class OffCenterDataset(torch.utils.data.Dataset):
    """Dataset wrapper that applies off-center shifting to images and masks."""
    
    def __init__(self, dataset, enabled=False, max_shift=0.4, random_shift=True):
        self.dataset = dataset
        self.enabled = enabled
        self.max_shift = max_shift
        self.random_shift = random_shift
        # For consistent shifts across evaluation
        if not random_shift:
            self.shifts = []
            for _ in range(len(dataset)):
                # Generate and store fixed shift parameters
                shift_x = 2 * self.max_shift * (np.random.rand() - 0.5)
                shift_y = 2 * self.max_shift * (np.random.rand() - 0.5)
                self.shifts.append((shift_x, shift_y))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, mask, info = self.dataset[idx]
        
        # Apply off-center shift if enabled
        if self.enabled:
            # Determine shift parameters
            if self.random_shift:
                # Generate random shift
                shift_x = 2 * self.max_shift * (np.random.rand() - 0.5)
                shift_y = 2 * self.max_shift * (np.random.rand() - 0.5)
            else:
                # Use consistent pre-generated shift
                shift_x, shift_y = self.shifts[idx]
            
            # Add batch dimension for processing
            image_batch = image.unsqueeze(0)
            mask_batch = mask.unsqueeze(0)
            
            # Apply identical shift to both image and mask
            image_batch = shift_image_batch(
                image_batch, 
                max_shift=self.max_shift,
                random_shift=False,  # Use our own parameters instead
                enabled=True,
                shift_x=shift_x,
                shift_y=shift_y
            )
            
            mask_batch = shift_image_batch(
                mask_batch,
                max_shift=self.max_shift,
                random_shift=False,  # Use our own parameters instead
                enabled=True,
                shift_x=shift_x,
                shift_y=shift_y
            )
            
            # Remove batch dimension
            image = image_batch.squeeze(0)
            mask = mask_batch.squeeze(0)
            
        return image, mask, info


def load_data_with_off_center(config, sam=False, off_center=False, max_shift = 0.2):
    """Load data with optional off-center shifting applied consistently."""
    # First load regular data
    train_loader, val_loader, test_loader = load_data_wrapper(config=config, sam=sam)
    
    if off_center:
        # Wrap datasets with off-center functionality
        train_dataset = OffCenterDataset(
            train_loader.dataset, 
            enabled=True, 
            max_shift=max_shift, 
            random_shift=True  # Random shifts for training
        )
        
        # For evaluation, use consistent shifts to ensure fair comparison
        test_dataset = OffCenterDataset(
            test_loader.dataset, 
            enabled=True, 
            max_shift=max_shift, 
            random_shift=False  # Fixed shifts for evaluation
        )

        val_dataset = OffCenterDataset(
            val_loader.dataset, 
            enabled=True, 
            max_shift=max_shift, 
            random_shift=False 
        )
        
        # Create new dataloaders with wrapped datasets
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=config["train_batch_size"], 
            shuffle=True, 
            num_workers=0
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=config["test_batch_size"], 
            shuffle=False, 
            num_workers=0
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=config["test_batch_size"], 
            shuffle=False, 
            num_workers=0
        )
    
    return train_loader, val_loader, test_loader
