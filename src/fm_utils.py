import numpy as np
import torch

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