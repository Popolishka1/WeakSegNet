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
            
            outputs.append(torch.from_numpy(masks[0]).float().to(self.device))
        return torch.stack(outputs, dim=0)

    def eval(self):
        return self
