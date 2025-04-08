import os
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms
import xml.etree.ElementTree as et
from torch.utils.data import Dataset, DataLoader, random_split, Subset


def data_transform(image_size: int = 256, train: bool = True):
    """
    Data transformations/augmentations for images and masks (**synchronized** augmentations when training).
    The syncronization of the augmentations are done in the Dataset class (OxfordPet),
    with torch.set_rng_state() and torch.get_rng_state().
    """
    # TODO (Paul): consider separating transformations/augmentations for each task (segmentation/classification/CAM)

    # Function to convert PIL mask to tensor. Don't use ToTensor() directly on PIL. See the documentation of ToTensor()
    def mask_to_tensor(mask):
        mask_np = np.array(mask, dtype=np.uint8)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()
        return mask_tensor
    
    # Base image transforms (always applied)
    image_transforms = [
        transforms.Resize(size=(image_size, image_size)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    # Base mask transforms (always applied)
    mask_transforms = [
        transforms.Resize(size=(image_size, image_size), interpolation=Image.NEAREST),
        transforms.Lambda(lambda mask: mask_to_tensor(mask))
    ]

    # Training-specific augmentations
    if train: # TODO (all): find other augmentations
        # For images
        image_transforms = [
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Resize(size=(image_size, image_size)),
            # transforms.RandomHorizontalFlip(p=0.5), # TODO: find a way to synchronize this with the mask
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ] 
        # TODO (Paul): I have a clue finally: use this pytorch package from torchvision.transforms import v2
        # https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_getting_started.html 
        
        # For masks - same flip (syncronized with image) but no color jitter
        mask_transforms = [
            transforms.Resize(size=(image_size, image_size), interpolation=Image.NEAREST),
            # transforms.RandomHorizontalFlip(p=0.5), # TODO: find a way to synchronize this with the image
            transforms.Lambda(lambda mask: mask_to_tensor(mask))
        ]

    image_transform = transforms.Compose(image_transforms)
    mask_transform = transforms.Compose(mask_transforms)

    return image_transform, mask_transform