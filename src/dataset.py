import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import xml.etree.ElementTree as et
from torch.utils.data import Dataset, DataLoader, random_split, Subset

class OxfordPet(Dataset):
    def __init__(self,
                 data_dir,
                 split,
                 image_transform=None,
                 mask_transform=None
                 ):
        
        self.data_dir = data_dir
        self.split = split
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        split_file = os.path.join(data_dir, "annotations", f"{split}.txt")
        print("Loading split from:", split_file)

        with open(split_file, "r") as f:
            lines = f.readlines()

        self.samples = []
        for line in lines:
            name, class_id, species, breed_id = line.strip().split()

            # Be careful, class_id and species-id are 1-indexed in the dataset
            # We convert them to 0-indexed for PyTorch
            # breed_id won't be used for classification, but we keep it for consistency
            self.samples.append({
                "name": name,
                "class_id": int(class_id) - 1,
                "species_id": int(species) - 1,
                "breed_id": int(breed_id) - 1
            })


    def __len__(self):
        return len(self.samples)

    @staticmethod
    def preprocess_mask(trimap):
        """Converts the trimap to a binary mask (1 for pet pixels, 0 for background)."""
        processed_mask = np.zeros_like(trimap, dtype=np.uint8)
        processed_mask[(trimap == 1) | (trimap == 3)] = 1 # TODO (all): refine this to process the frontier pixels (3)
        return processed_mask
    
    @staticmethod
    def read_xml_file(path):
        """Reads the XML file and returns the first bounding box as (xmin, xmax, ymin, ymax)."""
        if not os.path.exists(path):
            return None
        root = et.parse(path).getroot()
        bounding_box = []
        for box_attributes in root.findall('object'):
            box = box_attributes.find('bndbox')
            x_min = int(box.find('xmin').text)
            x_max = int(box.find('xmax').text)
            y_min = int(box.find('ymin').text)
            y_max = int(box.find('ymax').text)
            bounding_box.append((x_min, x_max, y_min, y_max))
        return bounding_box[0] if bounding_box else None

    def __getitem__(self, idx):
        sample = self.samples[idx]
        name = sample["name"]

        # Load and transform input image
        image_path = os.path.join(self.data_dir, "images", f"{name}.jpg")
        image = Image.open(image_path).convert("RGB")

        if self.image_transform:
            image = self.image_transform(image)
        
        # Load, preprocess and transform ground truth mask
        trimap_path = os.path.join(self.data_dir, "annotations", "trimaps", f"{name}.png")
        trimap = np.array(Image.open(trimap_path))
        gt_mask = Image.fromarray(self.preprocess_mask(trimap=trimap))
        if self.mask_transform:
            gt_mask = self.mask_transform(gt_mask)

        # Build info dictionary with IDs and bounding box (if applicable)
        info = {
            "name": name,
            "class_id": sample["class_id"],
            "species_id": sample["species_id"],
            "breed_id": sample["breed_id"],
            "bbox": [0, 0, 0, 0] # default placeholder
        }

        xml_path = os.path.join(self.data_dir, "annotations", "xmls", f"{name}.xml")
        bbox = self.read_xml_file(path=xml_path)
        if bbox is not None:
            info["bbox"] = bbox

        return image, gt_mask, info


class PseudoMaskDataset(Dataset):
    """Dataset wrapper to inject pseudo masks into training data."""
    def __init__(self, original_dataset, pseudo_masks):
        self.original_dataset = original_dataset
        self.pseudo_masks = pseudo_masks

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # Get original data (image, gt_mask, info)
        image, _, info = self.original_dataset[idx]
        # Replace ground truth mask with pseudo mask
        pseudo_mask = self.pseudo_masks[idx]
        return image, pseudo_mask, info


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
        transforms.Lambda(lambda mask: mask_to_tensor(mask=mask))
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
            transforms.Lambda(lambda mask: mask_to_tensor(mask=mask))
        ]

    image_transform = transforms.Compose(image_transforms)
    mask_transform = transforms.Compose(mask_transforms)

    return image_transform, mask_transform


def inverse_normalize(tensor):
    """
    Reverse normalization for visualization.
    Be careful, if you change normalization values in data_transform(), change them here too.
    """
    inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
    return inv_normalize(tensor)


def data_loading(path: str,
                 data_split_size: tuple,
                 image_size: int = 256,
                 seed: int = 42
                 ):
    """Load Oxford-IIIT Pet Dataset and split into train, val and test sets. Return corresponding loaders."""
    print("\n----Loading data...")
    batch_size_train, batch_size_val, batch_size_test, val_split = data_split_size

    # Train data gets augmentations
    train_image_transform, train_mask_transform = data_transform(image_size=image_size, train=True)
    
    # Test and validation data get basic transforms (no augmentations)
    test_image_transform, test_mask_transform = data_transform(image_size=image_size, train=False)

    # Load train dataset
    full_train_dataset = OxfordPet(
        data_dir=path,
        split="trainval",
        image_transform=train_image_transform, # Augmentations ON for train data
        mask_transform=train_mask_transform
    )

    # Load val dataset (SAME SPLIT that train dataset but different transforms)
    full_val_dataset = OxfordPet(
        data_dir=path,
        split="trainval", 
        image_transform=test_image_transform, # Augmentations OFF for val data
        mask_transform=test_mask_transform
    )
    
    # Load test dataset
    test_dataset = OxfordPet(
        data_dir=path,
        split="test", 
        image_transform=test_image_transform, # Augmentations OFF for test data
        mask_transform=test_mask_transform
    )

    # Split indices (same for both train and val datasets)
    train_size = int((1 - val_split) * len(full_train_dataset))
    train_indices, val_indices = random_split(range(len(full_train_dataset)),
                                              [train_size, len(full_train_dataset)-train_size],
                                              generator=torch.Generator().manual_seed(seed)
                                              )
    
    # Create subsets with proper transforms
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)
    
    # Create train, val & test loaders
    # TODO: num_workers > 0 if enough CPU cores (beug with Windows)
    train_loader = DataLoader(train_dataset, batch_size_train, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size_val, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size_test, shuffle=False, num_workers=0)

    print("[Data loaded succesfully]")
    print(f"\nTraining set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    return train_loader, val_loader, test_loader


def load_data_wrapper(config):
    """"Wrapper function to load data from the config file"""
    BASE_DIR = os.getcwd()
    FILE_PATH = os.path.join(BASE_DIR, config["data_folder"])
    
    # Data split and batch sizes from the config
    batch_size_train = config["train_batch_size"]
    batch_size_val = config["val_batch_size"]
    batch_size_test = config["test_batch_size"]
    val_split = config["val_split"]
    data_split_size = (batch_size_train, batch_size_val, batch_size_test, val_split)
    
    # Image size for resize  
    image_size = config["image_size"]
    
    return data_loading(path=FILE_PATH, data_split_size=data_split_size, image_size=image_size)
