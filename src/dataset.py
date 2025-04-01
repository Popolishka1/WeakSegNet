import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import xml.etree.ElementTree as et
from torch.utils.data import Dataset, DataLoader, random_split


class OxfordPet(Dataset):
    def __init__(self,
                 data_dir,
                 split,
                 image_transform=None,
                 mask_transform=None,
                 load_bbox = False
                 ):
        
        self.data_dir = data_dir
        self.split = split
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.load_bbox = load_bbox

        split_file = os.path.join(data_dir, "annotations", f"{split}.txt")
        print("Loading split from:", split_file)

        with open(split_file, "r") as f:
            lines = f.readlines()

        self.samples = []
        for line in lines:
            name, class_id, species, breed_id = line.strip().split()
            
            # Check if XML exists
            if load_bbox:
                xml_path = os.path.join(data_dir, "annotations", "xmls", f"{name}.xml")
                if not os.path.exists(xml_path):
                    continue # if no bbox: skip sample
            
            self.samples.append({
                "name": name,
                "class_id": int(class_id),
                "species_id": int(species),
                "breed_id": int(breed_id)
            })


    def __len__(self):
        return len(self.samples)

    @staticmethod
    def preprocess_mask(trimap):
        """Converts the trimap to a binary mask (1 for pet pixels, 0 for background)."""
        processed_mask = np.zeros_like(trimap, dtype=np.uint8)
        processed_mask[(trimap == 1) | (trimap == 3)] = 1 # TODO: refine this to process the frontier pixels
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
        gt_mask = Image.fromarray(self.preprocess_mask(trimap))
        if self.mask_transform:
            gt_mask = self.mask_transform(gt_mask)

        # Build info dictionary with IDs and bounding box (if applicable)
        info = {
            "name": name,
            "class_id": sample["class_id"],
            "species_id": sample["species_id"],
            "breed_id": sample["breed_id"]
        }

        if self.load_bbox:
            xml_path = os.path.join(self.data_dir, "annotations", "xmls", f"{name}.xml")
            info["bbox"] = self.read_xml_file(xml_path)

        return image, gt_mask, info


def mask_to_tensor(mask):
    mask_np = np.array(mask, dtype=np.uint8)
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()
    return mask_tensor


def data_transform(image_size: int = 224):
    # Define some transforms for the input images
    image_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
    ]) # TODO: start to think about some data augmentation

    # Define some transforms for the ground truth masks
    mask_transform = transforms.Compose([
    transforms.Lambda(lambda mask: mask_to_tensor(mask)),
    transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
    ])
    return image_transform, mask_transform


def data_loading(path,
                 data_split_size,
                 image_transform,
                 mask_transform,
                 load_bbox: bool = False,
                 seed: int = 42
                 ):
    print("\n----Loading data")
    batch_size_train, batch_size_val, batch_size_val, val_split = data_split_size

    # Load train & val dataset
    full_trainval_dataset = OxfordPet(data_dir=path,
                                      split="trainval",
                                      image_transform=image_transform,
                                      mask_transform=mask_transform,
                                      load_bbox=load_bbox
                                      )
    
    total_size = len(full_trainval_dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_trainval_dataset, [train_size, val_size], generator=generator)

    # Load test dataset
    test_dataset = OxfordPet(data_dir=path,
                             split="test",
                             image_transform=image_transform,
                             mask_transform=mask_transform,
                             load_bbox=load_bbox
                             )
    
    # Train, val & test loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size_val, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_val, shuffle=False)

    print("\n[Data loaded succesfully]")
    print(f"Total train+val set: {total_size} -> Train: {train_size} | Val: {val_size}")
    print(f"Test set: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader