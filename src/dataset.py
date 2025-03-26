import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

class OxfordPet(Dataset):

    def __init__(self, data_dir, split="trainval", transform=None, target_transform=None, return_id=True):

        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.return_info = return_id  # set it to True if you also want to return the different IDs (CLASS ID, SPECIES ID, BREED ID)

        split_file = os.path.join(data_dir, "annotations", f"{split}.txt")
        print("Loading split from:", split_file)

        with open(split_file, "r") as f:
            lines = f.readlines()

        self.samples = [] # TODO: fix following issue: the train/validation split randomly split the images and the masks but not the samples id
        for line in lines:
            name, class_id, species, breed_id = line.strip().split()
            self.samples.append({
                "name": name,
                "class_id": int(class_id),
                "species_id": int(species),
                "breed_id": int(breed_id)
            })


    def __len__(self):
        return len(self.samples)

    @staticmethod
    def preprocess_mask(mask):
        processed = np.zeros_like(mask, dtype=np.uint8)
        processed[(mask == 1) | (mask == 3)] = 1  
        return processed
    
    # TODO: think on how to include the xlms frames to do weak supervision

    def __getitem__(self, idx):
        sample = self.samples[idx]
        name = sample["name"]

        img_path = os.path.join(self.data_dir, "images", f"{name}.jpg")
        mask_path = os.path.join(self.data_dir, "annotations", "trimaps", f"{name}.png")

        image = Image.open(img_path).convert("RGB")

        trimap = np.array(Image.open(mask_path))
        mask = Image.fromarray(self.preprocess_mask(trimap))

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        if self.return_info:
            return image, mask, {
                "name": name,
                "class_id": sample["class_id"],
                "species_id": sample["species_id"],
                "breed_id": sample["breed_id"]
            }
        else:
            return image, mask


def mask_to_tensor(mask):
    # I found out that my code wasn't working because
    # in the mask_transform definded below in the data_transform function, the transforms.ToTensor()
    # transormation devides the 1 of the binary mask by 255
    # Reason : ToTensor() converts images to tensor by / every pixel value by 255
    # Solution : convert the PIL image to array  uint8 
    # Resulst: we get the correct target :)
    # TODO: maybe find a cleaner way to do it
    mask_np = np.array(mask, dtype=np.uint8)
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()
    return mask_tensor

def data_transform(image_size: int = 224):
    # Define some transforms for the input images
    image_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
    ]) # TODO: start to think about some data augmentation

    # Define some transforms for the masks
    mask_transform = transforms.Compose([
    transforms.Resize((image_size, image_size), interpolation=Image.NEAREST ),# TODO: ensure discret label values are preserved after resiz
    transforms.Lambda(lambda mask: mask_to_tensor(mask))
    ])

    return image_transform, mask_transform


def data_loading(path, image_size, batch_size_train, batch_size_val, batch_size_test, val_split: float = 0.2, seed: int = 42):
    print("\n----Loading data")
    image_transform, mask_transform = data_transform(image_size=image_size)

    # Load train+val dataset
    full_trainval_dataset = OxfordPet(data_dir=path, split="trainval", transform=image_transform, target_transform=mask_transform)
    total_size = len(full_trainval_dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_trainval_dataset, [train_size, val_size], generator=generator)

    # Load test dataset
    test_dataset = OxfordPet(data_dir=path, split="test", transform=image_transform, target_transform=mask_transform)
    
    # Train set, validation set & test set loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size_val, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=False)

    print("\n[Data loaded succesfully]")
    print(f"Total train+val set: {total_size} -> Train: {train_size} | Val: {val_size}")
    print(f"Test set: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader