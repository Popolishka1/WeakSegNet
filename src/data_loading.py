import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class OxfordPet(Dataset):

    def __init__(self, data_dir, split="trainval", transform=None, target_transform=None, return_id=False):

        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.return_info = return_id  # set it to True if you also want to return the different IDs (CLASS ID, SPECIES ID, BREED ID)

        split_file = os.path.join(data_dir, "annotations", f"{split}.txt")
        print("Loading split from:", split_file)

        with open(split_file, "r") as f:
            lines = f.readlines()

        self.samples = []
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


    def __getitem__(self, idx):
        sample = self.samples[idx]
        name = sample["name"]

        img_path = os.path.join(self.data_dir, "images", f"{name}.jpg")
        mask_path = os.path.join(self.data_dir, "annotations", "trimaps", f"{name}.png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Convert to binary mask (according to the README: 1 = Foreground, 2 = Background, 3 = Not classified)
        mask = np.array(mask)
        mask = np.where(mask == 2, 1, 0).astype(np.uint8)
        mask = Image.fromarray(mask)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask = torch.from_numpy(np.array(mask)).long()

        if self.return_info:
            return image, mask, {
                "name": name,
                "class_id": sample["class_id"],
                "species_id": sample["species_id"],
                "breed_id": sample["breed_id"]
            }
        else:
            return image, mask


def data_transform(image_size = 224):
    # Define some transforms for the input images
    image_transform = transforms.Compose([
    transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR),
    transforms.ToTensor(),
    ]) # TODO: start to think about some data augmentation

    # Define some transforms for the masks
    mask_transform = transforms.Compose([
    transforms.Resize((image_size, image_size), interpolation=Image.NEAREST), # à savoir: use nn interpolation to preserve labels
    transforms.ToTensor(),
    ])

    return image_transform, mask_transform


def data_loading(path, image_size, batch_size_train, batch_size_test):
    print("\n----Loading data")
    image_transform, mask_transform = data_transform(image_size)

    # Load train/val and test datasets
    trainval_dataset = OxfordPet(data_dir=path, split="trainval", transform=image_transform, target_transform=mask_transform)
    test_dataset = OxfordPet(data_dir=path, split="test", transform=image_transform, target_transform=mask_transform)
    
    # Train set & test set loader
    train_loader = DataLoader(trainval_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

    print("\n[Data loaded succesfully]")
    print(f"Number of train+validation instances: {len(trainval_dataset)}")
    print(f"Number of test instances: {len(test_dataset)}")
    return trainval_dataset, test_dataset, train_loader, test_loader