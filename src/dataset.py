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
                 transform=None,
                 target_transform=None,
                 fully_supervised=True,
                 weakly_supervised=False,
                 weakly_supervised_bbox_to_mask_dummy = False
                 ):
        
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.fully_supervised = fully_supervised
        self.weakly_supervised = weakly_supervised
        self.weakly_supervised_bbox_to_mask_dummy = weakly_supervised_bbox_to_mask_dummy

        split_file = os.path.join(data_dir, "annotations", f"{split}.txt")
        print("Loading split from:", split_file)

        with open(split_file, "r") as f:
            lines = f.readlines()

        self.samples = [] # TODO: fix following issue: the train/validation split randomly split the images and the masks but not the samples id
        for line in lines:
            name, class_id, species, breed_id = line.strip().split()
            
            # Check if XML exists
            if weakly_supervised and weakly_supervised_bbox_to_mask_dummy:
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
        processed_mask = np.zeros_like(trimap, dtype=np.uint8)
        processed_mask[(trimap == 1) | (trimap == 3)] = 1  
        return processed_mask
    
    @staticmethod
    def read_xml_file(path):
        root = et.parse(path).getroot()
        bounding_box = []
        for box_attributes in root.findall('object'):
            box = box_attributes.find('bndbox')
            x_min = int(box.find('xmin').text)
            x_max = int(box.find('xmax').text)
            y_min = int(box.find('ymin').text)
            y_max = int(box.find('ymax').text)
            bounding_box.append((x_min, x_max, y_min, y_max))
        return bounding_box 

    def __getitem__(self, idx):
        sample = self.samples[idx]
        name = sample["name"]

        img_path = os.path.join(self.data_dir, "images", f"{name}.jpg")

        image = Image.open(img_path).convert("RGB")
        image_size = image.size

        if self.transform:
            image = self.transform(image)

        info = {
            "name": name,
            "class_id": sample["class_id"],
            "species_id": sample["species_id"],
            "breed_id": sample["breed_id"]
        }
        
        if self.fully_supervised:
            mask_path = os.path.join(self.data_dir, "annotations", "trimaps", f"{name}.png")
            trimap = np.array(Image.open(mask_path))
            mask = Image.fromarray(self.preprocess_mask(trimap=trimap))

            if self.target_transform:
                mask = self.target_transform(mask)
            
            return image, mask, info
        
        if self.weakly_supervised:
            bounding_box_path = os.path.join(self.data_dir, "annotations", "xmls", f"{name}.xml")
            bounding_box = self.read_xml_file(path=bounding_box_path)
            info["bbox"] = bounding_box[0]

            if self.weakly_supervised_bbox_to_mask_dummy:
                dummy_mask = generate_mask(image_size, bounding_box[0])
                if self.target_transform:
                    dummy_mask = self.target_transform(dummy_mask)
                return image, dummy_mask, info
    
            else:
                return image, info


def generate_mask(image_size, bounding_box: tuple):
    """This function takes as an argument an image + the bbox and generates the corresponding weak mask"""
    height, width = image_size
    mask_np = np.zeros((width, height), dtype=np.uint8)
    xmin, xmax, ymin, ymax = map(int, bounding_box)
    mask_np[ymin:ymax, xmin:xmax] = 1
    mask = Image.fromarray(mask_np)
    return mask


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
    # Define some transforms for the input images (fully supervised case)
    image_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
    ]) # TODO: start to think about some data augmentation (might be more relevent for weakseg)

    # Define some transforms for the masks
    mask_transform = transforms.Compose([
    transforms.Lambda(lambda mask: mask_to_tensor(mask)),
    transforms.Resize((image_size, image_size), interpolation=Image.NEAREST), # TODO: ensure discret label values are preserved after resiz
    ])

    return image_transform, mask_transform


def data_loading(path,
                 experiment,
                 image_transform,
                 mask_transform,
                 size,
                 seed: int = 42
                 ):
    print("\n----Loading data")
    fully_supervised, weakly_supervised, weakly_supervised_bbox_to_mask_dummy = experiment
    batch_size_train, batch_size_val, batch_size_val, val_split = size

    # Load train+val dataset
    full_trainval_dataset = OxfordPet(data_dir=path,
                                      split="trainval",
                                      transform=image_transform,
                                      target_transform=mask_transform,
                                      fully_supervised=fully_supervised,
                                      weakly_supervised=weakly_supervised,
                                      weakly_supervised_bbox_to_mask_dummy=weakly_supervised_bbox_to_mask_dummy
                                      )
    
    total_size = len(full_trainval_dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_trainval_dataset, [train_size, val_size], generator=generator)

    # Load test dataset
    test_dataset = OxfordPet(data_dir=path,
                             split="test",
                             transform=image_transform,
                             target_transform=mask_transform,
                             fully_supervised=True,
                             weakly_supervised=False,
                             weakly_supervised_bbox_to_mask_dummy=False
                             )
    
    # Train set, validation set & test set loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size_val, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_val, shuffle=False)

    print("\n[Data loaded succesfully]")
    print(f"Total train+val set: {total_size} -> Train: {train_size} | Val: {val_size}")
    print(f"Test set: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader