import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, main_folder, transform=None):
        self.main_folder = main_folder
        self.transform = transform
        
        # List subfolders (representing classes)
        self.classes = os.listdir(main_folder)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect all image paths and corresponding labels
        self.image_paths = []
        self.labels = []
        
        for cls in self.classes:
            class_folder = os.path.join(main_folder, cls)
            if os.path.isdir(class_folder):
                for img_name in os.listdir(class_folder):
                    if img_name.endswith(('jpg', 'jpeg', 'png')):  # Filter image files
                        img_path = os.path.join(class_folder, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')  # Open image and convert to RGB
        
        if self.transform:
            image = self.transform(image)
        
        return {'data': image, 'label': label}