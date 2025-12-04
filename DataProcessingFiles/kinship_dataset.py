import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class KinshipPairDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, is_training=True):
        self.pairs_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.is_training = is_training

        if transform:
            self.transform = transform
        elif is_training:
            # Augmentation for training - increased strength
            self.transform = transforms.Compose([
                transforms.Resize((176, 176)),
                transforms.RandomCrop((160, 160)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomRotation(degrees=15),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            # No augmentation for eval
            self.transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __getitem__(self, idx):
        row = self.pairs_df.iloc[idx]

        # Ensure all paths are relative, and joined properly
        img1_path = os.path.normpath(os.path.join(self.root_dir, row['img1'].strip().lstrip("/")))
        img2_path = os.path.normpath(os.path.join(self.root_dir, row['img2'].strip().lstrip("/")))


        try:
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")
        except Exception as e:
            print(f"ERROR: Failed to loag image pair {img1_path}, {img2_path}")
            print(f"Reason: {e}")
            return self.__getitem__((idx+1)%len(self.pairs_df))

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        label = torch.tensor(row['label'], dtype=torch.float32)
        return img1, img2, label

    def __len__(self):
        return len(self.pairs_df)