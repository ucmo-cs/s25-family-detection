import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class KinshipPairDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.pairs_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        row = self.pairs_df.iloc[idx]

        # Ensure all paths are relative, and joined properly
        img1_path = os.path.normpath(os.path.join(self.root_dir, row['img1'].strip().lstrip("/")))
        img2_path = os.path.normpath(os.path.join(self.root_dir, row['img2'].strip().lstrip("/")))

        print("\n[DEBUG]")
        print("img1 raw:", repr(row['img1']))
        print("joined img1 path:", repr(img1_path))
        print("exists?", os.path.exists(img1_path))

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