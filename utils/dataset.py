# === LIBRARIES GENERAL ===
import cv2

import numpy as np
import albumentations as A

from PIL import Image
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

class TestDataset(Dataset):
    def __init__(self, pil_images, imagesInfo):
        self.imgPatches = pil_images
        self.patchesInfo = imagesInfo
        self.transform = self._get_transforms()
    
    def _get_transforms(self):
        aug_list = []
        aug_list.extend([
            A.Normalize(mean=0, std=1),
            ToTensorV2()  # (C, H, W)
        ])
        
        return A.Compose(aug_list)
    
    def __len__(self):
        return len(self.imgPatches)
    
    def __getitem__(self, idx):
        image = np.array(self.imgPatches[idx])
        transformed = self.transform(image)
        return transformed["image"]  #(C, H, W)

    def make_clahe(self, image):
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image.copy()
        
        if img_np.dtype != np.uint8 and img_np.dtype != np.uint16:
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

        if len(img_np.shape) == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        elif len(img_np.shape) > 3:
            img_np = img_np.squeeze()
    
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_clahe = clahe.apply(img_np)
        return img_clahe

    def __len__(self):
        return len(self.imgPatches)

    def __getitem__(self, idx):
        image = self.imgPatches[idx]
        image = self.make_clahe(image)
        augmented = self.transform(image=np.array(image))
        image = augmented['image']
        coords = self.patchesInfo[idx]
        return image, coords