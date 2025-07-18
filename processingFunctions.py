# === LIBRARIES GENERAL ===
import cv2
import torch

import numpy as np
import torch.nn as nn
import streamlit as st
import albumentations as A
import segmentation_models_pytorch as smp

from cellpose import models
from PIL import Image
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

# === CLASSES ===
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class TestDataset(Dataset):
    def __init__(self, pil_images, imagesInfo, augmentation=False):
        self.imgPatches = pil_images
        self.patchesInfo = imagesInfo
        self.augmentation = augmentation
        self.transform = self._get_transforms()
    
    def _get_transforms(self):
        aug_list = []
        
        if self.augmentation:
            aug_list.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5)
            ])
        
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

    def make_clahe(self, img):
        img_np = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_np)
        return Image.fromarray(img_clahe)

    def __len__(self):
        return len(self.imgPatches)

    def __getitem__(self, idx):
        image = self.imgPatches[idx]
        image = self.make_clahe(image)
        augmented = self.transform(image=np.array(image))
        image = augmented['image']
        coords = self.patchesInfo[idx]
        return image, coords

# === FUNCTIONS ===
def loadCheckpoint(model, checkpoint_path):
    model = nn.DataParallel(model) 
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.module    
    print(f"Model loaded from: {checkpoint_path}")
    
@st.cache_resource
def loadCellposeModel():
    return models.CellposeModel(
            gpu=True
        )

def buildModel(encoderName = 'resnet34', encoderWeights = 'imagenet', activation = None):
    model = smp.UnetPlusPlus(
        encoder_name=encoderName,
        encoder_weights=encoderWeights,
        in_channels=1,  # grayscale
        classes=2,
        activation=activation
    )
    return model

def cropLineBelow(imgPIL, countPx=120):
    width, height = imgPIL.size
    cropped_img = imgPIL.crop((0, 0, width, height - countPx))
    return cropped_img

def makePatches(imgPIL, img_name, patch_size=(512, 512), stride=(128, 128)):   
    imgPIL = cropLineBelow(imgPIL, countPx=128)
    img_np = np.array(imgPIL)
    img_height, img_width = img_np.shape[:2]

    patch_h, patch_w = patch_size
    stride_y, stride_x = stride

    patch_id = 0
    patch_list = []
    coords = []

    x_coords = list(range(0, img_width - patch_w + 1, stride_x))
    y_coords = list(range(0, img_height - patch_h + 1, stride_y))

    for y in y_coords:
        for x in x_coords:
            patch = imgPIL.crop((x, y, x + patch_w, y + patch_h))
            patch_list.append(patch)
            coords.append((x, y, patch_id))
            patch_id += 1
            
    return patch_list, coords

