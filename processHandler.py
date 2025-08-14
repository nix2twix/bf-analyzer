# -*- coding: cp1251 -*-
# === GENERAL VARIABLES ===
CHECKPOINTPATH = "1k2k-unet-model.pth"
PATTERN = r'\.(\d+)_(\d+)\.png$'

# === LIBRARIES GENERAL ===
import cv2  
import torch

import streamlit as st
import numpy as np

from stqdm import stqdm
from skimage import measure
from skimage.measure import label, regionprops
from skimage import measure, morphology
from PIL import Image
from torch.utils.data import DataLoader
from io import BytesIO

# === PROJECT SCRIPTS ===
from processingFunctions import makePatches, TestDataset
from processingFunctions import (
    buildModel, 
    loadCheckpoint, 
    loadCellposeModel,
    cropLineBelow)


# === SECONDARY FUNCTIONS ===
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def imgToBytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def bytesToImg(img_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(img_bytes))

# === PROCESSING BLOCK ===
@st.cache_data(show_spinner=False, ttl=6000, max_entries=10)
def segmentationImage(
    uploaded_file, 
    INFLINEPX, 
    width, 
    height, 
    imgName,
    threshold=0.5
):
    uploaded_file.seek(0)
    img_bytes = uploaded_file.read()
    origImg = bytesToImg(img_bytes)

    croppedImg = cropLineBelow(origImg, INFLINEPX)
    croppedW, croppedH = croppedImg.size

    if height > 512 and width > 512:
        imgPatches, patchesInfo = makePatches(
            croppedImg, patch_size=(512, 512), stride=(256, 256)
        )
    else:
        imgPatches = [croppedImg]
        patchesInfo = [(0, 0, 0)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] DEVICE IS {device}")

    test_dataset = TestDataset(imgPatches, patchesInfo)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = buildModel().to(device)
    loadCheckpoint(model, CHECKPOINTPATH)
    model.eval()

    classNames = ["bg", "biofilm", "intermediate", "single"]
    modelClassNames = [name for name in classNames if name != "bg"]

    probs = {name: np.zeros((croppedH, croppedW), dtype=np.float32) for name in modelClassNames}
    count = np.zeros((croppedH, croppedW), dtype=np.float32)

    with torch.no_grad():
        for images, patch_info in stqdm(test_loader):
            images = images.to(device)
            outputs = torch.softmax(model(images), dim=1).cpu()

            for idx in range(images.size(0)):
                x = patch_info[0].item()
                y = patch_info[1].item()

                for classIdx, name in enumerate(modelClassNames, start=1): 
                    probs[name][y:y+512, x:x+512] += outputs[idx, classIdx].numpy()
                count[y:y+512, x:x+512] += 1

    count += 1e-9
    for name in probs:
        probs[name] /= count

    # ==== Выбор предсказанного класса ====
    stackedProbs = np.stack([probs[name] for name in modelClassNames], axis=0) 
    predMask = np.argmax(stackedProbs, axis=0) + 1  # т.к. "bg" — 0
    predMask[np.max(stackedProbs, axis=0) <= threshold] = 0

    # ==== Бинарные маски ====
    predictedLabels = {}
    for i, name in enumerate(modelClassNames, start=1):
        binary_mask = (predMask == i).astype(np.uint8)
        # Расширяем до исходного размера
        mask_full = np.zeros((height, width), dtype=np.uint8)
        mask_full[:croppedH, :croppedW] = binary_mask
        predictedLabels[name] = mask_full

    print(f"[INFO] PROCESSING {imgName} DONE!")
    return predictedLabels

@st.cache_data(show_spinner=False, ttl=6000, max_entries=10)
def getPredictedObjects(predictedLabels):
    predictedObjects = {}
    for name, binary_mask in predictedLabels.items():
        mask_uint8 = binary_mask.astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(mask_uint8, connectivity=8)
        predictedObjects[name] = labels.astype(np.int32)

        print(f"[INFO] Class '{name}': {num_labels - 1} objects found")

    return predictedObjects

@st.cache_data(show_spinner=False, ttl=6000, max_entries=10)
def drawPicture(uploaded_file: bytes, objects, classColors, isShowIntermediate = False):
    uploaded_file.seek(0)
    origImgBytes = uploaded_file.read()
    origImg = bytesToImg(origImgBytes).convert("RGBA")

    overlay = np.array(origImg.copy())
    overlay[:, :, 3] = 0

    for className, mask in objects.items():
        if className not in classColors:
            continue
        if className == "intermediate" and not isShowIntermediate:
            color = classColors["biofilm"]  
        else:
            color = classColors[className]

        overlay[mask > 0] = color

    return Image.alpha_composite(origImg, Image.fromarray(overlay, mode="RGBA"))



@st.cache_data(show_spinner=False, ttl=6000, max_entries = 10)  
def calculateStatistics(objectsInfo, scale=0.05):
    stats = {}
    scale_factor = scale ** 2 

    for className, objList in objectsInfo.items():
        obj_count = len(objList)
        total_area_px = sum(obj["area"] for obj in objList)
        total_area_mkm = total_area_px * scale_factor

        stats[className] = {
            "count": obj_count,
            "total_area_px": total_area_px,
            "total_area_mkm": total_area_mkm
        }

    return stats

@st.cache_data(show_spinner=False, ttl=6000, max_entries = 10)    
def prepareObjectInfo(predictedLabels):
    objectsInfo = {}
    areaStats = {}

    for className, binaryMask in predictedLabels.items():
        mask_uint8 = binaryMask.astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(mask_uint8, connectivity=8)

        class_objects = []
        areas = []

        for obj_id in range(1, num_labels):  # 0 — фон
            obj_mask = (labels == obj_id)
            area = int(np.sum(obj_mask))
            areas.append(area)

            obj_info = {
                "id": obj_id,
                "area": area
            }

            if className == "single":
                props = regionprops(obj_mask.astype(np.uint8))
                if props:
                    obj_info["eccentricity"] = float(props[0].eccentricity)
                else:
                    obj_info["eccentricity"] = None

            class_objects.append(obj_info)

        objectsInfo[className] = class_objects

        areaStats[className] = {
            "min_area": int(np.min(areas)) if areas else 0,
            "max_area": int(np.max(areas)) if areas else 0
        }

    return objectsInfo, areaStats

@st.cache_data(show_spinner=False, ttl=6000, max_entries = 10)  
def filtrationObjects(objectsInfo, predictedObjects, params, showIntermedAsBf=True):
    filteredObjects = {}

    for className, objList in objectsInfo.items():
        obj_mask = predictedObjects[className]
        filtered_mask = np.zeros_like(obj_mask, dtype=np.int32)

        for obj in objList:
            area_ok = True
            ecc_ok = True

            if className == "single":
                area_ok = params["minSingleArea"] <= obj["area"] <= params["maxSingleArea"]
                ecc_ok = (obj.get("eccentricity") is None or obj["eccentricity"] >= params["minSingleEcc"])
            elif (className == "biofilm" or className == "intermediate") and showIntermedAsBf == True:
                area_ok = params["minBfArea"] <= obj["area"] <= params["maxBfArea"]
            elif className == "intermediate" and showIntermedAsBf == False:
                area_ok = params["minIntermediateArea"] <= obj["area"] <= params["maxIntermediateArea"]
                
            if area_ok and ecc_ok:
                filtered_mask[obj_mask == obj["id"]] = obj["id"]

        filteredObjects[className] = filtered_mask

    return filteredObjects

 

if __name__ == "__main__":
    
    with open(r"C:\Users\Victory\YandexDisk\PROJECTS\bf-analyzer\examples\18-BSE-1k-T1.086.128_640.png", "rb") as fh:
        uploaded_file = Image.open(fh)
        cellposeParams = [0.4, 0.0]
        labels = segmentationImage(uploaded_file,
                                 "name",
                                 cellposeParams)
        result = drawPicture(uploaded_file, labels)
        result.show()
        params = {
                "minSingleArea": 0,
                "maxSingleArea": 100,
                "minSingleEcc": 0.5,
                "minBfArea": 0,
                "maxBfArea": 100
        }

        filterLabels = filtrationObjects(uploaded_file,
                                       labels,
                                       params)
        
        result = drawPicture(uploaded_file, filterLabels)
        result.show()
    
