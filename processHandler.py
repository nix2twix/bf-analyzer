# -*- coding: cp1251 -*-
# === GENERAL VARIABLES ===
CHECKPOINTPATH = "allSEMchkpt350.pth"
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
    cellposeParams = [0.4, 0.0]):
    
    uploaded_file.seek(0)
    img_bytes = uploaded_file.read()
    
    origImg = bytesToImg(img_bytes)
    
    cropedImg = cropLineBelow(origImg, INFLINEPX)
    cropedImgWidth, cropedImgHeight = cropedImg.size
    
    imgPatches = []
    patchesInfo = []
    print(f"[INFO] START PROCESSING {imgName}...")

    if height > 512 and width > 512:
        imgPatches, patchesInfo = makePatches(
            cropedImg, patch_size=(512, 512), stride=(256, 256)
        )
    else:
        imgPatches.append(cropedImg)
        patchesInfo.append((0, 0, 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] DEVICE IS {device}!")

    test_dataset = TestDataset(imgPatches, patchesInfo)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = buildModel().to(device)
    loadCheckpoint(model, CHECKPOINTPATH)
    model.eval()

    probsCount = np.zeros((cropedImgHeight, cropedImgWidth), dtype=float)
    biofilmProbs = np.zeros((cropedImgHeight, cropedImgWidth), dtype=float)

    with torch.no_grad():
        for (images, patchesInfo) in stqdm(test_loader):
            images = images.to(device)
            outputs = model(images).cpu()
            for idx in range(images.size(0)):
                x = patchesInfo[0].item()
                y = patchesInfo[1].item()
                output_np = outputs[idx].numpy()[1]
                biofilmProbs[y:y+512, x:x+512] += output_np
                probsCount[y:y+512, x:x+512] += 1
                print(f'---> {patchesInfo[2].item()} <---')

    threshold = 0.5
    biofilmProbs = biofilmProbs / probsCount
    biofilmPredictions = (biofilmProbs > threshold).astype(np.uint8)

    origImgNP = np.array(cropedImg)
    cleaned_image = origImgNP.copy()
    cleaned_image[biofilmPredictions == 1] = 0

    singlePredictions = np.zeros_like(biofilmPredictions)
       
    print(f"[INFO] START CELLPOSE-SAM PROCESSING...")

    model_cp = loadCellposeModel() 
    singlePredictions, flows, styles = model_cp.eval(cleaned_image, 
                                                     channels=[0, 0], 
                                                     flow_threshold=cellposeParams[0], 
                                                     cellprob_threshold=cellposeParams[1])
    

    predictedLabels = {
    "single": singlePredictions,
    "bf": biofilmPredictions
    }
    
    for key, mask in predictedLabels.items():
        fullMask = np.zeros((height, width), dtype=mask.dtype)
        fullMask[:cropedImgHeight, :cropedImgWidth] = mask
        predictedLabels[key] = fullMask

    print(f"PROCESSED SUCCESSFULLY!")
    return predictedLabels

@st.cache_data(show_spinner=False, ttl=6000, max_entries=10)
def drawPicture(uploaded_file: bytes, 
                predictedLabels: dict) -> Image.Image:
    
    #st.image(predictedLabels["single"] * 255, caption="Single mask", clamp=True)
    #st.image(predictedLabels["bf"] * 255, caption="Biofilm mask", clamp=True)

    uploaded_file.seek(0)
    origImgBytes = uploaded_file.read()
    origImg = bytesToImg(origImgBytes).convert("RGBA")

    biofilmMask = (predictedLabels["bf"] == 1)
    bacteriaMask = (predictedLabels["single"] != 0)

    biofilmColor = np.array([36, 179, 83, 178], dtype=np.uint8)
    bacteriaColor = np.array([184, 61, 245, 178], dtype=np.uint8)

    overlay = np.array(origImg.copy())
    overlay[:, :, 3] = 0

    overlay[biofilmMask] = biofilmColor
    overlay[bacteriaMask] = bacteriaColor

    overlayRGBA = Image.fromarray(overlay, mode="RGBA")
    return Image.alpha_composite(origImg, overlayRGBA)


@st.cache_data(show_spinner=False, ttl=6000, max_entries = 10)  
def calculateStatistics(predictedLabels, scale=0.05):
    biofilmMask = (predictedLabels["bf"] == 1)
    bacteriaMask = (predictedLabels["single"] != 0)

    labeledBacteria = label(predictedLabels["single"])
    bacteriaCount = labeledBacteria.max()

    return {
        "biofilm_area": int(np.sum(biofilmMask)),
        "biofilm_mkm_area": int(np.sum(biofilmMask)) * (scale ** 2),
        "bacteria_count": int(bacteriaCount),
        "bacteries_mkm_area": int(np.sum(bacteriaMask)) * (scale ** 2)
    }

@st.cache_data(show_spinner=False, ttl=6000, max_entries = 10)    
def makeBacteriaInfo(predictedLabels):
    singlePredictions = predictedLabels["single"] 
    singleBacteriesInfo = []
    
    for i in range(singlePredictions.min() + 1,  singlePredictions.max() + 1):
        maskArray = np.zeros_like(singlePredictions)
        maskArray[singlePredictions == i] = 1
        maskLabel = label(maskArray)
        if (maskLabel.any()):
            properties = regionprops(maskLabel)[0]
        
            bacteriaInfo = {
                            "maskNum": i,
                            "maskArea": np.sum(maskArray),
                            "maskEcc": properties.eccentricity,
                            "centroidCoords": properties.centroid
            }
            singleBacteriesInfo.append(bacteriaInfo)
            
    if singleBacteriesInfo:
        areas = [bacteria["maskArea"] for bacteria in singleBacteriesInfo]
        min_area, max_area = int(min(areas)), int(max(areas))
        if min_area == max_area:
            min_area = 0
    else:
        min_area, max_area = 0, 10000
        
    return singleBacteriesInfo, min_area, max_area
 
def prepareObjectInfo(predictedLabels):
    bacteriaInfo, minSingleArea, maxSingleArea = makeBacteriaInfo(predictedLabels)

    bfPredictions = predictedLabels["bf"]
    maskUint8 = (bfPredictions * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(maskUint8, cv2.MORPH_CLOSE, kernel)
    labeledMask = label(closed > 0, connectivity=2)
    biofilmRegions = list(regionprops(labeledMask))

    if biofilmRegions:
        areas = [r.area for r in biofilmRegions]
        minBfArea, maxBfArea = int(min(areas)), int(max(areas))
        if minBfArea == maxBfArea:
            minBfArea = 0
    else:
        minBfArea, maxBfArea = 0, 10000

    return [bacteriaInfo, minSingleArea, maxSingleArea,
            biofilmRegions, minBfArea, maxBfArea]

@st.cache_data(show_spinner=False, ttl=6000, max_entries = 10)  
def filtrationObjects(bacteriaInfo, biofilmMask, params):
    # -------- Фильтрация single --------
    filteredIds = [
        b["maskNum"] for b in bacteriaInfo
        if params["minSingleArea"] < b["maskArea"] <= params["maxSingleArea"]
        and b["maskEcc"] >= params["minSingleEcc"]
    ]
    singleMask = np.isin(st.session_state.predictedLabels["single"], filteredIds).astype(np.uint8)

    # Морфология для single (сглаживание и удаление шумов)
    singleMask = morphology.remove_small_objects(
        measure.label(singleMask, connectivity=1),
        min_size=params.get("minSingleArea", 10)
    )
    singleMask = (singleMask > 0).astype(np.uint8)
    singleMask = cv2.morphologyEx(singleMask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    singleMask = cv2.morphologyEx(singleMask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # -------- Фильтрация bf --------
    # Лейблинг объектов биоплёнки
    labeled_bf = measure.label(biofilmMask, connectivity=1)
    filteredBfMask = np.zeros_like(biofilmMask, dtype=np.uint8)

    for region in measure.regionprops(labeled_bf):
        if params["minBfArea"] < region.area <= params["maxBfArea"]:
            filteredBfMask[tuple(zip(*region.coords))] = 1

    # Морфология для bf (сглаживание контуров)
    filteredBfMask = cv2.morphologyEx(filteredBfMask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    filteredBfMask = cv2.morphologyEx(filteredBfMask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    filteredBfMask = morphology.remove_small_holes(filteredBfMask.astype(bool), area_threshold=500)
    filteredBfMask = filteredBfMask.astype(np.uint8)

    return {
        "single": singleMask,
        "bf": filteredBfMask
    }
 

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
    
