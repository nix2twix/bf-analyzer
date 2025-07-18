# === GENERAL VARIABLES ===
CHECKPOINTPATH = "final_model_epoch_350.pth"
PATTERN = r'\.(\d+)_(\d+)\.png$'

# === LIBRARIES GENERAL ===
import cv2  
import torch

import streamlit as st
import numpy as np

from skimage import measure
from skimage.measure import label, regionprops
from PIL import Image
from torch.utils.data import DataLoader
from io import BytesIO

# === PROJECT SCRIPTS ===
from processingFunctions import makePatches, cropLineBelow, TestDataset
from processingFunctions import buildModel, loadCheckpoint, loadCellposeModel


# === SECONDARY FUNCTIONS ===
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# === PROCESSING BLOCK ===
@st.cache_data(show_spinner = False)
def startProcessing(image_bytes, 
                    imgName
                    ):
    
    if image_bytes != None:
        img = Image.open(image_bytes)
        width, height = img.size
        imgPatches = []
        patchesInfo = []
        print(f"[INFO] START PROCESSING {imgName}...")
        
        if ((height > 512) or (width > 512)):
            img = cropLineBelow(img, countPx=128)
            width, height = img.size
            imgPatches, patchesInfo = makePatches(img, imgName, 
                                                  patch_size = (512, 512),
                                                  stride=(384,384))
        else:
            imgPatches.append(img)
            patchesInfo.append((0, 0, 0))
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] DEVICE IS {device}!")
        
        test_dataset = TestDataset(imgPatches, patchesInfo)
        test_loader = DataLoader(test_dataset,
                                    batch_size=1,
                                    shuffle=False)

        model = buildModel().to(device)   
        loadCheckpoint(model, CHECKPOINTPATH)
        model.eval()

        probsCount = np.zeros((height, width), dtype=float)
        biofilmProbs = np.zeros((height, width), dtype=float)
        biofilmPredictions = np.zeros((height, width), dtype=float)
    
        #probsCount = np.load("src/probsCount.npy")
        with torch.no_grad():
            for (images, patchesInfo) in test_loader:
                images = images.to(device)
                outputs = model(images)
                outputs = outputs.cpu()
                   
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
    
        origImgNP = np.array(img) 
        cleaned_image = origImgNP.copy()
        cleaned_image[biofilmPredictions == 1] = 0 #black 
        
        print(f"[INFO] START CELLPOSE-SAM PROCESSING...")
        model_cp = loadCellposeModel() 
        singlePredictions, flows, styles = model_cp.eval(cleaned_image, channels=[0, 0], flow_threshold=1, cellprob_threshold=2)
    
        # PREDICTIONS
        singlePredictions = np.array(singlePredictions != 0, dtype=np.uint8)
    
        biofilm_mask = (biofilmPredictions == 1)
        bacteria_mask = (singlePredictions == 1)
    
        overlap = (biofilmPredictions == 1) & (singlePredictions == 1)
        singlePredictions[overlap] = 0
    
        # SETTINGS FOR VIZUALIZATION
        biofilm_color = np.array([36, 179, 83, 255], dtype=np.uint8)  # RGBA
        bacteria_color = np.array([184, 61, 245, 255], dtype=np.uint8)
        difference_color = np.array([255, 125, 0, 255], dtype=np.uint8)
    
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        overlay[biofilm_mask] = biofilm_color
        overlay[bacteria_mask] = bacteria_color
    
        # SAVE 
        origRGBA = Image.fromarray(origImgNP).convert("RGBA")  
        alpha_mask = (overlay[..., :3] != 0).any(axis=-1)
        overlay[alpha_mask, 3] = 178  # 70%
        overlayRGBA = Image.fromarray(overlay, mode="RGBA")    
        composite = Image.alpha_composite(origRGBA, overlayRGBA)
    
        buf = BytesIO()
        composite.save(buf, format='PNG')
        processedImgBytes = buf.getvalue()
    
        labeled_bacteria = measure.label(singlePredictions)
        bacteria_count = labeled_bacteria.max()
        biofilm_area = np.sum(biofilm_mask)
    
        resultInfo = {
        "biofilm_area": int(np.sum(biofilm_mask)),
        "biofilm_mkm_area": int(np.sum(biofilm_mask)) * 0.05,
        "bacteria_count": int(bacteria_count),
        "bacteries_mkm_area": int(np.sum(bacteria_mask)) * 0.05
        }
        
        predictedLabels = {
        "single": singlePredictions, 
        "bf": biofilmPredictions
        }
        
        print(f"PROCESSED SUCCESSFULLY!")
    
        return processedImgBytes, resultInfo, predictedLabels
    else:
        return image_bytes
    
def startFiltration(processedImgBytes,
                    predictedLabels, 
                    minSingleArea,
                    maxSingleArea, 
                    minEcc,
                    minBfAreaPercent,
                    maxBfAreaPercent,
                    imgSize,
                    scale = 0.05):
    
    singlePredictions = predictedLabels[0] 
    bfPredictions = predictedLabels[1]
    
    # SINGLE BACTERIES FILTRATION
    props = measure.regionprops(singlePredictions)
    filteredSingleMasks = np.copy(singlePredictions)
    for prop in props:
        if (prop.area > maxSingleArea) or (prop.eccentricity < minEcc) or (prop.area < minSingleArea):
            filteredSingleMasks[filteredSingleMasks == prop.label] = 0
            labeled_bacteria = measure.label(singlePredictions)
    
    # BIOFILM FILTRATION
    mask_uint8 = (bfPredictions * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    
    labeled_mask = label(closed > 0, connectivity=2)
    filteredBfMask = np.zeros_like(bfPredictions)
    for region in regionprops(labeled_mask):
        if (region.area >= ((maxBfAreaPercent/ 100) * imgSize)) or (region.area < ((minBfAreaPercent/ 100) * imgSize)):
            filteredBfMask[labeled_mask == region.label] = 1
    
    resultInfo = {
    "biofilm_area": int(np.sum(bfPredictions)),
    "biofilm_mkm_area": int(np.sum(bfPredictions)) * scale,
    "bacteria_count": int(labeled_bacteria.max()),
    "bacteries_mkm_area": int(np.sum(singlePredictions)) * scale
    }
    
    predictedLabels = {
    "single": singlePredictions, 
    "bf": bfPredictions
    }
    return processedImgBytes, resultInfo, predictedLabels
    
if __name__ == "__main__":
    
    with open("1-BSE-1k-T1.bmp", "rb") as fh:
        uploaded_file = BytesIO(fh.read())
        
        result = startProcessing(uploaded_file,
                                "1-BSE-1k-T1.bmp",
                                50, 
                                2000, 
                                0.85)
    
        image = Image.open(BytesIO(result[0]))
        image.show()
    
    
    
    