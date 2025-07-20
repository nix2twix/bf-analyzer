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
from PIL import Image, ImageFont, ImageDraw
from torch.utils.data import DataLoader

# === PROJECT SCRIPTS ===
from processingFunctions import makePatches, cropLineBelow, TestDataset
from processingFunctions import buildModel, loadCheckpoint, loadCellposeModel


# === SECONDARY FUNCTIONS ===
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# === PROCESSING BLOCK ===
# _img передавать как np.array иначе PIL не кэшится
@st.cache_data(show_spinner = False)
def segmentationImage(_img, imgName,
                      cellposeParams):
    width, height = _img.size
    imgPatches = []
    patchesInfo = []
    print(f"[INFO] START PROCESSING {imgName}...")
        
    if ((height > 512) or (width > 512)):
        imgPatches, patchesInfo = makePatches(_img, 
                                              patch_size = (512, 512),
                                              stride=(384,384))
    else:
        imgPatches.append(_img)
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
    
    origImgNP = np.array(_img) 
    cleaned_image = origImgNP.copy()
    cleaned_image[biofilmPredictions == 1] = 0 #black 
        
    print(f"[INFO] START CELLPOSE-SAM PROCESSING...")
    model_cp = loadCellposeModel() 
    singlePredictions, flows, styles = model_cp.eval(cleaned_image, 
                                                     channels=[0, 0], 
                                                     flow_threshold=cellposeParams[0], 
                                                     cellprob_threshold=cellposeParams[1])
    #singlePredictions = np.array(singlePredictions != 0, dtype=np.uint8)

    predictedLabels = {
    "single": singlePredictions,
    "bf": biofilmPredictions
    }
        
    print(f"PROCESSED SUCCESSFULLY!")
    return predictedLabels

@st.cache_data(show_spinner=False)    
def drawPicture(_origImg, predictedLabels):
    _origImg = _origImg.convert("RGBA") 

    biofilmPredictions = predictedLabels["bf"]
    singlePredictions = predictedLabels["single"]
    
    biofilm_mask = (biofilmPredictions == 1)
    bacteria_mask = (singlePredictions != 0)
    
    overlap = (biofilmPredictions == 1) & (singlePredictions == 1)
    singlePredictions[overlap] = 0
    
    biofilm_color = np.array([36, 179, 83, 178], dtype=np.uint8) 
    bacteria_color = np.array([184, 61, 245, 178], dtype=np.uint8)
    
    overlay = np.array(_origImg.copy())
    overlay[:,:,3] = 0

    if biofilm_mask.any():
        overlay[biofilm_mask,:] = biofilm_color
        
    if bacteria_mask.any():
        overlay[bacteria_mask,:] = bacteria_color
    
    overlayRGBA = Image.fromarray(overlay, mode="RGBA")    
    composite = Image.alpha_composite(_origImg, overlayRGBA)
   
    return composite

@st.cache_data(show_spinner=False)
def calculateStatistics(predictedLabels, scale = 0.05):
    biofilmPredictions = predictedLabels["bf"]
    singlePredictions = predictedLabels["single"]    

    biofilm_mask = (biofilmPredictions == 1)
    bacteria_mask = (singlePredictions != 0)
    
    labeled_bacteria = measure.label(singlePredictions)
    bacteria_count = labeled_bacteria.max()
    
    resultInfo = {
    "biofilm_area": int(np.sum(biofilm_mask)),
    "biofilm_mkm_area": int(np.sum(biofilm_mask)) * scale,
    "bacteria_count": int(bacteria_count),
    "bacteries_mkm_area": int(np.sum(bacteria_mask)) * scale
    }
    
    return resultInfo
  
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
        
    return singleBacteriesInfo

def filtrationObjects(origImg,
               predictedLabels, 
               params):
        
    width, height = origImg.size
    imgSize = width * height
    
    singlePredictions = predictedLabels["single"] 
    bfPredictions = predictedLabels["bf"]
    
    singleInfo = makeBacteriaInfo(predictedLabels)
    
    # SINGLE BACTERIES FILTRATION    
    filteredSingleMasks = singlePredictions.copy()
    for bacteria in singleInfo:
        if (bacteria["maskArea"] > params["maxSingleArea"]):
            filteredSingleMasks[singlePredictions == bacteria["maskNum"]] = 0
        elif (bacteria["maskArea"] <= params["minSingleArea"]):
            filteredSingleMasks[singlePredictions == bacteria["maskNum"]] = 0
        elif (bacteria["maskEcc"] < params["minSingleEcc"]):
            filteredSingleMasks[singlePredictions == bacteria["maskNum"]] = 0
 
    
    # BIOFILM FILTRATION
    mask_uint8 = (bfPredictions * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    labeled_mask = label(closed > 0, connectivity=2)
    
    filteredBfMask = bfPredictions.copy()
    for region in regionprops(labeled_mask):
        if (region.area > ((params["maxBfArea"] / 100) * imgSize)):
            filteredBfMask[labeled_mask == region.label] = 0
        elif (region.area <= ((params["minBfArea"]/ 100) * imgSize)):
            filteredBfMask[labeled_mask == region.label] = 0
    
    filteredLabels = {
    "single": filteredSingleMasks, 
    "bf": filteredBfMask
    }

    return filteredLabels
    
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
    
    
    
    