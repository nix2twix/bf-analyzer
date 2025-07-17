# === GENERAL VARIABLES ===
CHECKPOINTPATH = "final_model_epoch_350.pth"
PATTERN = r'\.(\d+)_(\d+)\.png$'

# === LIBRARIES GENERAL ===
import io   
import torch

import streamlit as st
import numpy as np

from skimage import measure
from PIL import Image
from cellpose import io
from torch.utils.data import DataLoader
from io import BytesIO

# === PROJECT SCRIPTS ===
from processingFunctions import makePatches, cropLineBelow, TestDataset
from processingFunctions import buildModel, loadCheckpoint, loadCellposeModel


# === SECONDARY FUNCTIONS ===
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def filterMasks(masks, probs, min_area, max_area, min_circularity=0.85):
    props = measure.regionprops(masks)
    areas = [prop.area for prop in props]
    filtered_probs = np.copy(probs)
    filtered_masks = np.copy(masks)
    for prop in props:
        ecc = prop.eccentricity 
        
        if (prop.area > max_area) or (prop.eccentricity < min_circularity) or (prop.area < min_area):
            filtered_masks[filtered_masks == prop.label] = 0
            filtered_probs[filtered_masks == prop.label] = 0.0

    return filtered_masks, filtered_probs


# === PROCESSING BLOCK ===
@st.cache_data(show_spinner = False)
def startProcessing(image_bytes, 
                    imgName,
                    min_area,
                    max_area, 
                    min_ecc):
    
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
        singlePredictions, flows, styles = model_cp.eval(cleaned_image, channels=[0, 0])

        singleProbs = 1 / (1 + np.exp(-singlePredictions))

        singlePredictions, singleProbs = filterMasks(singlePredictions, 
                                                        singleProbs, 
                                                        min_area, 
                                                        max_area, 
                                                        min_ecc)
    
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
        
        print(f"PROCESSED SUCCESSFULLY!")
    
        return processedImgBytes, resultInfo
    else:
        return image_bytes
    
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
    
    
    
    