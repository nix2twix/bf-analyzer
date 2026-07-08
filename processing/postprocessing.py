import cv2
import numpy as np
from skimage.measure import regionprops, label
from skimage import morphology
from PIL import Image


def smoothMask(mask, morph_kernel=3, morph_iters=1, gauss_kernel=3):
    """Сглаживание маски морфологическими операциями и гауссом"""
    mask_uint8 = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    morph = mask_uint8.copy()
    for _ in range(morph_iters):
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    morph_float = morph.astype(np.float32)
    blurred = cv2.GaussianBlur(morph_float, (gauss_kernel, gauss_kernel), 0)
    smoothed = (blurred > 0.9).astype(np.uint8)  # 0.9 для более агрессивного объединения
    return smoothed

# Алиас для обратной совместимости
smoothMaskFull = smoothMask

def fillHolesMask(mask, area_thresh=200):
    """Заполнение маленьких дырок в маске"""
    mask_bool = mask.astype(bool)
    filled = morphology.remove_small_holes(mask_bool, area_threshold=area_thresh)
    return filled.astype(np.uint8)

def postprocessByProbs(predMask, probMaps, classLabels, class_weights=None):
    """
    Постобработка с учетом вероятностей и весов классов
    class_weights: словарь весов для классов {название_класса: вес}
    """
    if class_weights is None:
        class_weights = {class_name: 1.0 for class_name in classLabels.keys()}
    
    # Бинарная маска для сглаживаемых классов
    smooth_classes = [v for k, v in classLabels.items() if k not in ["bg", "background"]]
    binMask = np.isin(predMask, smooth_classes).astype(np.uint8)
    #binMaskImg = Image.fromarray(binMask * 255)
    #binMaskImg.show() 

    binMask = smoothMask(binMask, morph_kernel=5, morph_iters=4, gauss_kernel=5) 
    binMask = fillHolesMask(binMask, area_thresh=300)
    
    #binMaskImg = Image.fromarray((binMask * 255).astype(np.uint8))
    #binMaskImg.show()

    binMaskLabeled = label(binMask)
    regions = regionprops(binMaskLabeled)
    
    binObjectsInfo = []
    for region in regions:
        objData = {
            'id': region.label,
            'class': None,
            'area': region.area,
            'eccentricity': region.eccentricity,
            'bbox': region.bbox,
            'coords': region.coords,
            'probsCoefs': {label: 0 for label in classLabels.keys()}
        }
        binObjectsInfo.append(objData)
    
    processedMask = np.zeros_like(predMask, dtype=np.uint8)
    
    for objData in binObjectsInfo:
        if objData['area'] < 200:
            coords = objData['coords']
            processedMask[coords[:, 0], coords[:, 1]] = 0
            continue
            
        min_row, min_col, max_row, max_col = objData['bbox']
        objBbox = (binMaskLabeled[min_row:max_row, min_col:max_col] == objData['id'])
        for class_name in classLabels.keys():
            if class_name in probMaps:
                probBbox = probMaps[class_name][min_row:max_row, min_col:max_col]
                objProbs = probBbox[objBbox]
                objData['probsCoefs'][class_name] = np.mean(objProbs) if len(objProbs) > 0 else 0
        
        weighted_probs = {}
        for class_name, prob in objData['probsCoefs'].items():
            weight = class_weights.get(class_name, 1.0)
            weighted_probs[class_name] = prob * weight

        bestClass = max(weighted_probs, key=weighted_probs.get)
        objData['class'] = bestClass
        
        coords = objData['coords']
        processedMask[coords[:, 0], coords[:, 1]] = classLabels[bestClass]
    
    return processedMask


def postprocessByClassFilters(
    predMask,
    probs,
    classLabels,
    postprocess_params,
    prob_threshold=0.0
):
    """
    Постобработка с фильтрацией по площади и эксцентриситету
    """
    updatedMask = predMask.copy()
    classIdxList = list(classLabels.values())

    for classIdx in classIdxList:
        class_name = [k for k, v in classLabels.items() if v == classIdx][0]
        if "background" in class_name or class_name == "bg":
            continue
            
        binaryMask = (updatedMask == classIdx)
        if np.count_nonzero(binaryMask) == 0:
            continue

        labeled = label(binaryMask, connectivity=2)
        for region in regionprops(labeled):
            obj_mask = (labeled == region.label)
            region_coords = np.where(obj_mask)

            # 1. Проверка фильтров текущего класса
            passes_filters = True
            
            if postprocess_params and class_name in postprocess_params:
                params = postprocess_params[class_name]
                
                if "area" in params:
                    min_area, max_area = params["area"]
                    if not (min_area <= region.area <= max_area):
                        passes_filters = False
                
                if passes_filters and "ecc" in params:
                    min_ecc, max_ecc = params["ecc"]
                    if not (min_ecc <= region.eccentricity <= max_ecc):
                        passes_filters = False

            if passes_filters:
                continue  

            # 2. Проверяем следующий класс
            class_scores = {}
            for name, idx in classLabels.items():
                if name in probs and "background" not in name and name != "bg":
                    prob_values = probs[name][region_coords]
                    class_scores[idx] = np.mean(prob_values) if len(prob_values) > 0 else 0.0

            sorted_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)

            assigned = False
            for candidate_class, score in sorted_classes:
                if score < prob_threshold:
                    continue

                candidate_name = [k for k, v in classLabels.items() if v == candidate_class][0]

                passes_candidate = True
                
                if postprocess_params and candidate_name in postprocess_params:
                    params = postprocess_params[candidate_name]
                    
                    if "area" in params:
                        min_area, max_area = params["area"]
                        if not (min_area <= region.area <= max_area):
                            passes_candidate = False
                    
                    if passes_candidate and "ecc" in params:
                        min_ecc, max_ecc = params["ecc"]
                        if not (min_ecc <= region.eccentricity <= max_ecc):
                            passes_candidate = False

                if passes_candidate:
                    updatedMask[obj_mask] = candidate_class
                    assigned = True
                    break

            if not assigned:
                bg_idx = classLabels.get("background", classLabels.get("bg", 0))
                updatedMask[obj_mask] = bg_idx

    return updatedMask