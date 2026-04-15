import cv2
import numpy as np
from skimage.measure import regionprops, label
from skimage import morphology

def smoothMaskFull(mask, morph_kernel=5, morph_iters=2, gauss_kernel=7):
    """Сглаживание маски морфологическими операциями и гауссом"""
    mask_uint8 = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    morph = mask_uint8.copy()
    for _ in range(morph_iters):
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    morph_float = morph.astype(np.float32)
    blurred = cv2.GaussianBlur(morph_float, (gauss_kernel, gauss_kernel), 0)
    smoothed = (blurred > 0.5).astype(np.uint8)
    return smoothed

def fillHolesMask(mask, area_thresh=200):
    """Заполнение маленьких дырок в маске"""
    mask_bool = mask.astype(bool)
    filled = morphology.remove_small_holes(mask_bool, area_threshold=area_thresh)
    return filled.astype(np.uint8)

def postprocessByProbs(predMasks, probMaps, classLabels, class_weights=None):
    """
    Постобработка с учетом вероятностей и весов классов
    """
    first_mask = next(iter(predMasks.values()))
    h, w = first_mask.shape
    
    # Обрабатываем каждый класс отдельно
    processedMask = {label: np.zeros((h, w), dtype=np.int32) for label in classLabels}
    all_objects = []
    next_id = 1
    
    # Определяем классы фона
    bg_labels = [name for name in classLabels.keys() 
                 if "background" in name or name == "bg"]
    
    # Для каждого не-фонового класса
    for classLabel in classLabels:
        if classLabel in bg_labels:
            continue
            
        # Бинарная маска для этого класса
        class_mask = predMasks.get(classLabel, 0) > 0
        if not np.any(class_mask):
            continue
            
        # Сглаживаем маску класса
        smoothed_mask = fillHolesMask(class_mask.astype(np.uint8), area_thresh=30)
        smoothed_mask = smoothMaskFull(smoothed_mask, morph_kernel=5, morph_iters=3, gauss_kernel=7)
        
        # Выделяем отдельные объекты в этом классе
        labeled_mask = label(smoothed_mask)
        regions = regionprops(labeled_mask)
        
        #print(f"\n[DEBUG] postprocessByProbs: class {classLabel} has {len(regions)} regions")
        
        for region in regions:
            objData = {
                'id': next_id,
                'class': classLabel,
                'area': region.area,
                'eccentricity': region.eccentricity,
                'bbox': region.bbox,
                'coords': region.coords,
                'probsCoefs': {label: 0 for label in classLabels}
            }
            
            # Вычисляем вероятности для объекта
            min_row, min_col, max_row, max_col = region.bbox
            objBbox = (labeled_mask[min_row:max_row, min_col:max_col] == region.label)
            
            for classLabel2 in classLabels:
                if classLabel2 in probMaps:
                    probBbox = probMaps[classLabel2][min_row:max_row, min_col:max_col]
                    obj_pixels = probBbox[objBbox]
                    if len(obj_pixels) > 0:
                        objData['probsCoefs'][classLabel2] = np.mean(obj_pixels)
            
            # Применяем веса если есть
            if class_weights:
                weighted_probs = {}
                for cls, prob in objData['probsCoefs'].items():
                    weight = class_weights.get(cls, 1.0)
                    weighted_probs[cls] = prob * weight
                bestClass = max(weighted_probs, key=weighted_probs.get)
            else:
                bestClass = max(objData['probsCoefs'], key=objData['probsCoefs'].get)
            
            objData['class'] = bestClass
            all_objects.append(objData)
            
            # Записываем в маску
            processedMask[bestClass][region.coords[:, 0], region.coords[:, 1]] = next_id
            
            next_id += 1
    
    # Выводим статистику
    class_counts = {}
    for objData in all_objects:
        class_counts[objData['class']] = class_counts.get(objData['class'], 0) + 1
    #print(f"[DEBUG] postprocessByProbs: objects per class: {class_counts}")
    #print(f"[DEBUG] postprocessByProbs: total objects: {len(all_objects)}")
    
    return processedMask, all_objects

def postprocessByClassFilters(
    predMask,
    probs,
    classLabels,
    postprocess_params,
    prob_threshold=0.0
):
    """
    Постобработка с фильтрацией по площади и эксцентриситету
    
    Args:
        predMask: исходная маска предсказаний (H, W)
        probs: словарь карт вероятностей для каждого класса
        classLabels: словарь имя_класса -> индекс
        postprocess_params: словарь с параметрами фильтрации для каждого класса
        prob_threshold: порог уверенности для переклассификации
    
    Returns:
        updatedMask: обновленная маска с учетом фильтров
    """
    updatedMask = predMask.copy()
    
    for class_name, class_idx in classLabels.items():
        if "background" in class_name or class_name == "bg":
            continue
            
        binaryMask = (updatedMask == class_idx)
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

                # Проверяем фильтры кандидата
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
                # Присваиваем фон
                bg_idx = classLabels.get("background", classLabels.get("bg", 0))
                updatedMask[obj_mask] = bg_idx

    return updatedMask