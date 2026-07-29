import torch
import numpy as np
import streamlit as st

from torch.utils.data import DataLoader
from .preprocessing import cropLineBelow, makePatches
from .postprocessing import (
    smoothMask, 
    fillHolesMask, 
    postprocessByProbs, 
    postprocessByClassFilters
)
from utils.dataset import TestDataset
from models.model import buildModel, loadCheckpoint


def apply_postsegmentation_to_masks(processedMask, probs, model_config, threshold=0):
    """
    Применяет постобработку к маскам с использованием новых функций
    """
    import cv2
    import numpy as np
    
    height, width = None, None
    for mask in processedMask.values():
        if mask is not None:
            height, width = mask.shape
            break
    
    if height is None:
        return processedMask
    
    # Собираем unified mask из нумерованных масок
    unified_mask = np.zeros((height, width), dtype=np.int32)
    for class_name, labeled_mask in processedMask.items():
        if class_name in model_config.class_labels and "background" not in class_name:
            class_idx = model_config.class_labels[class_name]
            unified_mask[labeled_mask > 0] = class_idx
    
    processed_mask = postprocessByProbs(
            predMask=unified_mask,
            probMaps=probs,
            classLabels=model_config.class_labels
    )
    print(f"[INFO] Postprocessed by probabilities")
    
    # Шаг 2: Сглаживание и заполнение дыр для финальной маски
    binMask = (processed_mask > 0).astype(np.uint8)
    binMask = smoothMask(binMask, morph_kernel=3, morph_iters=3, gauss_kernel=3)
    binMask = fillHolesMask(binMask, area_thresh=200)
    
    # Применяем сглаженную маску
    processed_mask[binMask == 0] = 0
    
    # Шаг 3: Фильтрация по площади и эксцентриситету
    if hasattr(model_config, 'postprocess_params') and model_config.postprocess_params:
        processed_mask = postprocessByClassFilters(
            predMask=processed_mask,
            probs=probs,
            classLabels=model_config.class_labels,
            postprocess_params=model_config.postprocess_params,
            prob_threshold=threshold
        )
    
    # Конвертируем обратно в словарь нумерованных масок
    current_processed_masks = {}
    for name, class_idx in model_config.class_labels.items():
        if "background" in name or name == "bg":
            continue
        binary_mask = (processed_mask == class_idx).astype(np.uint8)
        _, labeled = cv2.connectedComponents(binary_mask, connectivity=8)
        current_processed_masks[name] = labeled
    
    return current_processed_masks


def simple_labeling(processedMask):
    """Только нумерация объектов без постобработки"""
    import cv2
    import numpy as np
    
    labeled_masks = {}
    for class_name, mask in processedMask.items():
        if "background" in class_name or class_name == "bg":
            labeled_masks[class_name] = np.zeros_like(mask, dtype=np.uint32)
            continue
        
        binary_mask = (mask > 0).astype(np.uint8)
        _, labeled = cv2.connectedComponents(binary_mask, connectivity=8)
        labeled_masks[class_name] = labeled.astype(np.uint32)
    
    return labeled_masks


@st.cache_data(show_spinner=False, ttl=6000, max_entries=10) 
def segmentationImage(
    uploaded_file, 
    INFLINEPX, 
    width, 
    height, 
    imgName,
    model_config,
    threshold=0
):
    """
    Универсальная функция сегментации для любой модели.
    Возвращает оба варианта: с постобработкой и без.
    
    Returns:
        postprocessed_masks: словарь нумерованных масок с постобработкой
        raw_masks: словарь нумерованных масок без постобработки
        probs: словарь карт вероятностей для каждого класса
    """
    import time
    
    origImg = uploaded_file
    
    # Обрезка информационной строки снизу
    croppedImg = cropLineBelow(origImg, INFLINEPX)
    croppedW, croppedH = croppedImg.size

    # Фрагментирование с шагом stride
    if height > 512 and width > 512:
        imgPatches, patchesInfo = makePatches(
            croppedImg, patch_size=(512, 512), stride=(256, 256)
        )
    else:
        imgPatches = [croppedImg]
        patchesInfo = [(0, 0, 0)]

    # Устройство (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] DEVICE IS {device}")
    print(f"[INFO] MODEL: {model_config.display_name}")

    # Загрузка модели
    model = buildModel(
        classesCount=model_config.num_classes,
        encoderName=model_config.encoder_name,
        encoderWeights=model_config.encoder_weights,
        activation=model_config.activation
    ).to(device)
    
    loadCheckpoint(model, model_config.checkpoint_path)
    model.eval()

    # Подготовка данных
    test_dataset = TestDataset(imgPatches, patchesInfo)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Инициализация карт вероятностей
    probs = {name: np.zeros((croppedH, croppedW), dtype=np.float32) 
             for name in model_config.class_names}
    count = np.zeros((croppedH, croppedW), dtype=np.float32)

    # Прогресс-бар
    total_patches = len(test_loader)
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0, text="Preparing for inference...")
        time_text = st.empty()
    
    start_time = time.time()
    
    with torch.no_grad():
        for idx, (images, patch_info) in enumerate(test_loader):
            progress = (idx + 1) / total_patches
            elapsed = time.time() - start_time
            
            if idx > 0:
                avg_time_per_patch = elapsed / (idx + 1)

            progress_bar.progress(progress, text=f"Processing patch {idx + 1}/{total_patches}")
            
            if idx > 0:
                time_text.caption(f"⏱️ Elapsed: {elapsed:.1f}s | Speed: {avg_time_per_patch:.1f}s/patch")
            else:
                time_text.caption("⏱️ Starting inference...")
            
            images = images.to(device)
            outputs = torch.softmax(model(images), dim=1).cpu()

            for img_idx in range(images.size(0)):
                x = patch_info[0].item()
                y = patch_info[1].item()

                for classIdx, name in enumerate(model_config.class_names):
                    prob = outputs[img_idx, classIdx].numpy()
                    h_out, w_out = prob.shape  
                    probs[name][y:y+h_out, x:x+w_out] += prob 

                count[y:y+h_out, x:x+w_out] += 1
    
    progress_container.empty()
    
    # Нормализация вероятностей
    count += 1e-9
    for name in probs:
        probs[name] /= count
 
    # Получение маски предсказания
    stackedProbs = np.stack([probs[name] for name in model_config.class_names], axis=-1)   
    maxProbs = np.max(stackedProbs, axis=-1)
    predMask = np.argmax(stackedProbs, axis=-1)
    predMask[maxProbs < threshold] = 0

    # Преобразуем predMask в словарь бинарных масок
    binary_masks = {}
    for i, name in enumerate(model_config.class_names):
        binary_masks[name] = (predMask == i).astype(np.uint8)
    
    # Получаем оба варианта: с постобработкой и без
    raw_masks = simple_labeling(binary_masks)
    postprocessed_masks = apply_postsegmentation_to_masks(binary_masks, probs, model_config, threshold)
    
    # Формирование итоговых нумерованных масок для каждого класса
    def expand_to_full_size(masks_dict):
        """Расширяет маски до исходного размера"""
        full_masks = {}
        for name, labeled_mask in masks_dict.items():
            mask_full = np.zeros((height, width), dtype=np.uint32)
            mask_full[:croppedH, :croppedW] = labeled_mask
            full_masks[name] = mask_full
        # Добавляем фон
        for name in model_config.class_labels.keys():
            if "background" in name or name == "bg":
                full_masks[name] = np.zeros((height, width), dtype=np.uint32)
                break
        return full_masks
    
    raw_masks_full = expand_to_full_size(raw_masks)
    postprocessed_masks_full = expand_to_full_size(postprocessed_masks)
    
    print(f"[INFO] segmentation {imgName} DONE!")
    return postprocessed_masks_full, raw_masks_full, probs