import torch
import numpy as np
from torch.utils.data import DataLoader
from stqdm import stqdm
import streamlit as st
from .preprocessing import cropLineBelow, makePatches
from .postprocessing import postprocessByClassFilters, postprocessByProbs
from src.dataset import TestDataset
from model.model import buildModel, loadCheckpoint

@st.cache_data(show_spinner=False, ttl=6000, max_entries=10) 
def segmentationImage(
    uploaded_file, 
    INFLINEPX, 
    width, 
    height, 
    imgName,
    model_config,
    threshold=0.5
):
    """
    Универсальная функция сегментации для любой модели
    
    Args:
        uploaded_file: загруженное изображение (PIL Image)
        INFLINEPX: высота информационной строки для обрезки
        width: ширина исходного изображения
        height: высота исходного изображения
        imgName: имя изображения
        model_config: конфигурация модели (ModelConfig)
        threshold: порог уверенности
    
    Returns:
        predictedLabels: словарь нумерованных масок для каждого класса (каждый объект имеет уникальный ID в пределах класса)
        probs: словарь карт вероятностей для каждого класса
    """
    import cv2
    from skimage.measure import label as skimage_label
    
    origImg = uploaded_file
    
    # Обрезка информационной строки снизу
    croppedImg = cropLineBelow(origImg, INFLINEPX)
    croppedW, croppedH = croppedImg.size

    # Фрагментирование с шагом stride
    if height > 512 and width > 512:
        imgPatches, patchesInfo = makePatches(
            croppedImg, patch_size=(512, 512), stride=(128, 128)
        )
    else:
        imgPatches = [croppedImg]
        patchesInfo = [(0, 0, 0)]

    # Устройство (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] DEVICE IS {device}")
    print(f"[INFO] MODEL: {model_config.display_name}")
    print(f"[INFO] CLASSES: {model_config.class_names}")

    # Загрузка модели из конфига
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

    # Инференс
    with torch.no_grad():
        for images, patch_info in stqdm(test_loader, desc="Processing patches"):
            images = images.to(device)
            outputs = torch.softmax(model(images), dim=1).cpu()

            for idx in range(images.size(0)):
                x = patch_info[0].item()
                y = patch_info[1].item()

                for classIdx, name in enumerate(model_config.class_names):
                    prob = outputs[idx, classIdx].numpy()
                    h_out, w_out = prob.shape  

                    probs[name][y:y+h_out, x:x+w_out] += prob 

                count[y:y+h_out, x:x+w_out] += 1
        
    # Нормализация вероятностей
    count += 1e-9  # избегаем деления на ноль
    for name in probs:
        probs[name] /= count
 
    # Получение маски предсказания
    stackedProbs = np.stack([probs[name] for name in model_config.class_names], axis=-1)   
    maxProbs = np.max(stackedProbs, axis=-1)
    predMask = np.argmax(stackedProbs, axis=-1)
    predMask[maxProbs < threshold] = 0  # применяем порог

    # Инициализация итоговой маски
    current_mask = predMask.copy()
    
    # Шаг 1: Постобработка по вероятностям с весами классов
    if hasattr(model_config, 'class_weights') and model_config.class_weights:
        from .postprocessing import postprocessByProbs
        
        # Преобразуем predMask в словарь бинарных масок для postprocessByProbs
        binary_masks = {}
        for i, name in enumerate(model_config.class_names):
            binary_masks[name] = (current_mask == i).astype(np.uint8)
        
        processed_masks, all_objects = postprocessByProbs(
            predMasks=binary_masks,
            probMaps=probs,
            classLabels=model_config.class_labels,
            class_weights=model_config.class_weights
        )
        current_processed_masks = processed_masks
        print(f"[INFO] Postprocessed by probabilities: {len(all_objects)} objects detected")
    else:
        # Если нет постобработки по вероятностям, создаем нумерованные маски из predMask
        current_processed_masks = {}
        for name, class_idx in model_config.class_labels.items():
            if "background" in name or name == "bg":
                continue
            binary_mask = (current_mask == class_idx).astype(np.uint8)
            _, labeled = cv2.connectedComponents(binary_mask, connectivity=8)
            current_processed_masks[name] = labeled
    
    # Шаг 2: Фильтрация по площади и эксцентриситету
    if hasattr(model_config, 'postprocess_params') and model_config.postprocess_params:
        # Конвертируем нумерованные маски в единую для postprocessByClassFilters
        unified_mask = np.zeros((croppedH, croppedW), dtype=np.int32)
        for name, labeled_mask in current_processed_masks.items():
            if name in model_config.class_labels:
                class_idx = model_config.class_labels[name]
                unified_mask[labeled_mask > 0] = class_idx
        
        # Применяем фильтрацию
        unified_mask = postprocessByClassFilters(
            predMask=unified_mask,
            probs=probs,
            classLabels=model_config.class_labels,
            postprocess_params=model_config.postprocess_params,
            prob_threshold=threshold
        )
        
        # Конвертируем обратно в нумерованные маски
        current_processed_masks = {}
        for name, class_idx in model_config.class_labels.items():
            if "background" in name or name == "bg":
                continue
            binary_mask = (unified_mask == class_idx).astype(np.uint8)
            _, labeled = cv2.connectedComponents(binary_mask, connectivity=8)
            current_processed_masks[name] = labeled
        
        print(f"[INFO] Postprocessed by class filters applied")
    
    # Формирование итоговых нумерованных масок для каждого класса
    predictedLabels = {}
    
    # Добавляем обработанные классы
    for name, labeled_mask in current_processed_masks.items():
        mask_full = np.zeros((height, width), dtype=np.uint32)
        mask_full[:croppedH, :croppedW] = labeled_mask
        predictedLabels[name] = mask_full
    
    # Добавляем фон
    for name in model_config.class_labels.keys():
        if "background" in name or name == "bg":
            predictedLabels[name] = np.zeros((height, width), dtype=np.uint32)
            break
    
    print(f"[INFO] PROCESSING {imgName} DONE!")
    return predictedLabels, probs