import numpy as np
import cv2
from skimage.measure import regionprops
import streamlit as st

@st.cache_data(show_spinner=False, ttl=6000, max_entries=10)
def getPredictedObjects(predictedBinMasks):
    """Получение связанных компонент для каждого класса"""
    predictedObjects = {}
    for className, binary_mask in predictedBinMasks.items():
        mask = binary_mask.astype(np.uint8)
        _, labeled_img = cv2.connectedComponents(mask, connectivity=8)
        predictedObjects[className] = labeled_img
    return predictedObjects

@st.cache_data(show_spinner=False, ttl=6000, max_entries=10)    
def prepareObjectInfo(predictedLabels, model_config):
    """
    Подготовка информации об объектах для фильтрации и статистики
    """
    objectsInfo = {}
    areaStats = {}

    for className, labeledMask in predictedLabels.items():
        # Пропускаем фон
        if "background" in className or className == "bg":
            continue
            
        props = regionprops(labeledMask.astype(np.int32))

        class_objects = [{
            "id": prop.label,
            "area": prop.area,
            "eccentricity": prop.eccentricity,
            "bbox": prop.bbox
        } for prop in props]

        objectsInfo[className] = class_objects
        areas = [prop.area for prop in props]
        if areas:
            areaStats[className] = {
                "min_area": max(10, int(np.min(areas))),
                "max_area": int(np.max(areas))
            }
        else:
            areaStats[className] = {"min_area": 0, "max_area": 1}
    
    return objectsInfo, areaStats

@st.cache_data(show_spinner=False, ttl=6000, max_entries=10) 
def prepareFilteredObjectInfo(filteredObjects):
    """
    Подготовка информации об отфильтрованных объектах для финальной статистики
    """
    filteredObjectsInfo = {}
    for className, labeledMask in filteredObjects.items():
        if "background" in className or className == "bg":
            continue
            
        props = regionprops(labeledMask.astype(np.uint8))
        class_objects = [{
            "id": prop.label,
            "area": prop.area,
            "eccentricity": prop.eccentricity
        } for prop in props]
        
        filteredObjectsInfo[className] = class_objects
    return filteredObjectsInfo

def groupObjectsByClass(binObjectsInfo, classLabels):
    """Группировка объектов по классам """
    objectsInfoByClass = {label: [] for label in classLabels}
    
    for objData in binObjectsInfo:
        objectsInfoByClass[objData['class']].append({
            "id": objData['id'],
            "area": objData['area'],
            "eccentricity": objData['eccentricity'],
            "bbox": objData['bbox']
        })
    
    return objectsInfoByClass