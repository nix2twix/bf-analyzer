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

def getOverlaysFromMask(predictedObjects, objectsInfo, scale):
    overlays = []
    for className, labeled_img in predictedObjects.items():
        for objectId in range(1, labeled_img.max() + 1):
            index = {}
            for class_name, objs in objectsInfo.items():
                for obj in objs:
                    index[(class_name, obj["id"])] = obj
    
            mask = (labeled_img == objectId).astype(np.uint8)

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_NONE,
            )

            if len(contours) == 0:
                continue

            path = ""

            for contour in contours:
                if len(contour) < 3:
                    continue

                points = contour[:, 0, :]

                path += f"M {points[0][0]} {points[0][1]}"

                for x, y in points[1:]:
                    path += f" L {x} {y}"

                path += " Z"

            areaInPx = index.get((className, objectId), {}).get("area")
            areaScaled = areaInPx * (scale ** 2)

            overlays.append({
                "id": f"{objectId}_{className}",
                "type": "path",
                "class": className,
                "data": {
                    "d": path,
                },
                "tooltip": (
                    f"ID: {objectId}\n",
                    f"Class: {className}\n",
                    f"Area: {areaScaled:.2f} μm²"
                ),
            })
    return overlays

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