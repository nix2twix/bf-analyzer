import streamlit as st

@st.cache_data(show_spinner=False, ttl=6000, max_entries=10)  
def calculateStatistics(objectsInfo, scale=0.05):
    """Расчет статистики по объектам"""
    stats = {}
    scale_factor = scale ** 2 

    for className, objList in objectsInfo.items():
        obj_count = len(objList)
        total_area_px = sum(obj["area"] for obj in objList)
        total_area_mkm = total_area_px * scale_factor

        stats[className] = {
            "count": obj_count,
            "total_area_px": total_area_px,
            "total_area_mkm": total_area_mkm
        }

    return stats