import numpy as np
import streamlit as st

@st.cache_data(show_spinner=False, ttl=6000, max_entries=10)   
def filtrationObjects(objectsInfo, predictedObjects, params, model_config):
    """
    Фильтрация объектов на основе параметров
    params: словарь вида {
        'biofilm_area': {'min': 100, 'max': 100000}
    }
    """
    filteredObjects = {}
    
    # Получаем конфигурацию фильтрации из модели
    filtration_config = model_config.filtration_params
    '''
    print("\n" + "="*70)
    print("[DEBUG] filtrationObjects START")
    print("="*70)
    '''
    for className, objList in objectsInfo.items():
        if not objList or className not in predictedObjects:
            first_mask = next(iter(predictedObjects.values()))
            filteredObjects[className] = np.zeros_like(first_mask, dtype=np.int32)
            continue
            
        classMask = predictedObjects[className]
        all_ids = np.array([obj["id"] for obj in objList])
        areas = np.array([obj["area"] for obj in objList])
        eccentricities = np.array([obj.get("eccentricity", np.nan) for obj in objList])
        '''
        print(f"\n--- Class: {className} ---")
        print(f"  Total objects: {len(objList)}")
        print(f"  IDs: {all_ids}")
        print(f"  Areas: {areas}")
        print(f"  Eccentricities: {eccentricities}")
        '''
        valid_mask = np.ones(len(objList), dtype=bool)
        
        # Применяем фильтры
        for param_name, (min_default, max_default) in filtration_config.items():
            if className in param_name:
                #print(f"  Filter: {param_name} (default: {min_default}-{max_default})")
                
                # Получаем текущие значения из params
                if param_name in params:
                    current_min = params[param_name].get('min', min_default)
                    current_max = params[param_name].get('max', max_default)
                else:
                    current_min = min_default
                    current_max = max_default
                
                #print(f"    current range: {current_min} - {current_max}")
                
                if "area" in param_name:
                    area_condition = (areas >= current_min) & (areas <= current_max)
                    #print(f"    area condition: {area_condition}")
                    valid_mask = valid_mask & area_condition
                    
                elif "ecc" in param_name:
                    ecc_condition = (eccentricities >= current_min) & (eccentricities <= current_max)
                    ecc_condition = ecc_condition | np.isnan(eccentricities)
                    #print(f"    ecc condition: {ecc_condition}")
                    valid_mask = valid_mask & ecc_condition
                
        valid_ids = all_ids[valid_mask]
        
        if len(valid_ids) > 0:
            filtered_mask = np.where(np.isin(classMask, valid_ids), classMask, 0)
            #print(f"  Filtered mask pixels: {np.sum(filtered_mask > 0)}")
        else:
            filtered_mask = np.zeros_like(classMask, dtype=np.int32)
            #print(f"  Filtered mask pixels: 0")
        
        filteredObjects[className] = filtered_mask
    ''' 
    print("\n" + "="*70)
    print("[DEBUG] filtrationObjects END")
    print("="*70 + "\n")
    '''
    return filteredObjects