# === LIBRARIES GENERAL ===
import streamlit as st
import numpy as np
from PIL import Image
from typing import Optional, Dict

# === PROJECT SCRIPTS ===
from models.configs import MODEL_CONFIGS, ModelConfig
from segmentation.filtration import filtrationObjects

class StateManager:
    """Управляет состоянием приложения"""
    def __init__(self):
        self.state = st.session_state
        self._initialize()
    
    def _initialize(self):
        """Инициализация состояния при первом запуске"""
        if "initialized" not in self.state:
            self.reset_all()
            self.state.initialized = True
            self.state.segmentation_in_progress = False
            self.state.manual_scale_set = False
            self.state.apply_postsegmentation = True
            self.state.postprocessed_masks = None
            self.state.raw_masks = None
            self.state.probs = None
        if "export_dirty" not in self.state:
            self.state.export_dirty = True

    def reset_all(self):
        """Полный сброс состояния"""
        # IMAGE
        self.state.imageName = None
        self.state.uploadedImage = None
        self.state.imgWidth = 0
        self.state.imgHeight = 0
        self.state.scaleInfo = None
        self.state.imgScale = None
        self.state.scaleText = None
        self.state.infoLineHeight = None

        self.state.postprocessed_masks = None
        self.state.raw_masks = None
        self.state.probs = None
        
        # SCALE
        self.state.manual_scale_set = False

        # PREDS
        self.state.probThreshold = 0.5
        self.state.predictedObjects = None
        self.state.filteredObjects = None
        self.state.objectsInfo = None
        self.state.filteredObjectsInfo = None
        self.state.modelType = "Bacillus"
        
        # FILTRATION 
        self.state.filtration_params = {}
        self.state.slider_ranges = {}  
        self.state.area_stats = {}
        
        # VIZUALIZATION
        self.state.isShowIntermediate = True
        self.state.classColors = self._get_model_colors("Bacillus")
        
        # EXPORT
        self.state.polygonsCVAT = None
        self.state.uploadedAnnName = None
        self.state.uploadedAnn = None
        self.state.zipBuffer = None
        self.state.export_dirty = True
        
        # MODEL
        self.state.model_config = self.get_config()
        self._init_dynamic_filtration_settings()
        
        # segmentation SETTINGS
        self.state.apply_postsegmentation = True
        
        # CACHE
        self._clear_image_cache()
    
    def _clear_image_cache(self):
        """Очистка кэша изображений в session_state"""
        if "image_cache" in st.session_state:
            st.session_state.image_cache = {}

    def invalidate_export(self):
        """Invalidate archives whenever their source masks change."""
        self.state.polygonsCVAT = None
        self.state.zipBuffer = None
        self.state.export_dirty = True
        
    def _init_dynamic_filtration_settings(self):
        """
        Первоначальная инициализация параметров фильтрации по конфигу
        """
        config = self.get_config()
        if not config:
            return
        
        for param_name, (min_val, max_val) in config.filtration_params.items():
            if param_name not in self.state.slider_ranges:
                self.state.slider_ranges[param_name] = (min_val, max_val)
        
            if param_name not in self.state.filtration_params:
                self.state.filtration_params[param_name] = {
                'min': min_val,
                'max': max_val
                }
    
 
    def toggle_postsegmentation(self, value: bool):
        """Переключает между постобработанными и сырыми масками"""
        if self.state.apply_postsegmentation != value:
            self.state.apply_postsegmentation = value
        
            # Проверяем, что маски существуют
            if self.state.postprocessed_masks is not None and self.state.raw_masks is not None:
                if value:
                    self.state.predictedObjects = self.state.postprocessed_masks
                else:
                    self.state.predictedObjects = self.state.raw_masks
            
                # Пересчитываем статистику для новых масок
                if self.state.predictedObjects is not None and self.state.model_config:
                    from segmentation.objects import prepareObjectInfo
                    objects_info, area_stats = prepareObjectInfo(
                        self.state.predictedObjects, 
                        self.state.model_config
                    )
                    self.update_area_stats(objects_info, area_stats, preserve_filters=True)
                    self.apply_filtration()
                    self._clear_image_cache()
    
    def get_apply_postsegmentation(self) -> bool:
        """Возвращает текущий статус постобработки"""
        return self.state.get("apply_postsegmentation", True)
                
    def update_slider_range(self, param_name: str, min_val: float, max_val: float, preserve_value: bool = False):
        """Обновление диапазона слайдера"""
        self.state.slider_ranges[param_name] = (min_val, max_val)
        if not preserve_value or param_name not in self.state.filtration_params:
            return

        current = self.state.filtration_params[param_name]
        current_min = max(min_val, min(current.get("min", min_val), max_val))
        current_max = max(min_val, min(current.get("max", max_val), max_val))
        if current_min > current_max:
            current_min = current_max

        self.state.filtration_params[param_name] = {"min": current_min, "max": current_max}
    
    def update_filter_parameter(self, param_name: str, value: dict):
        """Обновление параметра фильтрации (ожидает словарь с min/max)"""
        self.state.filtration_params[param_name] = value
    
    def update_filtration_params(self, ui_params: Dict) -> bool:
        """Update filters and report whether their values changed."""
        changed = False
        for param_name, (min_val, max_val) in ui_params.items():
            value = {
                'min': min_val,
                'max': max_val
            }
            if self.state.filtration_params.get(param_name) != value:
                self.state.filtration_params[param_name] = value
                changed = True
        return changed
    
    def get_slider_range(self, param_name: str, default=(100, 1000)) -> tuple:
        """Получить диапазон слайдера"""
        return self.state.slider_ranges.get(param_name, default)
    
    def get_filtration_params(self) -> Dict:
        """Получить текущие параметры фильтрации"""
        return self.state.filtration_params.copy()
    
    def _get_model_colors(self, model_name: str) -> Dict:
        """Получить цвета для модели"""
        config = MODEL_CONFIGS.get(model_name)
        if config:
            return config.class_colors
        return {
            "biofilm": (36, 179, 83, 178),
            "intermediate": (221, 255, 51, 178),
            "single": (184, 61, 245, 178)
        }
    
    def set_model(self, model_name: str) -> bool:
        """Смена модели - динамически"""
        if model_name == self.state.get("modelType"):
            return False

        self.state.modelType = model_name
        self.state.model_config = self.get_config()
        
        # Обновляем цвета
        self.state.classColors = self._get_model_colors(model_name)
        
        # Переинициализируем параметры фильтрации
        self._init_dynamic_filtration_settings()
        
        self.reset_results()
        self.state.last_uploaded_file = None
        self.state.last_uploaded_ann = None
        self._clear_image_cache()

        return True
    
    def has_results(self) -> bool:
        return self.state.get("predictedObjects") is not None
    
    def reset_results(self):
        self.state.predictedObjects = None
        self.state.filteredObjects = None
        self.state.objectsInfo = None
        self.state.filteredObjectsInfo = None
        self.state.isShowIntermediate = True
        self.invalidate_export()
        self.state.postprocessed_masks = None
        self.state.raw_masks = None
        self.state.probs = None
        self._clear_image_cache()
    
    def get_config(self) -> Optional[ModelConfig]:
        return MODEL_CONFIGS.get(self.state.get("modelType"))
    
    def set_image(self, uploaded_file):
        self.state.imageName = uploaded_file.name
        self.state.uploadedImage = Image.open(uploaded_file).convert("L")
        self.state.imgWidth, self.state.imgHeight = self.state.uploadedImage.size

        self.state.scaleInfo = None
        self.state.imgScale = None
        self.state.scaleText = None
        self.state.infoLineHeight = None
        self.state.manual_scale_set = False 
        self.state.postprocessed_masks = None
        self.state.raw_masks = None
        self.state.probs = None
        self.reset_results()
    
    def update_scale(self, scale_data: tuple, auto_scale: float):
        if scale_data is not None:
            self.state.scaleInfo = scale_data
            self.state.imgScale = auto_scale
        
            if scale_data:
                self.state.scaleText = f"{scale_data[4]} (μm) / {scale_data[2]} (px) = {auto_scale:.2f}"
                self.state.infoLineHeight = self.state.imgHeight - scale_data[0]
        else:
            if self.state.scaleInfo is None:
                self.state.scaleText = "1 (px) / 1 (μm) = 1"
                self.state.infoLineHeight = 0
                self.state.imgScale = 1

    def update_area_stats(self, objects_info: Dict, area_stats: Dict, preserve_filters: bool = False):
        """Обновление статистики с динамическими диапазонами"""
        self.state.objectsInfo = objects_info
        self.state.filteredObjectsInfo = objects_info
        self._update_slider_ranges_from_stats(area_stats, preserve_filters)
        self._update_eccentricity_ranges(objects_info, preserve_filters)
    
    def _update_slider_ranges_from_stats(self, area_stats: Dict, preserve_filters: bool = False):
        """Динамическое обновление диапазонов на основе статистики"""
        config = self.get_config()
        if not config:
            return

        for param_name in list(self.state.slider_ranges.keys()):
            if "area" not in param_name:
                continue
        
            class_name = param_name.replace("_area", "")

            if class_name in area_stats:
                detected_min = area_stats[class_name]["min_area"]
                detected_max = area_stats[class_name]["max_area"]
        
                # Базовые значения
                min_val = max(detected_min, 10)
                max_val = detected_max

                # Если все объекты одного размера, расширяем диапазон для удобства
                if min_val >= max_val:
                    min_val = max(10, min_val - 100)
                    max_val = max_val + 100
            
                # Обновляем диапазон слайдера
                self.update_slider_range(param_name, min_val, max_val, preserve_filters)
            
                # Обновляем filtration_params (храним оба значения)
                if not preserve_filters:
                    self.state.filtration_params[param_name] = {
                        'min': min_val,
                        'max': max_val
                    }
                    
    def _update_eccentricity_ranges(self, objects_info: Dict, preserve_filters: bool = False):
        """Обновление диапазонов для эксцентриситета на основе объектов"""
        config = self.get_config()
        if not config:
            return
    
        for param_name in list(self.state.slider_ranges.keys()):
            if "ecc" not in param_name:
                continue
        
            class_name = param_name.replace("_ecc", "")
    
            ecc_values = []
            if class_name in objects_info:
                for obj in objects_info[class_name]:
                    ecc = obj.get("eccentricity")
                    if ecc is not None and not np.isnan(ecc):
                        ecc_values.append(ecc)
    
            if ecc_values:
                detected_min = min(ecc_values)
                detected_max = max(ecc_values)
            
                min_val = detected_min
                max_val = detected_max

                # Если все объекты одного размера, расширяем диапазон
                if min_val >= max_val:
                    min_val = max(0.0, min_val - 0.1)
                    max_val = min(1.0, max_val + 0.1)
        
                # Обновляем диапазон слайдера
                self.update_slider_range(param_name, min_val, max_val, preserve_filters)
            
                # Обновляем filtration_params
                if not preserve_filters:
                    self.state.filtration_params[param_name] = {
                        'min': min_val,
                        'max': max_val
                    }
                
   
    def apply_filtration(self):
        """Применение фильтрации к predictedObjects"""
        if self.state.predictedObjects is not None and self.state.objectsInfo is not None:
            config = self.get_config()

            if not self.state.filtration_params:
                self._init_dynamic_filtration_settings()

            self.state.filteredObjects = filtrationObjects(
                objectsInfo=self.state.objectsInfo,
                predictedObjects=self.state.predictedObjects,
                params=self.state.filtration_params,
                model_config=config 
            )
    
            self.state.filteredObjectsInfo = None
            self.invalidate_export()
            # Очищаем кэш изображений при изменении фильтрации
            self._clear_image_cache()
        elif self.state.predictedObjects is not None:
            self.state.filteredObjects = self.state.predictedObjects.copy()
            self.state.filteredObjectsInfo = None
            self.invalidate_export()
            self._clear_image_cache()
