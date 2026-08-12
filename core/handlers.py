# === LIBRARIES GENERAL ===
import streamlit as st
import numpy as np

# === PROJECT SCRIPTS ===
from core.stateManager import StateManager
from segmentation.inference import segmentationImage
from segmentation.objects import (
    prepareObjectInfo
)
from utils.drawing import checkSize, correctSize
from utils.autoscale import estimateScale
from utils.importer import loadMasksFromZip
from utils.exporter import makeCVATbackup, saveResultsAsZip


class AppHandlers:
    """Обработчики событий приложения"""
    def __init__(self, state_manager: StateManager):
        self.state = state_manager
    
    def handle_file_upload(self, uploaded_file):
        """Обработка загрузки файла"""
        if uploaded_file is None:
            self.state.reset_all()
            return
        
        if uploaded_file.name != self.state.state.imageName:
            self.state.set_image(uploaded_file)
            self._handle_image_padding()
            self.handle_scale_detection()

    def _handle_image_padding(self):
        """Обработка паддинга изображения"""
        wPad, hPad = checkSize(self.state.state.uploadedImage, min_size=512)
        if (wPad != self.state.state.imgWidth) or (hPad != self.state.state.imgHeight):
            paddedImg = correctSize(self.state.state.uploadedImage, wPad, hPad)
            st.warning(f"Image will be padded from {self.state.state.imgWidth}x{self.state.state.imgHeight} "
                      f"to {wPad}x{hPad} for segmentation (multiples of 32).")
            self.state.state.uploadedImage = paddedImg
            self.state.state.imgWidth, self.state.state.imgHeight = paddedImg.size
    
    def handle_scale_detection(self, force=False):
        """Обработка определения масштаба"""
        if not force and self.state.state.get("manual_scale_set", False):
           return False
    
        if self.state.state.uploadedImage is not None:
            tempArrImg = np.array(self.state.state.uploadedImage, dtype='uint8')
            autoScale, scaleData = estimateScale(tempArrImg)
            
            if autoScale is not None:
                self.state.update_scale(scaleData, autoScale)
                self.state.state.manual_scale_set = False
                
                print(f"[DEBUG] Auto scale detected: {autoScale}")
                return True
            else:
                self.state.update_scale(None, None)
                return False
        
        return False
    
    def handle_annotation_apply(self, uploaded_ann):
        """Применение загруженной аннотации"""
        if uploaded_ann is None:
            return False

        # Загружаем аннотации с передачей model_config
        self.state.state.uploadedAnnName = uploaded_ann.name
        self.state.state.predictedObjects = loadMasksFromZip(
            uploaded_ann, 
            self.state.state.imgWidth, 
            self.state.state.imgHeight,
            self.state.state.model_config
        )
        self.state.state.uploadedAnn = uploaded_ann

        # Сбрасываем кэш статистики
        self.state.state.objectsInfo = None
        self.state.state.areaStats = None
        self.state.state.filteredObjects = None
        self.state.state.filteredObjectsInfo = None

        return True

    def handle_segmentation(self):
        """Обработка сегментации"""
    
        if self.state.state.get("segmentation_in_progress", False):
            st.warning("Segmentation is already running. Please wait...")
            return

        try:
            import torch
            import gc
    
            st.cache_data.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            pass

        self.state.state.segmentation_in_progress = True
        progress_container = st.empty()
        status_container = st.empty()
        try:
            model_config = self.state.state.model_config
            if not model_config:
                error_msg = f"No configuration found for model: {self.state.state.modelType}"
                st.error(error_msg)
                return

            info_line_height = self.state.state.infoLineHeight or 0

            postprocessed_masks, raw_masks, probs = segmentationImage(
                uploaded_file=self.state.state.uploadedImage,
                INFLINEPX=info_line_height,
                width=self.state.state.imgWidth,
                height=self.state.state.imgHeight,
                imgName=self.state.state.imageName,
                model_config=model_config,
                threshold=0.5
            )
        
            # Сохраняем оба варианта
            self.state.state.postprocessed_masks = postprocessed_masks
            self.state.state.raw_masks = raw_masks
            self.state.state.probs = probs
        
            # Устанавливаем текущие маски в зависимости от настройки
            apply_postsegmentation = self.state.state.get("apply_postsegmentation", True)
            if apply_postsegmentation:
                self.state.state.predictedObjects = postprocessed_masks
            else:
                self.state.state.predictedObjects = raw_masks

            # Подсчет объектов
            total_objects = 0
            for class_name, mask in self.state.state.predictedObjects.items():
                if class_name not in ["bg", "background"]:
                    num_objects = len(np.unique(mask)) - 1
                    total_objects += num_objects

        
            # Подготовка информации об объектах
            self.state.state.objectsInfo, area_stats = prepareObjectInfo(
                self.state.state.predictedObjects, 
                model_config
            )
        
            self.state.state.filteredObjects = None
            self.state.update_area_stats(self.state.state.objectsInfo, area_stats)
            self.state.apply_filtration()
        
            status_container.success("✅ Segmentation completed!")
            st.toast(f"✅ Segmentation complete! Found {total_objects} objects", icon="🎉")
        
            progress_container.empty()
            self.state.state.segmentation_in_progress = False
            st.rerun()

        except Exception as e:
            progress_container.empty()
            status_container.empty()
            self.state.state.segmentation_in_progress = False
            st.error(f"❌ Segmentation failed: {str(e)}")
            st.rerun()
    
    def handle_statistics_update(self):
        """Обработка обновления статистики"""
        if self.state.state.predictedObjects is not None:
            if self.state.state.objectsInfo is None:
                objects_info, area_stats = prepareObjectInfo(
                    self.state.state.predictedObjects, 
                    self.state.state.model_config 
                )
                self.state.update_area_stats(objects_info, area_stats)
    
        self.state.apply_filtration()
            
    def handle_filtration(self, filter_params=None):
        """Обработка фильтрации"""      
        if self.state.state.predictedObjects is not None:
            if filter_params and self.state.update_filtration_params(filter_params):
                self.state.apply_filtration()
            
    def prepare_export_data(self):
        """Подготовка данных для экспорта"""
        if self.state.state.filteredObjects is not None:
            from segmentation.objects import prepareFilteredObjectInfo
            from segmentation.statistics import calculateStatistics
            from utils.drawing import drawPicture

            filtered_info = self.state.state.filteredObjectsInfo
            if filtered_info is None:
                filtered_info = prepareFilteredObjectInfo(self.state.state.filteredObjects)
                self.state.state.filteredObjectsInfo = filtered_info
            result_info = calculateStatistics(filtered_info, scale=self.state.state.imgScale)
            processed_image = drawPicture(
                self.state.state.uploadedImage,
                self.state.state.filteredObjects,
                self.state.state.classColors,
                self.state.state.isShowIntermediate,
            )
            self.state.state.polygonsCVAT = makeCVATbackup(
                self.state.state.uploadedImage,
                self.state.state.imageName,
                self.state.state.filteredObjects,
                self.state.state.imgWidth,
                self.state.state.imgHeight,
                self.state.state.model_config 
            )
            self.state.state.zipBuffer = saveResultsAsZip(
                filteredObjects=self.state.state.filteredObjects,
                classColors=self.state.state.classColors,
                drawImgPIL=processed_image,
                filteredObjectsInfo=self.state.state.filteredObjectsInfo,
                scale=self.state.state.imgScale,
                imgWidth=self.state.state.imgWidth,
                imgHeight=self.state.state.imgHeight,
                infoLineHeight=self.state.state.infoLineHeight,
                filtration_params=self.state.state.filtration_params,
                resultInfo=result_info,
                imgName=self.state.state.imageName,
                model_config=self.state.state.model_config
            )
            self.state.state.export_dirty = False
