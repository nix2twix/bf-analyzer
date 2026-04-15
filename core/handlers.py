# === LIBRARIES GENERAL ===
import streamlit as st
import numpy as np

# === PROJECT SCRIPTS ===
from core.stateManager import StateManager
from processing import (
    segmentationImage,
    getPredictedObjects,
    prepareObjectInfo,
    filtrationObjects
)
from processing.objects import groupObjectsByClass 
from src.drawing import checkSize, correctSize
from src.autoscale import estimateScale
from src.converter import loadMasksFromZip, makeCVATbackupRLE, saveResultsAsZip

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
    
    def _handle_image_padding(self):
        """Обработка паддинга изображения"""
        wPad, hPad = checkSize(self.state.state.uploadedImage, min_size=512)
        if (wPad != self.state.state.imgWidth) or (hPad != self.state.state.imgHeight):
            paddedImg = correctSize(self.state.state.uploadedImage, wPad, hPad)
            st.warning(f"Image will be padded from {self.state.state.imgWidth}x{self.state.state.imgHeight} "
                      f"to {wPad}x{hPad} for segmentation (multiples of 32).")
            self.state.state.uploadedImage = paddedImg
            self.state.state.imgWidth, self.state.state.imgHeight = paddedImg.size
    
    def handle_scale_detection(self):
        """Обработка определения масштаба"""
        print(f"[DEBUG] handle_scale_detection called. scaleInfo={self.state.state.scaleInfo}")
    
        if self.state.state.uploadedImage is not None and self.state.state.scaleInfo is None:
            print("[DEBUG] No scaleInfo, attempting auto-detection...")
            tempArrImg = np.array(self.state.state.uploadedImage, dtype='uint8')
            autoScale, scaleData = estimateScale(tempArrImg)
            self.state.update_scale(scaleData, autoScale)
            print(f"[DEBUG] Auto scale detected: {autoScale}")
        else:
            print(f"[DEBUG] Skipping scale detection: scaleInfo={self.state.state.scaleInfo}")
    
    def handle_annotation_apply(self, uploaded_ann):
        """Применение загруженной аннотации"""
        if uploaded_ann is None:
            return False
    
        print(f"[INFO] Applying annotation: {uploaded_ann.name}")
    
        # Сбрасываем результаты сегментации если они были
        if self.state.state.predictedObjects is not None:
            print("[INFO] Clearing previous results")
    
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
    
        print("[INFO] Statistics cache cleared")
        return True

    def handle_segmentation(self):
        """Обработка сегментации"""
        print(f"[INFO] Starting segmentation with model: {self.state.state.modelType}")

        self.state.state.segmentation_in_progress = True

        with st.spinner("⏳ Image processing..."):
            model_config = self.state.state.model_config

            if not model_config:
                st.error(f"No configuration found for model: {self.state.state.modelType}")
                self.state.state.segmentation_in_progress = False
                return

            try:
                # Сегментация - возвращает уже нумерованные маски
                processedMask, probs = segmentationImage(
                    uploaded_file=self.state.state.uploadedImage,
                    INFLINEPX=self.state.state.infoLineHeight,
                    width=self.state.state.imgWidth,
                    height=self.state.state.imgHeight,
                    imgName=self.state.state.imageName,
                    model_config=model_config,
                    threshold=0.5
                )
        
                # processedMask уже содержит нумерованные маски объектов
                self.state.state.predictedObjects = processedMask
        
                # Получаем информацию об объектах и статистику
                self.state.state.objectsInfo, area_stats = prepareObjectInfo(
                    processedMask, 
                    model_config
                )
        
                # Инициализируем filteredObjects
                self.state.state.filteredObjects = None
        
                # Обновляем статистику и параметры из area_stats
                self.state.update_area_stats(self.state.state.objectsInfo, area_stats)
        
                # Применяем фильтрацию
                self.state.apply_filtration()

                st.success("✅ Segmentation completed!")
                self.state.state.segmentation_in_progress = False
                st.rerun()
    
            except Exception as e:
                st.error(f"Segmentation failed: {str(e)}")
                print(f"[ERROR] Segmentation error: {e}")
                import traceback
                traceback.print_exc()
                self.state.state.segmentation_in_progress = False
    
    def handle_statistics_update(self):
        """Обработка обновления статистики"""
        if self.state.state.predictedObjects is not None:
            # Пересчитываем objectsInfo если нужно
            if self.state.state.objectsInfo is None:
                print("[INFO] Calculating new statistics from predicted objects")
                objects_info, area_stats = prepareObjectInfo(
                    self.state.state.predictedObjects, 
                    self.state.state.model_config 
                )
                self.state.update_area_stats(objects_info, area_stats)
    
            # Применяем фильтры если они есть и filteredObjects ещё не созданы
            if self.state.state.filtration_params and self.state.state.filteredObjects is None:
                print("[INFO] Applying filters to objects")
                self.state.state.filteredObjects = filtrationObjects(
                    self.state.state.objectsInfo,
                    self.state.state.predictedObjects,
                    self.state.state.filtration_params,
                    self.state.state.model_config
                )
            elif not self.state.state.filtration_params and self.state.state.filteredObjects is None:
                # Если фильтров нет, filteredObjects = predictedObjects
                self.state.state.filteredObjects = self.state.state.predictedObjects
            
            # Сбрасываем filteredObjectsInfo для пересчета
            self.state.state.filteredObjectsInfo = None
    
    def handle_filtration(self, filter_params=None):
        """Обработка фильтрации"""
        if self.state.state.predictedObjects is not None:
            if filter_params:
                self.state.update_filtration_params_from_ui(filter_params)
            self.state.apply_filtration()
    
    def prepare_export_data(self, processed_image, result_info):
        """Подготовка данных для экспорта"""
        if self.state.state.filteredObjects is not None:
            self.state.state.polygonsCVAT = makeCVATbackupRLE(
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