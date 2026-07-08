# === LIBRARIES GENERAL ===
import streamlit as st
import numpy as np
import time

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
from logger import logger


class AppHandlers:
    """Обработчики событий приложения"""
    
    def __init__(self, state_manager: StateManager):
        self.state = state_manager
    
    def handle_file_upload(self, uploaded_file):
        """Обработка загрузки файла"""
        if uploaded_file is None:
            logger.log("FILE_UPLOAD", "No file provided, resetting state", "WARNING")
            self.state.reset_all()
            return
        
        if uploaded_file.name != self.state.state.imageName:
            # Логируем загрузку
            file_size_mb = uploaded_file.size / (1024 * 1024)
            logger.log_image_upload(
                uploaded_file.name,
                self.state.state.imgWidth if self.state.state.imgWidth else 0,
                self.state.state.imgHeight if self.state.state.imgHeight else 0,
                file_size_mb
            )
            
            self.state.set_image(uploaded_file)
            self._handle_image_padding()
            # Определяем масштаб сразу после загрузки
            self.handle_scale_detection()
        else:
            logger.log("FILE_UPLOAD", f"File '{uploaded_file.name}' already loaded, skipping", "DEBUG")
    
    def _handle_image_padding(self):
        """Обработка паддинга изображения"""
        wPad, hPad = checkSize(self.state.state.uploadedImage, min_size=512)
        if (wPad != self.state.state.imgWidth) or (hPad != self.state.state.imgHeight):
            paddedImg = correctSize(self.state.state.uploadedImage, wPad, hPad)
            logger.log(
                "IMAGE_PADDING",
                f"from {self.state.state.imgWidth}x{self.state.state.imgHeight} to {wPad}x{hPad}",
                "INFO"
            )
            st.warning(f"Image will be padded from {self.state.state.imgWidth}x{self.state.state.imgHeight} "
                      f"to {wPad}x{hPad} for segmentation (multiples of 32).")
            self.state.state.uploadedImage = paddedImg
            self.state.state.imgWidth, self.state.state.imgHeight = paddedImg.size
    
    def handle_scale_detection(self, force=False):
        """Обработка определения масштаба"""
        filename = self.state.state.imageName
        
        # Пропускаем, если масштаб установлен вручную и не force
        if not force and self.state.state.get("manual_scale_set", False):
            logger.log("SCALE_DETECTION", f"image='{filename}', skipped (manual scale set)", "DEBUG")
            return False
    
        if self.state.state.uploadedImage is not None:
            logger.log("SCALE_DETECTION", f"image='{filename}', attempting auto-detection...", "DEBUG")
            
            tempArrImg = np.array(self.state.state.uploadedImage, dtype='uint8')
            autoScale, scaleData = estimateScale(tempArrImg)
            
            if autoScale is not None:
                self.state.update_scale(scaleData, autoScale)
                self.state.state.manual_scale_set = False
                
                # Логируем успешное определение масштаба
                detected_text = scaleData[3] if scaleData and len(scaleData) > 3 else "Unknown"
                logger.log_scale_detection(filename, autoScale, detected_text)
                
                print(f"[DEBUG] Auto scale detected: {autoScale}")
                return True
            else:
                logger.log("SCALE_DETECTION", f"image='{filename}', auto-detection failed", "WARNING")
                self.state.update_scale(None, None)
                return False
        
        logger.log("SCALE_DETECTION", f"image='{filename}', no image uploaded", "WARNING")
        return False
    
    def handle_annotation_apply(self, uploaded_ann):
        """Применение загруженной аннотации"""
        if uploaded_ann is None:
            logger.log("ANNOTATION_APPLY", "No annotation file provided", "WARNING")
            return False

        filename = self.state.state.imageName
        logger.log("ANNOTATION_APPLY", f"image='{filename}', annotation='{uploaded_ann.name}'", "INFO")

        # Сбрасываем результаты сегментации если они были
        if self.state.state.predictedObjects is not None:
            logger.log("ANNOTATION_APPLY", f"image='{filename}', clearing previous results", "DEBUG")

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

        logger.log("ANNOTATION_APPLY", f"image='{filename}', statistics cache cleared", "INFO")
        return True

    def handle_segmentation(self):
        """Обработка сегментации - сохраняет оба варианта масок"""
    
        filename = self.state.state.imageName
        model = self.state.state.modelType
        scale_info = self.state.state.imgScale
    
        logger.log_segmentation_start(filename, model, scale_info)
    
        if self.state.state.get("segmentation_in_progress", False):
            logger.log("SEGMENTATION", f"image='{filename}', already in progress, skipping", "WARNING")
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
            logger.log_error("memory_cleanup", e, filename)

        self.state.state.segmentation_in_progress = True
        progress_container = st.empty()
        status_container = st.empty()
        start_time = time.time()

        try:
            model_config = self.state.state.model_config
            if not model_config:
                error_msg = f"No configuration found for model: {self.state.state.modelType}"
                logger.log("SEGMENTATION_ERROR", f"image='{filename}', {error_msg}", "ERROR")
                st.error(error_msg)
                return

            info_line_height = self.state.state.infoLineHeight or 0

            # ВЫЗЫВАЕМ НОВУЮ ВЕРСИЮ - получаем оба варианта
            postprocessed_masks, raw_masks, probs = segmentationImage(
                uploaded_file=self.state.state.uploadedImage,
                INFLINEPX=info_line_height,
                width=self.state.state.imgWidth,
                height=self.state.state.imgHeight,
                imgName=self.state.state.imageName,
                model_config=model_config,
                threshold=0.5
            )
        
            elapsed_time = time.time() - start_time
            logger.log("SEGMENTATION", f"image='{filename}', segmentation completed in {elapsed_time:.2f} seconds", "INFO")
        
            # Сохраняем оба варианта
            self.state.state.postprocessed_masks = postprocessed_masks
            self.state.state.raw_masks = raw_masks
            self.state.state.probs = probs
        
            # Устанавливаем текущие маски в зависимости от настройки
            apply_postprocessing = self.state.state.get("apply_postprocessing", True)
            if apply_postprocessing:
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
        
            logger.log_segmentation_complete(filename, total_objects, elapsed_time, model)
            status_container.success("✅ Segmentation completed!")
            st.toast(f"✅ Segmentation complete! Found {total_objects} objects", icon="🎉")
        
            progress_container.empty()
            self.state.state.segmentation_in_progress = False
            st.rerun()

        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.log_segmentation_error(filename, str(e), model)
            progress_container.empty()
            status_container.empty()
            self.state.state.segmentation_in_progress = False
            st.error(f"❌ Segmentation failed: {str(e)}")
            st.rerun()
    
    def handle_statistics_update(self):
        """Обработка обновления статистики"""
        filename = self.state.state.imageName
        
        if self.state.state.predictedObjects is not None:
            # Пересчитываем objectsInfo если нужно
            if self.state.state.objectsInfo is None:
                logger.log("STATISTICS_UPDATE", f"image='{filename}', calculating new statistics from predicted objects", "DEBUG")
                objects_info, area_stats = prepareObjectInfo(
                    self.state.state.predictedObjects, 
                    self.state.state.model_config 
                )
                self.state.update_area_stats(objects_info, area_stats)
    
            # Применяем фильтры если они есть и filteredObjects ещё не созданы
            if self.state.state.filtration_params and self.state.state.filteredObjects is None:
                logger.log("STATISTICS_UPDATE", f"image='{filename}', applying filters to objects", "DEBUG")
                self.state.state.filteredObjects = filtrationObjects(
                    self.state.state.objectsInfo,
                    self.state.state.predictedObjects,
                    self.state.state.filtration_params,
                    self.state.state.model_config
                )
            elif not self.state.state.filtration_params and self.state.state.filteredObjects is None:
                # Если фильтров нет, filteredObjects = predictedObjects
                logger.log("STATISTICS_UPDATE", f"image='{filename}', no filters, using predicted objects as filtered", "DEBUG")
                self.state.state.filteredObjects = self.state.state.predictedObjects
            
            # Сбрасываем filteredObjectsInfo для пересчета
            self.state.state.filteredObjectsInfo = None
            logger.log("STATISTICS_UPDATE", f"image='{filename}', filteredObjectsInfo reset", "DEBUG")
        else:
            logger.log("STATISTICS_UPDATE", f"image='{filename}', no predicted objects, skipping", "DEBUG")
    
    def handle_filtration(self, filter_params=None):
        """Обработка фильтрации"""
        filename = self.state.state.imageName
        
        if self.state.state.predictedObjects is not None:
            if filter_params:
                logger.log("FILTRATION", f"image='{filename}', updating filtration params from UI", "DEBUG")
                self.state.update_filtration_params_from_ui(filter_params)
            self.state.apply_filtration()
            logger.log("FILTRATION", f"image='{filename}', filtration applied", "INFO")
        else:
            logger.log("FILTRATION", f"image='{filename}', no predicted objects, skipping", "WARNING")
    
    def prepare_export_data(self, processed_image, result_info):
        """Подготовка данных для экспорта"""
        filename = self.state.state.imageName
        
        if self.state.state.filteredObjects is not None:
            logger.log("EXPORT_PREPARE", f"image='{filename}', preparing CVAT backup...", "DEBUG")
            
            self.state.state.polygonsCVAT = makeCVATbackupRLE(
                self.state.state.uploadedImage,
                self.state.state.imageName,
                self.state.state.filteredObjects,
                self.state.state.imgWidth,
                self.state.state.imgHeight,
                self.state.state.model_config 
            )
            
            # Логируем размер CVAT backup
            if self.state.state.polygonsCVAT:
                cvat_size = len(self.state.state.polygonsCVAT.getvalue()) / 1024
                logger.log_export(filename, "CVAT_backup", cvat_size)
            
            logger.log("EXPORT_PREPARE", f"image='{filename}', preparing Results zip...", "DEBUG")
            
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
            
            # Логируем размер Results
            if self.state.state.zipBuffer:
                results_size = len(self.state.state.zipBuffer.getvalue()) / 1024
                logger.log_export(filename, "Results", results_size)
                
            logger.log("EXPORT_PREPARE", f"image='{filename}', export data ready", "INFO")
        else:
            logger.log("EXPORT_PREPARE", f"image='{filename}', no filtered objects, skipping", "WARNING")