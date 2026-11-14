# === LIBRARIES GENERAL ===
import streamlit as st
import numpy as np
import zipfile, io, os

# === PROJECT SCRIPTS ===
from core.stateManager import StateManager 
from core.batchProcessor import BatchProcessor
from segmentation.inference import segmentationImage
from segmentation.objects import (
    prepare_objectInfo
)
from segmentation.statistics import calculateStatistics
from utils.drawing import checkSize, correctSize
from utils.autoscale import estimateScale
from utils.importer import loadMasksFromZip, checkAnnotationSize
from utils.exporter import (
    prepare_CVATbackup,
    prepare_CVATbackup_batch,
    saveResultsAsZip,
    saveBatchSummaryAsZip
)

class AppHandlers:
    """Обработчики событий приложения"""
    def __init__(self, state_manager: StateManager):
        self.state = state_manager
    
    def handle_fileUpload(self, uploaded_file):
        if uploaded_file is None:
            self.state.reset_all()
            return

        # === BATCH: ZIP ===
        if _is_zip(uploaded_file):
            self.handle_batchUpload(uploaded_file)
            return

        # === SINGLE: обычная картинка ===
        if self.state.state.get("batch_mode", False):
            self.state.setBatchMode(False)

        if uploaded_file.name != self.state.state.imageName:
            self.state.setImage(uploaded_file)
            self._handle_image_padding()
            self.handle_scaleDetection()

    def handle_batchUpload(self, uploaded_zip):
        """
        Шаг 2: только распаковка и валидация архива.
        Обработка появится на Шаге 3 (BatchProcessor).
        """
        self.state.setBatchMode(True)

        images = self._extract_images_from_zip(uploaded_zip)
        self.state.state.batch_images_raw = images

        if not images:
            st.error("❌ ZIP archive contains no supported images.")
            self.state.state.batch_mode = False
            return

    def _extract_images_from_zip(self, uploaded_zip):
        """Возвращает список (unique_name, bytes), дедуплицируя имена."""
        supported = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        seen = {}
        result = []

        try:
            with zipfile.ZipFile(uploaded_zip) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    name = info.filename
                    _, ext = os.path.splitext(name)
                    if ext.lower() not in supported:
                        continue
                    data = zf.read(info)
                    if not data:
                        continue

                    # дедупликация: img.bmp -> img_1.bmp, img_2.bmp
                    base = os.path.basename(name)
                    if base in seen:
                        seen[base] += 1
                        stem, ext2 = os.path.splitext(base)
                        unique = f"{stem}_{seen[base]}{ext2}"
                    else:
                        seen[base] = 0
                        unique = base

                    result.append((unique, data))
        except zipfile.BadZipFile:
            st.error("❌ File is not a valid ZIP archive.")
            return []

        return result         

    def _handle_image_padding(self):
        """Обработка паддинга изображения"""
        wPad, hPad = checkSize(self.state.state.uploadedImage, min_size=512)
        if (wPad != self.state.state.imgWidth) or (hPad != self.state.state.imgHeight):
            paddedImg = correctSize(self.state.state.uploadedImage, wPad, hPad)
            st.warning(f"Image will be padded from {self.state.state.imgWidth}x{self.state.state.imgHeight} "
                      f"to {wPad}x{hPad} for segmentation (multiples of 32).")
            self.state.state.uploadedImage = paddedImg
            self.state.state.imgWidth, self.state.state.imgHeight = paddedImg.size
    
    def handle_scaleDetection(self, force=False):
        """Обработка определения масштаба"""
        if not force and self.state.state.get("manual_scale_set", False):
           return False
    
        if self.state.state.uploadedImage is not None:
            tempArrImg = np.array(self.state.state.uploadedImage, dtype='uint8')
            autoScale, scaleData = estimateScale(tempArrImg)
            
            if autoScale is not None:
                self.state.updateScale(scaleData, autoScale)
                self.state.state.manual_scale_set = False
                
                print(f"[DEBUG] Auto scale detected: {autoScale}")
                return True
            else:
                self.state.updateScale(None, None)
                return False
        
        return False
    
    def handle_imgAreaScaled(self):
        self.state.state.imgAreaScaled = (self.state.state.imgWidth * 
                                    (self.state.state.imgHeight - self.state.state.infoLineHeight) *
                                    (self.state.state.imgScale ** 2))

    def handle_annotationUpload(self, uploaded_ann):
        isMatch, _, _ = checkAnnotationSize(uploaded_ann, 
                                                    self.state.state.imgWidth, 
                                                    self.state.state.imgHeight,
                                                    minimumDiff = 128)
        return isMatch
    
    def handle_annotationApply(self, uploaded_ann):
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

            image_array = np.asarray(
                self.state.state.uploadedImage,
                dtype=np.uint8
            )

            image_bytes = image_array.tobytes()
            postprocessed_masks, raw_masks, probs = segmentationImage(
                image_bytes=image_bytes,
                INFLINEPX=info_line_height,
                width=self.state.state.imgWidth,
                height=self.state.state.imgHeight,
                imgName=self.state.state.imageName,
                model_config=model_config,
                threshold=0.1
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
            self.state.state.objectsInfo, area_stats = prepare_objectInfo(
                self.state.state.predictedObjects, 
                model_config
            )
        
            self.state.state.filteredObjects = None
            self.state.updateAreaStats(self.state.state.objectsInfo, area_stats)
            self.state.applyFiltration()
        
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

    def handle_batchSegmentation(self):
        images = self.state.state.get("batch_images_raw", [])
        if not images:
            st.warning("No images to process.")
            return

        if self.state.state.get("batch_in_progress", False):
            st.warning("Batch processing is already running. Please wait...")
            return

        model_config = self.state.state.model_config
        if not model_config:
            st.error(f"No configuration found for model: {self.state.state.modelType}")
            return

        self.state.state.batch_in_progress = True
        overall_bar = st.progress(0.0, text="Preparing...")
        patch_bar = st.progress(0.0, text="")
        status = st.empty()

        try:
            processor = BatchProcessor(self.state)

            def _progress(i, total, name):
                overall_bar.progress(
                    i / total,
                    text=f"Processing image {i}/{total}: {name}",
                )
                # сбрасываем внутренний бар на новую картинку
                patch_bar.progress(0.0, text=f"Patches: 0/? ({name})")

            def _patch_cb_factory(name):
                def _cb(done, total_patches):
                    patch_bar.progress(
                        done / max(total_patches, 1),
                        text=f"Patches {done}/{total_patches} ({name})",
                    )
                return _cb

            items = processor.process_all(
                images,
                progress_cb=_progress,
                patch_progress_cb_factory=_patch_cb_factory,
            )
            items.sort(key=lambda it: it.unique_name.lower())
            self.state.state.batch_items = items
            self.state.state.batch_errors = [
                (it.unique_name, it.error) for it in items if it.error is not None
            ]

            ok_count = sum(1 for it in items if it.ok)
            overall_bar.progress(1.0, text=f"Done: {ok_count}/{len(items)} processed")
            patch_bar.progress(1.0, text="")
            status.success(f"✅ Batch processed: {ok_count}/{len(items)} image(s) ok.")
            st.toast(f"✅ Batch done: {ok_count}/{len(items)}", icon="🎉")

        except Exception as e:
            st.error(f"❌ Batch processing failed: {e}")
        finally:
            self.state.state.batch_in_progress = False
            st.rerun()

    def handle_statisticsAppear(self):
       """Обработка первичного расчета статистики"""
       return calculateStatistics(
                self.state.state.filteredObjectsInfo,
                scale=self.state.state.imgScale
            )
    
    def handle_statisticsUpdate(self):
        """Обработка обновления статистики"""
        if self.state.state.predictedObjects is not None:
            if self.state.state.objectsInfo is None:
                objects_info, area_stats = prepare_objectInfo(
                    self.state.state.predictedObjects, 
                    self.state.state.model_config 
                )
                self.state.updateAreaStats(objects_info, area_stats)
    
        self.state.applyFiltration()
            
    def handle_filtration(self, filter_params=None):
        """Обработка фильтрации"""
        if self.state.state.predictedObjects is not None:
            if filter_params:
                self.state.update_filtration_params(filter_params)

            self.state.applyFiltration()
            
    def prepare_exportData(self):
        """Подготовка данных для экспорта"""
        if self.state.state.filteredObjects is not None:
            from segmentation.objects import prepare_filteredObjectInfo
            from segmentation.statistics import calculateStatistics
            from utils.drawing import drawPicture

            filtered_info = self.state.state.filteredObjectsInfo
            if filtered_info is None:
                filtered_info = prepare_filteredObjectInfo(self.state.state.filteredObjects)
                self.state.state.filteredObjectsInfo = filtered_info
            result_info = calculateStatistics(filtered_info, scale=self.state.state.imgScale)
            processed_image = drawPicture(
                self.state.state.uploadedImage,
                self.state.state.filteredObjects,
                self.state.state.classColors,
                self.state.state.isShowIntermediate,
            )
            self.state.state.polygonsCVAT = prepare_CVATbackup(
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
            

    def prepare_batchExportData(self):
        """
        Готовит batch-архивы: CVAT backup + сводный Excel.
        """
        items = self.state.state.get("batch_items", []) or []
        if not items:
            st.warning("No batch results to export.")
            return

        ok_items = [it for it in items if it.ok]
        if not ok_items:
            st.error("All items failed — nothing to export.")
            return

        model_config = self.state.state.model_config
        if not model_config:
            st.error("Model config is not set.")
            return

        # --- CVAT ---
        try:
            self.state.state.polygonsCVAT = prepare_CVATbackup_batch(
                items=ok_items,
                model_config=model_config,
            )
            self.state.state.batch_zip_cvat = self.state.state.polygonsCVAT
        except Exception as e:
            st.error(f"❌ CVAT backup failed: {e}")
            return

        # --- Excel (сводный) ---
        try:
            # передаём все items, чтобы ошибочные попали на лист Errors
            self.state.state.zipBuffer = saveBatchSummaryAsZip(
                items=items,
                model_config=model_config,
            )
            self.state.state.batch_zip_excel = self.state.state.zipBuffer
        except Exception as e:
            st.error(f"❌ Summary Excel failed: {e}")
            # CVAT уже собрали, но раз Excel упал — оставим export_dirty=True,
            # чтобы пользователь мог повторить
            self.state.state.export_dirty = False
            st.toast("⚠️ CVAT backup ready, but summary Excel failed", icon="⚠️")
            return

        # --- Освобождение памяти ---
        # плотные маски уже не нужны ни CVAT (запакован), ни Excel (готов)
        for it in ok_items:
            it.filtered_masks = {}
            it.exported = True

        self.state.state.export_dirty = False
        st.toast("✅ Batch archives ready (CVAT + Excel)", icon="📦")

def _is_zip(uploaded_file) -> bool:
    """Определяет, является ли загруженный файл ZIP-архивом."""
    name = (getattr(uploaded_file, "name", "") or "").lower()
    if name.endswith(".zip"):
        return True
    try:
        head = uploaded_file.getbuffer()[:4]
        return head == b"PK\x03\x04"
    except Exception:
        return False
