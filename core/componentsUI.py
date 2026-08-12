import numpy as np
import streamlit as st
from PIL import Image
from datetime import datetime
from typing import Dict, Any, Optional
from core.factoryUI import ModelUIFactory

class UIComponents:
    """
    Собирает готовые секции интерфейса, используя фабрику для специфичных частей.
    Отвечает за компоновку и взаимодействие с состоянием.
    """
    
    def __init__(self, state_manager):
        self.state = state_manager
        self.factory = ModelUIFactory()
        # Кэш для изображений в session_state
        if "image_cache" not in st.session_state:
            st.session_state.image_cache = {}
    
    # === ПУБЛИЧНЫЕ МЕТОДЫ ===
    def render_workflow_area(self) -> Optional[Any]:
        """Отрисовывает рабочую область с кэшированием"""
        if self.state.state.uploadedImage is None:
            st.info("The SEM image wasn't uploaded. Please upload the image using the menu on the right.")
            return None

        if self.state.state.filteredObjects is not None:
            from utils.drawing import drawPicture, resizeForDisplay
        
            has_objects = False
            for class_name, mask in self.state.state.filteredObjects.items():
                if class_name not in ["bg", "background"] and np.any(mask):
                    has_objects = True
                    break
        
            if has_objects:
                # The state manager clears this cache whenever masks change.
                # Do not hash every mask pixel on each Streamlit rerun.
                cache_key = f"segmented_{self.state.state.isShowIntermediate}"
            
                if cache_key not in st.session_state.image_cache:
                    display_image = drawPicture(
                        self.state.state.uploadedImage,
                        self.state.state.filteredObjects,
                        self.state.state.classColors,
                        self.state.state.isShowIntermediate
                    )
                    # Уменьшаем только если очень большое (для ускорения загрузки)
                    if display_image.width > 1200:
                        display_image = resizeForDisplay(display_image, max_width=1200)
                    st.session_state.image_cache[cache_key] = display_image
                else:
                    display_image = st.session_state.image_cache[cache_key]
            
                # Адаптивное отображение
                #st.image(display_image, width='stretch')
                return display_image
            else:
                st.warning("⚠️ No objects found after filtering")
                # st.image(
                #     self.state.state.uploadedImage,
                #     width='stretch'
                # )
                return Image.open(self.state.state.uploadedImage)
        else:
            cache_key = f"orig_{self.state.state.imageName}"
            if cache_key not in st.session_state.image_cache:
                display_image = self.state.state.uploadedImage
                if display_image.width > 1200:
                    from utils.drawing import resizeForDisplay
                    display_image = resizeForDisplay(display_image, max_width=1200)
                st.session_state.image_cache[cache_key] = display_image
            else:
                display_image = st.session_state.image_cache[cache_key]
            
            # st.image(
            #     display_image,
            #     width='stretch'
            # )
            return display_image
    
    def render_workflow_area_mini(self) -> Optional[Any]:
        """Мини-версия для контекста экспорта (без отрисовки, только возврат изображения)"""
        if self.state.state.uploadedImage is None:
            return None

        if self.state.state.filteredObjects is not None:
            from utils.drawing import drawPicture
            
            has_objects = False
            for class_name, mask in self.state.state.filteredObjects.items():
                if class_name not in ["bg", "background"] and np.sum(mask > 0) > 0:
                    has_objects = True
                    break
            
            if has_objects:
                display_image = drawPicture(
                    self.state.state.uploadedImage,
                    self.state.state.filteredObjects,
                    self.state.state.classColors,
                    self.state.state.isShowIntermediate
                )
                return display_image
            else:
                return self.state.state.uploadedImage
        else:
            return self.state.state.uploadedImage
    
    def clear_image_cache(self):
        """Очистка кэша изображений"""
        st.session_state.image_cache = {}
    
    def render_file_uploader_only(self):
        """Отрисовывает только загрузчик файла"""
        return st.file_uploader(
            label="Upload SEM image",
            label_visibility="collapsed",
            #help="⬇️ Upload SEM-image",
            type=["bmp", "png", "jpg", "jpeg"],
            key="uploader"
        )
    
    def render_model_selector_only(self) -> str:
        """Отрисовывает только селектор модели"""
        model_options = {
            "Bacillus": "Bacillus",
            "Coccus": "Coccus"
        }
        
        selected_display = st.segmented_control(
            label="Type of microorganisms:",
            default=list(model_options.values())[0],
            options=list(model_options.values()),
            selection_mode="single",
            required=True,
            width="stretch",
            key="modelChooser"
            )
        
        for key, value in model_options.items():
            if value == selected_display:
                return key
        
        return "Bacillus"
    
    def render_segmentation_button_only(self) -> bool:
        """Отрисовывает кнопку сегментации"""
        if self.state.state.get("segmentation_in_progress", False):
            st.info("⏳ Segmentation in progress...")
            return False
        else:
            return st.button(
                "🧪 Start segmentation",
                disabled=self.state.state.uploadedImage is None,
                width='stretch'
            )
    
    def render_annotation_uploader_only(self) -> Optional[Any]:
        """Отрисовывает только загрузчик аннотаций"""
        return st.file_uploader(
            "⬇ Upload mask from CVAT (ZIP)",
            type=['zip'],
            key="annotation_uploader_stats",
            disabled=False,
            help="Upload annotations exported from CVAT to recalculate statistics"
        )
    
    def render_filtration_ui(self, container) -> Dict:
        """Отрисовывает UI фильтрации в указанном контейнере"""
        config = self.state.get_config()
    
        if config and self.state.state.predictedObjects is not None:
            with container:
                       
                params = self.factory.create_filtration_ui(config, self.state.state)
                if params:
                    return params
        return {}
    
    def render_statistics_content(self, container, result_info: Dict, img_area: float):
        """Отрисовывает контент статистики"""
        config = self.state.get_config()
        
        if self.state.state.filteredObjects is not None and config and result_info:
            with container:
                self.factory.create_statistics_ui(config, result_info, img_area)
  
    def render_export_buttons(self, suffix: str = ""):
        """Отрисовывает кнопки экспорта с уникальными ключами"""
        col1, col2 = st.columns(2)
        
        cvat_key = f"cvat_export_{suffix}" if suffix else "cvat_export_main"
        results_key = f"results_export_{suffix}" if suffix else "results_export_main"
        
        if self.state.state.export_dirty:
            if st.button(
                    "⬇️ Save changes",
                    key=f"prepare_results_{suffix}",
                    width="stretch",
                    disabled=self.state.state.filteredObjects is None,
                ):
                    return "prepare"
            return None

        with col1:
            st.download_button(
                label="📥 CVAT backup",
                data=self.state.state.polygonsCVAT or b"",
                file_name=f"backup-{datetime.now().strftime('%d-%m-%Y-%H-%M')}.zip",
                mime="application/zip",
                disabled=self.state.state.uploadedImage is None or self.state.state.polygonsCVAT is None,
                width='stretch',
                help="Download masks in CVAT format",
                key=cvat_key
            )
        with col2:
            st.download_button(
                label="💾 Results",
                    data=self.state.state.zipBuffer or b"",
                file_name=f"results-{datetime.now().strftime('%Y-%m-%d-%H-%M')}.zip",
                mime="application/zip",
                disabled=self.state.state.uploadedImage is None or self.state.state.zipBuffer is None,
                width='stretch',
                help="Download segmentation results",
                key=results_key
            )
    def render_postprocess_toggle(self):
        current_value = self.state.state.get("apply_postsegmentation", False)
            
        postsegmentation_toggle = st.toggle(
            "Morphological postprocess",
            value=current_value,
            help="When enabled applies smoothing, hole filling and class filtering.\n"
                    "When disabled uses raw connected components",
            key="postsegmentation_toggle"
        )
            
        if postsegmentation_toggle != current_value:
            self.state.toggle_postsegmentation(postsegmentation_toggle)
            st.rerun()

    def render_scale_bar(self):
        """Метод для отрисовки информации о масштабе"""
        with st.container(horizontal=True):

            img_scale = self.state.state.imgScale
            scale_text = self.state.state.scaleText
            
            if img_scale is None or img_scale == 0:
                display_scale = 1.0
                display_help = "Scale not set. Click ✏️ to set manually or run segmentation to auto-detect."
            else:
                display_scale = img_scale
                display_help = scale_text if scale_text else "Scale set"
        
            if st.button(f"🔎 Scale: {display_scale:.4f} μm/px", 
                         help=display_help,
                         key="scale_edit_btn",
                         width='stretch'):
                st.session_state.show_scale_dialog = True
                st.rerun()

        if st.session_state.get("show_scale_dialog", False):
            self.show_scale_dialog()
    
    def show_scale_dialog(self):
        """Показывает диалог для ручной установки масштаба"""
        @st.dialog("Set the scale manually")
        def dialog():
            # Определяем текущее значение для отображения
            if self.state.state.scaleInfo and len(self.state.state.scaleInfo) > 4:
                current_value = float(self.state.state.scaleInfo[4])
                is_auto = not self.state.state.get("manual_scale_set", False)
            else:
                current_value = 10.0
                is_auto = False
        
            # Показываем статус
            if is_auto:
                st.info("📏 Currently using auto-detected scale")
            else:
                st.info("✏️ Currently using manually set scale")
        
            manualScaleValue = st.number_input(
                "Enter the length of scale line (μm):",
                min_value=0.001,
                max_value=1000.0,
                value=current_value,
                step=0.1,
                format="%.0f",
                key="manual_scale_input"
            )
        
            col1, col2, col3 = st.columns(3)
        
            with col1:
                if st.button("Cancel", key="cancel_scale_btn", width='stretch'):
                    st.session_state.show_scale_dialog = False
                    st.rerun()
        
            with col2:
                if st.button("🔄 Reset to Auto", key="reset_scale_btn", width='stretch', 
                            disabled=not self.state.state.get("manual_scale_set", False)):
                    self.state.state.manual_scale_set = False
                    self.state.state.scaleInfo = None
                    self.state.state.imgScale = None
        
                    # Сразу запускаем автоопределение если есть изображение
                    if self.state.state.uploadedImage is not None:
                        from utils.autoscale import estimateScale
                        tempArrImg = np.array(self.state.state.uploadedImage, dtype='uint8')
                        autoScale, scaleData = estimateScale(tempArrImg)
                        self.state.update_scale(scaleData, autoScale)
        
                    # Пересчитываем статистику если есть результаты
                    if self.state.state.filteredObjects is not None:
                        self.state.state.filteredObjectsInfo = None
        
                    st.session_state.show_scale_dialog = False
                    st.rerun()
        
            with col3:
                if st.button("Submit", key="submit_scale_btn", width='stretch', type="primary"):
                    # Применяем новое значение
                    new_scale = manualScaleValue / 1008
                
                    self.state.state.imgScale = new_scale
                    self.state.state.scaleText = f"{manualScaleValue} (μm) / {1008} (px) = {new_scale:.4f}"
                    self.state.state.scaleInfo = (0, 0, 1008, 0, manualScaleValue, 0, 0)
                    self.state.state.manual_scale_set = True

                    if self.state.state.filteredObjects is not None:
                        self.state.state.filteredObjectsInfo = None
                
                    st.session_state.show_scale_dialog = False
                    st.rerun()
    
        dialog()
