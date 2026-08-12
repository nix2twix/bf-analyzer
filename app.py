# === LIBRARIES GENERAL ===
import streamlit as st
from streamlit_image_overlay import streamlit_image_overlay as overlay 

# === PROJECT SCRIPTS ===
from segmentation.objects import prepareFilteredObjectInfo
from segmentation.statistics import calculateStatistics

from utils.exporter import makeCVATbackup

from styles.styles import loadStyles, loadFooter
       
from core.stateManager import StateManager
from core.handlers import AppHandlers
from core.componentsUI import UIComponents


st.set_page_config(page_title="Biofilm Analyzer", layout="wide", page_icon="🧪")
loadStyles()

state_manager = StateManager()
handlers = AppHandlers(state_manager)
ui = UIComponents(state_manager)

# === HEADER ===
st.markdown("")
st.header("🧪 Biofilm Analyzer", anchor=False)
st.markdown("This tool is designed for processing SEM images of biofilms")
st.markdown('<hr style="margin: 0rem 0;">', unsafe_allow_html=True)

# === INTERFACE ===
blockWorkspace, gap, blockTools = st.columns([2.5, 0.01, 1.0    ])

with blockTools:
    tabsTools = st.tabs(["🔬 Segmentation", "📊 Statistics & Tools"])
    
    # Вкладка 1: Segmentation
    with tabsTools[0]:
        # 1. Загрузчик изображения
        uploaded_file = ui.render_file_uploader_only()
        
        # 2. Выбор модели
        model_selected = ui.render_model_selector_only()
        if model_selected:
            if state_manager.set_model(model_selected):
                ui.clear_image_cache()
    
        # Обработка загрузки файла
        if uploaded_file and uploaded_file != state_manager.state.get("last_uploaded_file"):
            handlers.handle_file_upload(uploaded_file)
            state_manager.state.last_uploaded_file = uploaded_file
            ui.clear_image_cache()
        
        # 3. Запуск сегментации
        seg_clicked = ui.render_segmentation_button_only()
        if seg_clicked:
            handlers.handle_segmentation()
            state_manager.state.last_uploaded_ann = None
            state_manager.state.uploadedAnnName = None
            # Очищаем кэш после сегментации
            ui.clear_image_cache()
        
        # Определение масштаба
        if state_manager.state.uploadedImage is not None and state_manager.state.scaleInfo is None:
            handlers.handle_scale_detection()
        
        # Обновление статистики
        if (state_manager.state.predictedObjects is not None and 
            state_manager.state.objectsInfo is None):
            handlers.handle_statistics_update()
        
        # 4. Кнопки экспорта
        st.markdown("📤 Export results")
        
        if ui.render_export_buttons(suffix="seg") == "prepare":
            handlers.prepare_export_data()
            st.rerun()

    # === Вкладка 2: Statistics & Import ===
    with tabsTools[1]:
        filtrationCol, gap, statCol = st.columns([1.5, 0.01, 1.6])
        with filtrationCol:
            st.markdown("⚙️ Filters")

        with statCol:
            st.markdown("📊 Statistics")

        if state_manager.state.filteredObjects is None:
            st.markdown("")
            st.info("No results for statistics calculation.")
        else:
            with filtrationCol:
                filter_params = ui.render_filtration_ui(filtrationCol)
                if filter_params:
                    handlers.handle_filtration(filter_params)
            
            with statCol:
                if state_manager.state.filteredObjectsInfo is None:
                    state_manager.state.filteredObjectsInfo = prepareFilteredObjectInfo(
                        state_manager.state.filteredObjects
                )
                resultInfo = calculateStatistics(
                    state_manager.state.filteredObjectsInfo,
                    scale=state_manager.state.imgScale
                )
                
                state_manager.state.imgArea = (state_manager.state.imgWidth * 
                          (state_manager.state.imgHeight - state_manager.state.infoLineHeight) *
                          (state_manager.state.imgScale ** 2))
                
                ui.render_statistics_content(statCol, resultInfo, state_manager.state.imgArea)
                st.markdown("📊 Tools")
                st.markdown("")
                ui.render_postprocess_toggle()
                ui.render_scale_bar()
        # Загрузчик аннотаций
        st.markdown("📂 Import annotations")
        
        if state_manager.state.uploadedImage is not None:
            uploadedAnn = ui.render_annotation_uploader_only()
            if uploadedAnn and uploadedAnn != state_manager.state.get("last_uploaded_ann"):
                state_manager.state.polygonsCVAT = makeCVATbackup(
                    state_manager.state.uploadedImage,
                    state_manager.state.imageName,
                    {},
                    state_manager.state.imgWidth,
                    state_manager.state.imgHeight,
                    state_manager.state.model_config
                )
                if handlers.handle_annotation_apply(uploadedAnn):
                    state_manager.state.last_uploaded_ann = uploadedAnn
                    handlers.handle_statistics_update()
                    ui.clear_image_cache()
                    st.rerun()
        else:
            st.markdown("")
            st.info("Upload an image first to import annotations.")
        
        # Кнопки экспорта во вкладке Statistics
        st.markdown("📤 Export results")

        if ui.render_export_buttons(suffix="stats") == "prepare":
            handlers.prepare_export_data()
            st.rerun()

# === Левая панель: Workflow ===
with blockWorkspace:
    # Отрисовка рабочей области
    if state_manager.state.uploadedImage:
        overlay(state_manager.state.uploadedImage, 
                overlays=state_manager.state.overlays,
                styles=state_manager.state.overlayStyles,
            )
                    

    # Помощь
    with st.expander("❓ Help", expanded=False):
        st.markdown("<p>1. You can check user manual <a href='https://disk.yandex.ru/d/SHoSu6qIvqi62w'>here</a>.</p>", 
                    unsafe_allow_html=True)
        st.markdown("<p>2. An examples of SEM-images is available <a href='https://disk.yandex.ru/d/sp1UwEoEBgbyCw'>here</a>.</p>", 
                    unsafe_allow_html=True)
        st.markdown("<p>3. If you have any problems, you can try to clear cash:</p>", 
                    unsafe_allow_html=True)
        
        if st.button("♻ Clear cache"):
            st.cache_data.clear()
            ui.clear_image_cache()
            st.rerun()

        st.markdown("<p>Contact e-mail: pawlova12@yandex.ru</p>", 
                    unsafe_allow_html=True)

loadFooter()
