# === GENERAL LIBS 
import streamlit as st

# === PROJECT LIBS 
from segmentation.objects import prepare_filteredObjectInfo
from utils.importer import checkAnnotationSize
from styles.styles import (
    loadStyles, 
    loadFooter,
    loadHeader
)
       
from core.stateManager import StateManager
from core.handlers import AppHandlers
from core.componentsUI import UIComponents
from core.messages import Message, Error

# === PAGE 
st.set_page_config(page_title="Biofilm Analyzer", layout="wide", page_icon="🧪")
loadStyles()
loadHeader()

stManager = StateManager()
handlers = AppHandlers(stManager)
ui = UIComponents(stManager)
contentCol_workspace, gap, contentCol_tools = st.columns([2.5, 0.01, 1.5])

# === PAGE CONTENT ===
with contentCol_tools:
    contentTab_tools = st.tabs(["🔬 Segmentation", "📊 Statistics & Tools"])
    
    # Вкладка 1: Segmentation
    with contentTab_tools[0]:
        # 1. Загрузчик изображения
        _uploadedFile = ui.render_fileUploader()
        
        # 2. Выбор модели
        _modelSelected = ui.render_modelSelector()
        if _modelSelected:
            if stManager.setModel(_modelSelected):
                ui.clear_imageCache()
    
        # Обработка загрузки файла
        if _uploadedFile and (_uploadedFile != stManager.state.get("lastUploadedFile")):
            handlers.handle_fileUpload(_uploadedFile)
            stManager.state.lastUploadedFile = _uploadedFile
            ui.clear_imageCache()
        
        # 3. Запуск сегментации
        seg_clicked = ui.render_segmentationButton()
        if seg_clicked:
            handlers.handle_segmentation()
            stManager.state.last_uploaded_ann = None
            stManager.state.uploadedAnnName = None
            ui.clear_imageCache()
        
        # Определение масштаба
        if stManager.state.uploadedImage is not None and stManager.state.scaleInfo is None:
            handlers.handle_scaleDetection()

        if stManager.state.imgScale is not None:
            handlers.handle_imgAreaScaled()

        # Обновление статистики
        if (stManager.state.predictedObjects is not None and 
            stManager.state.objectsInfo is None):
            handlers.handle_statisticsUpdate()
            
        # 4. Кнопки экспорта
        st.markdown("📤 Export results")
    if ui.render_exportButtons(suffix = "seg") == "prepare":
        handlers.prepare_exportData()
        st.rerun()

    # === Вкладка 2: Statistics & Import ===
    with contentTab_tools[1]:
        contentCol_filtration, gap, contentCol_statistics = st.columns([1.5, 0.01, 1.6])
        with contentCol_filtration:
            st.markdown("⚙️ Filters")

        with contentCol_statistics:
            st.markdown("📊 Statistics")

        if stManager.state.filteredObjects is None:
            st.markdown("")
            Message.noStatistisResults()
        else:
            with contentCol_filtration:
                _filterParams = ui.render_filtrationUI(stManager.state.imgScale)
                if _filterParams:
                    handlers.handle_filtration(_filterParams)
            
            with contentCol_statistics:
                if stManager.state.filteredObjectsInfo is None:
                    stManager.state.filteredObjectsInfo = prepare_filteredObjectInfo(
                        stManager.state.filteredObjects
                )
                _resultInfo = handlers.handle_statisticsAppear()                
               
                ui.render_statisticsContent(_resultInfo, stManager.state.imgAreaScaled)
                st.markdown("📊 Tools")
                st.markdown("")
                ui.render_postprocessToggle()
                #ui.render_scale_bar()

        st.markdown("📂 Import annotations")
        if stManager.state.uploadedImage is not None:
            _uploadedAnn = ui.render_annotationUploader()
            if _uploadedAnn and _uploadedAnn != stManager.state.get("lastUploadedAnn"):
                if handlers.handle_annotationUpload(_uploadedAnn):
                    handlers.handle_annotationApply(_uploadedAnn)
                    stManager.state.lastUploadedAnn = _uploadedAnn
                    handlers.handle_statisticsUpdate()
                    ui.clear_imageCache()
                    st.rerun()
                else:
                    Error.cantMatchAnnotation()
        else:
            st.markdown("")
            Message.needUploadImageToAnnotate()
        
        st.markdown("📤 Export results")
# === Левая панель: Workspace ===
with contentCol_workspace:
    if stManager.state.uploadedImage:
        stManager.state.scale_overlay = stManager.render_scaleOverlay()
        ui.render_overlayUI()

    # Новости
    with st.expander("🆕 What's new?", expanded=False):
        language = st.segmented_control(
            "Language",
            ["RU", "EN"],
            default="RU",
            label_visibility="collapsed",
            key="news_interface_language"
        ) 
        Message.whatsNew(language)
        st.markdown("")
    # Помощь
    with st.expander("❓ Help", expanded=False):
        language = st.segmented_control(
            "Language",
            ["RU", "EN"],
            default="RU",
            label_visibility="collapsed",
            key="help_interface_language"
        ) 
        Message.help(language)
        st.markdown("")
        if st.button("♻ Clear cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            ui.clear_imageCache()
            st.rerun()

loadFooter()
