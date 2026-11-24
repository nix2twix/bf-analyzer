# === GENERAL LIBS 
import streamlit as st

# === PROJECT LIBS 
from segmentation.objects import prepare_filteredObjectInfo
from utils.examples import get_example_by_name
       
from core.stateManager import StateManager
from core.handlers import AppHandlers
from core.componentsUI import UIComponents
from core.messages import Message, Error
from core.guideContent import GuideContent
# === PAGE 
stManager = StateManager()
handlers = AppHandlers(stManager)
ui = UIComponents(stManager)
contentCol_workspace, gap, contentCol_tools = st.columns([2.5, 0.01, 1.5])

# === PAGE CONTENT ===
with contentCol_tools:
    contentTab_tools = st.tabs(["🔬 Segmentation", "📊 Statistics & Annotations"])
    
    # Вкладка 1: Segmentation
    with contentTab_tools[0]:
        # 1. Загрузчик изображения или обработка выбранного примера
        _uploadedFile = ui.render_fileUploader()
        _exampleName = st.session_state.get("selected_example_name")
        if _exampleName and st.session_state.get("uploader") is None:
            _example = get_example_by_name(_exampleName)
            if _example is not None:
                _uploadedFile = _example

        if st.session_state.get("uploader") is not None:
            st.session_state.selected_example_name = None

        # 2. Выбор модели
        _modelSelected = ui.render_modelSelector()
        if _modelSelected:
            if stManager.setModel(_modelSelected):
                ui.clear_imageCache()
    
        # Обработка загрузки файла
        _current_name = getattr(_uploadedFile, "name", None) if _uploadedFile else None
        if _uploadedFile and _current_name != stManager.state.get("lastUploadedFileName"):
            handlers.handle_fileUpload(_uploadedFile)
            stManager.state.lastUploadedFileName = _current_name
        
        # 3. Запуск сегментации
        if stManager.state.get("batch_mode", False):
            # === BATCH ===
            _raw = stManager.state.get("batch_images_raw", [])
            _selected_map = stManager.state.get("batch_selected_map") or {}
            stManager.state.batch_selected_names = [
                _n for _n, _ in _raw
                if _selected_map.get(_n, True)
            ]
            scale_options, scale_mode = ui.render_scaleSelector()
            if ui.render_batchSegmentationButton():
                handlers.handle_batchSegmentation(scale_mode=scale_mode)
        else:
            # === SINGLE ===
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
            
    st.markdown("📤 Export results")
    if ui.render_exportButtons(suffix="seg") == "prepare":
        if stManager.state.get("batch_mode", False):
            handlers.prepare_batchExportData()
        else:
            handlers.prepare_exportData()
        st.rerun()
        
    # === Вкладка 2: Statistics & Import ===
    with contentTab_tools[1]:
        if stManager.state.get("batch_mode", False):
            st.info("📦 Batch mode: filters and per-image statistics are disabled.")
        else:
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

        st.markdown("📂 Import CVAT-annotations")
        _is_batch_or_empty = (
            stManager.state.get("batch_mode", False)
            or stManager.state.uploadedImage is None
        )

        if _is_batch_or_empty:
            _uploadedAnn = st.file_uploader(
                "",
                label_visibility="collapsed",
                type=["zip"],
                key="batch_ann_uploader",
                help = "Upload backup to compute statistics for batch or when no image is loaded",
            )
            if _uploadedAnn and _uploadedAnn != stManager.state.get("lastBatchAnn"):
                ok = handlers.handle_batchAnnotationUpload(_uploadedAnn)
                if ok:
                    stManager.state.lastBatchAnn = _uploadedAnn
                    stManager.state.lastBackupOnly = _uploadedAnn
                    st.rerun()
        else:
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

# === Левая панель: Workspace ===
with contentCol_workspace:
    if stManager.state.get("batch_mode", False):
        _raw = stManager.state.get("batch_images_raw", [])
        _selected_map = stManager.state.get("batch_selected_map") or {}
        for _n, _ in _raw:
            _key = f"batch_chk_{_n}"
            if _key not in st.session_state:
                st.session_state[_key] = _selected_map.get(_n, True)

        st.markdown(f"### 📦 Batch processing")
        with st.expander("Select files to process", expanded=True):
            ui.render_batchSelectorUI(_raw)
            _selected_map = stManager.state.get("batch_selected_map") or {}

            stManager.state.batch_selected_names = [
                _n for _n, _ in _raw
                if _selected_map.get(_n, True)
            ]
        st.markdown("### 📊 Processing summary")
        _items = stManager.state.get("batch_items", []) or []
        if _items:
            ui.render_batchSummaryTable()
    else:
        if stManager.state.uploadedImage:
            stManager.state.scale_overlay = stManager.render_scaleOverlay()
            ui.render_overlayUI()

    # === Экспендеры — общие для обоих режимов ===
    with st.expander("🖼️ Examples", expanded=True):
        ui.renderExamples()

    with st.expander("🚀 Quick start guide", expanded=False):
        GuideContent.quickStart("EN")
        st.page_link(
            "pages/docs.py",
            label="📖 Click to open full documentation",
            width="stretch",
        )

    with st.expander("📚 Citation & Contacts", expanded=False):
        Message.citationAndContacts()

