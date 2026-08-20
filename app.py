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
blockWorkspace, gap, blockTools = st.columns([2.5, 0.01, 1.5])

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
                filter_params = ui.render_filtration_ui(
                    filtrationCol,
                    state_manager.state.imgScale
                )

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
                #ui.render_scale_bar()
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
        state_manager.state.scale_overlay = state_manager.get_scale_overlay()

        overlays = (
            state_manager.state.overlays
            + state_manager.state.scale_overlay
        )

        styles = state_manager.uniteOverlayStyles([
            state_manager.state.overlayStyles,
            state_manager.state.scaleOverlayStyles
        ])

        overlay(
            state_manager.state.uploadedImage,
            overlays=overlays,
            styles=styles,
            key="main-image-viewer",
        )
 
    # Новости
    with st.expander("🆕 What's new?", expanded=False):
        language = st.segmented_control(
            "Language",
            ["RU", "EN"],
            default="RU",
            label_visibility="collapsed",
            key="news_interface_language"
        ) 
        if language == "RU":
            st.markdown("""
                ### **Работа с оверлеем** 
                --- 
                🔍 **Масштабирование**

                    Наведите курсор на рабочую область, ограниченную серой рамкой, и используйте колесо мыши.

                ↩️ **Сброс масштаба**
                 
                    Выполните двойной клик левой кнопкой мыши по изображению.

                🖱️ **Информация об объекте** 
                
                    Наведите курсор на распознанный объект: он подсветится, а рядом появится подсказка с его параметрами.
            """)
        else:
            st.markdown("""
                ### **Using the overlay**
                ---
                
                🔍 **Zoom** 
                
                    Place the cursor inside the workspace outlined by the gray border and use the mouse wheel.

                ↩️ **Reset zoom**
                    
                    Double-click the left mouse button on the image.

                🖱️ **Object information** 
                    
                    Hover over a detected object: it will be highlighted and a tooltip with its parameters will appear.
            """)
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
        if language == "RU":
            st.markdown("""
                📖 **Руководство** — подробная инструкция по работе с Biofilm Analyzer доступна [здесь](https://disk.yandex.ru/i/67FqW7pGcJ6ELg).

                🖼️ **Примеры изображений** — примеры SEM-снимков доступны [здесь](https://disk.yandex.ru/d/sp1UwEoEBgbyCw).

                ♻️ **Очистка кэша** — если сайт работает некорректно, попробуйте очистить кэш с помощью кнопки ниже.

                ✉️ **Обратная связь** — pawlova12@yandex.ru
            """)
        else: 
            st.markdown("""
                📖 **User manual** — a detailed user guide is available [here](https://disk.yandex.ru/i/67FqW7pGcJ6ELg).

                🖼️ **Image examples** — examples of SEM images are available [here](https://disk.yandex.ru/d/sp1UwEoEBgbyCw).

                ♻️ **Clear cache** — if the website is not working correctly, try clearing the cache using the button below.

                ✉️ **Contact** — pawlova12@yandex.ru
            """)
        st.markdown("")
        if st.button("♻ Clear cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            ui.clear_image_cache()
            st.rerun()

loadFooter()
