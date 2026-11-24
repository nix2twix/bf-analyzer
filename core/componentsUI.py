import numpy as np
import streamlit as st
from PIL import Image
from datetime import datetime
from typing import Dict, Any, Optional
from core.factoryUI import ModelUIFactory
from utils.examples import load_examples, get_example_by_name
from streamlit_image_overlay import streamlit_image_overlay as overlay 

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
    
                return display_image
            else:
                st.warning("⚠️ No objects found after filtering")
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
    
    def clear_imageCache(self):
        """Очистка кэша изображений"""
        st.session_state.image_cache = {}
    
    def render_fileUploader(self):
        """Отрисовывает загрузчик файла"""
        return st.file_uploader(
            label="Upload SEM image or ZIP archive",
            label_visibility="collapsed",
            type=["bmp", "png", "jpg", "jpeg", "tif", "tiff", "zip"],
            key="uploader"
        )
    def render_scaleSelector(self) -> list:
        _container = st.container(horizontal=True, vertical_alignment="center")
        with _container:
            scale_options = ["First image only", "Each image", "Skip detection"]
            _scale_mode = st.segmented_control(
                "Scale detection:",
                options=scale_options,
                default=scale_options[0],
                key="batch_scale_mode",
                selection_mode="single",
                required=True,
                width="stretch",
                help=f"Detect or scip the detection of scale bar per image. For each image it could be slow. If all images in batch have same scale, run \'first image only\'"
            )
        return scale_options, _scale_mode
    
    def render_modelSelector(self) -> str:
        """Отрисовывает селектор модели"""
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


    
    def render_batchSelectorUI(self, _raw):
        """
        Предпросмотр изображений из архива с возможностью выбора файлов.
        Выбор сохраняется в batch_selected_map независимо от текущей страницы.
        """
        import io
        from PIL import Image as PILImage, ImageOps

        if not _raw:
            return

        selected_map = self.state.state.batch_selected_map
        # === Управление выбором ===
        _container = st.container(horizontal=True, vertical_alignment="center")
        with _container:
            if st.button("Select all", key="batch_select_all", width="stretch", type="primary"):
                for name, _ in _raw:
                    selected_map[name] = True
                st.rerun()
            if st.button("Deselect all", key="batch_deselect_all", width="stretch"):
                for name, _ in _raw:
                    selected_map[name] = False
                st.rerun()

        selected_count = sum(
            selected_map.get(name, True) for name, _ in _raw
        )

        with _container:
            st.caption(
            f"Selected: {selected_count} / {len(_raw)} images"
            )
        
        # === Пагинация ===
        PAGE_SIZE = 4
        total_pages = max(1, (len(_raw) + PAGE_SIZE - 1) // PAGE_SIZE)

        if total_pages > 1:
            page = st.pagination(
                num_pages=total_pages,
                default=1,
                key="batch_preview_page",
                width="stretch"
            )
        else:
            page = 1

        start = (page - 1) * PAGE_SIZE
        chunk = _raw[start:start + PAGE_SIZE]

        # === Предпросмотр ===
        THUMB_W = 480
        THUMB_H = 360

        cols = st.columns(PAGE_SIZE, gap="small")

        for col, (name, data) in zip(cols, chunk):
            key = f"batch_chk_{name}"

            # Восстанавливаем состояние чекбокса из общего словаря
            st.session_state[key] = selected_map.get(name, True)

            with col:
                try:
                    img = PILImage.open(io.BytesIO(data))
                    img = ImageOps.exif_transpose(img).convert("L")

                    # Масштабируем изображение, не изменяя пропорции
                    img.thumbnail(
                        (THUMB_W, THUMB_H),
                        PILImage.Resampling.LANCZOS,
                    )

                    # Создаём холст с пропорциями 4:3
                    canvas = PILImage.new(
                        "L",
                        (THUMB_W, THUMB_H),
                        color=32,
                    )

                    x = (THUMB_W - img.width) // 2
                    y = (THUMB_H - img.height) // 2
                    canvas.paste(img, (x, y))
                    st.image(canvas)

                except Exception:
                    st.warning(f"⚠️ Can't preview {name}")
                st.markdown(
                    """
                    <style>
                    div[data-testid="stCheckbox"] label p {
                        display: -webkit-box;
                        -webkit-box-orient: vertical;
                        -webkit-line-clamp: 2;
                        overflow: hidden;
                        overflow-wrap: anywhere;
                        line-height: 1.25;
                        max-height: 2.5em;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                st.checkbox(
                    name,
                    key=key,
                    on_change=lambda n=name, k=key: (
                        selected_map.__setitem__(
                            n, st.session_state[k]
                        )
                    ),
                )


    def render_batchSummaryTable(self):
        """
        Таблица batch-сводки через st.dataframe по 4 картинки.
        Цвета классов — через эмодзи в заголовках (column_config).
        """
        import pandas as pd

        items = self.state.state.get("batch_items", []) or []
        if not items:
            st.info("No batch results yet.")
            return

        items = sorted(items, key=lambda it: it.unique_name.lower())
        config = self.state.getConfig()
        if config is None:
            return

        class_names = [
            c for c in config.class_names
            if "background" not in c and c != "bg"
        ]

        def _class_title(c):
            if hasattr(config, "class_titles") and c in config.class_titles:
                return config.class_titles[c][0]
            return c

        _COLOR_EMOJI = ["🟢", "🟡", "🟣", "🔵", "⚪", "🔴"]
        emoji_for_class = {
            c: _COLOR_EMOJI[i % len(_COLOR_EMOJI)]
            for i, c in enumerate(class_names)
        }

        PAGE_SIZE = 8
        total_pages = max(1, (len(items) + PAGE_SIZE - 1) // PAGE_SIZE)
        if total_pages > 1:
            page = st.pagination(
                num_pages=total_pages, default=1, key="batch_summary_page",
            )
        else:
            page = 1

        start = (page - 1) * PAGE_SIZE
        page_items = items[start:start + PAGE_SIZE]

        # --- Данные ---
        rows = []
        for idx, it in enumerate(page_items, start=start + 1):
            has_scale = bool(getattr(it, "scale_ok", False)) and it.scale > 0
            row = {
                "#": idx,
                "Image": it.unique_name,
            }
            total_px = it.width * it.height
            for c in class_names:
                title = _class_title(c)
                stats = (it.stats or {}).get(c, {})
                count = stats.get("count", 0)
                area_px = stats.get("total_area_px", 0)
                area_um2 = round(area_px * (it.scale ** 2), 2) if has_scale else "—"
                percent = (area_px / total_px * 100.0) if total_px > 0 else 0.0
                row[f"{title} count"] = count
                if has_scale:
                    row[f"{title} area (µm²)"] = area_um2
                else: 
                    row[f"{title} area (px²)"] = area_px 
                row[f"{title} % area"] = round(percent, 2)
            row["Scale (µm/px)"] = round(it.scale, 4) if has_scale else "—"
            rows.append(row)

        df = pd.DataFrame(rows)

        column_config = {
            "#": st.column_config.NumberColumn("#", width="small", format="%d"),
            "Image": st.column_config.TextColumn("Image", width="medium"),
            "Scale (µm/px)": st.column_config.TextColumn(
                "Scale (µm/px)",
                width="small",
                help="Detected from the scale bar. '—' means not detected.",
            ),
        }
        for c in class_names:
            title = _class_title(c)
            emoji = emoji_for_class.get(c, "⚪")
            column_config[f"{title} count"] = st.column_config.NumberColumn(
                f"{emoji} {title} count", width="small", format="%d",
                help=f"Number of detected objects of class {title}",
            )
            column_config[f"{title} area (px²)"] = st.column_config.NumberColumn(
                f"{emoji} area\n(px²)", width="small", format="%d",
                help=f"{title}: total area in pixels²",
            )
            column_config[f"{title} area (µm²)"] = st.column_config.TextColumn(
                f"{emoji} area\n(µm²)", width="small",
                help=f"{title}: total area in µm² ('—' if scale not detected)",
            )
            column_config[f"{title} % area"] = st.column_config.NumberColumn(
                f"{emoji} %\narea", width="small", format="%.2f",
                help=f"{title}: percent of image area",
            )

        st.dataframe(
            df,
            column_config=column_config,
            hide_index=True,
            width="content"
        )

        # --- Ошибки ---
        _err = [it for it in items if it.error is not None]
        if _err:
            st.error(f"⚠️ {len(_err)} file(s) failed")
            with st.expander("Show errors", expanded=False):
                for it in _err:
                    st.markdown(f"❌ **{it.unique_name}** — {it.error}")

    def render_segmentationButton(self) -> bool:
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

    def render_overlayUI(self):
        _overlays = self.state.uniteOverlayPaths(
                        self.state.state.overlays, 
                        self.state.state.scale_overlay
                    )

        _styles = self.state.uniteOverlayStyles([
                    self.state.state.overlayStyles,
                    self.state.state.scaleOverlayStyles
                    ])

        overlay(
            self.state.state.uploadedImage,
            overlays=_overlays,
            styles=_styles,
            key="main-image-viewer",
        )
    
    def render_annotationUploader(self) -> Optional[Any]:
        """Отрисовывает только загрузчик аннотаций"""
        return st.file_uploader(
            "",
            type=['zip'],
            key="annotation_uploader_stats",
            disabled=False,
            help="Upload annotations exported from CVAT to recalculate statistics"
        )
    
    def render_filtrationUI(self, scale) -> Dict:
        """Отрисовывает UI фильтрации."""
        if self.state.state.get("batch_mode", False):
            st.info("⚙️ Filters are disabled for batch processing.")
            return {}
        config = self.state.getConfig()

        if config and self.state.state.predictedObjects is not None:
            return self.factory.create_filtration_ui(
                config,
                self.state.state,
                scale
            )

        return {}
    def render_statisticsContent(self, result_info: Dict, img_area: float):
        """Отрисовывает контент статистики"""
        config = self.state.getConfig()
        if self.state.state.filteredObjects is not None and config and result_info:
            self.factory.create_statistics_ui(config, result_info, img_area)
    
    def render_exportButtons(self, suffix: str = ""):
        col1, col2 = st.columns(2)

        cvat_key = f"cvat_export_{suffix}" if suffix else "cvat_export_main"
        results_key = f"results_export_{suffix}" if suffix else "results_export_main"

        is_batch = self.state.state.get("batch_mode", False)

        if is_batch:
            n_ok = sum(
                1 for it in (self.state.state.get("batch_items") or []) if it.ok
            )
            cvat_file_name = f"backup-{n_ok}-{datetime.now().strftime('%d-%m-%Y-%H-%M')}.zip"
        else:
            cvat_file_name = f"backup-1-{datetime.now().strftime('%d-%m-%Y-%H-%M')}.zip"

        if self.state.state.export_dirty:
            if is_batch:
                has_data = bool(self.state.state.get("batch_items"))
            else:
                has_data = self.state.state.filteredObjects is not None

            if st.button(
                "⬇️ Save changes",
                key=f"prepare_results_{suffix}",
                width="stretch",
                disabled=not has_data,
            ):
                return "prepare"
            return None

        with col1:
            st.download_button(
                label="📥 CVAT backup",
                data=self.state.state.polygonsCVAT or b"",
                file_name=cvat_file_name,
                mime="application/zip",
                disabled=(
                    (not is_batch and self.state.state.uploadedImage is None)
                    or self.state.state.polygonsCVAT is None
                ),
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
                disabled=(
                    (not is_batch and self.state.state.uploadedImage is None)
                    or self.state.state.zipBuffer is None
                ),
                width='stretch',
                help="Download segmentation results",
                key=results_key
            )

    def render_postprocessToggle(self):
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

    def render_batchSegmentationButton(self) -> bool:
        if self.state.state.get("batch_in_progress", False):
            st.info("⏳ Batch processing in progress...")
            return False
        return st.button(
            "🚀 Start batch segmentation",
            disabled=not self.state.state.get("batch_images_raw"),
            width="stretch",
            key="batch_start_btn",
        )

    def render_batchSummary(self):
        items = self.state.state.get("batch_items", []) or []
        if not items:
            return
        ok = sorted((it for it in items if it.ok), key=lambda it: it.unique_name.lower())
        err = sorted((it for it in items if not it.ok), key=lambda it: it.unique_name.lower())

        st.markdown(
            f"**Batch results:** {len(ok)} processed, {len(err)} with errors."
        )
        with st.expander("Show details", expanded=False):
            for it in ok:
                counts = {k: v["count"] for k, v in (it.stats or {}).items() if v.get("count")}
                st.write(f"✅ **{it.unique_name}** — {counts or 'no objects'}")
            for it in err:
                st.write(f"❌ **{it.unique_name}** — {it.error}")

    def renderExamples(self):
        examples = load_examples()
        if not examples:
            st.info("Examples are not available.")
        else:
            st.markdown(
                """
                <style>
                div[data-testid="stImage"] img {
                    object-fit: cover;
                    border-radius: 8px;
                }
                .example-caption {
                    font-size: 0.78rem;
                    line-height: 1.15;
                    color: #9aa0a6;
                    min-height: 2.3em;
                    margin: 0.35rem 0 0.25rem 0;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            current_name = st.session_state.get("selected_example_name")
            uploader_active = st.session_state.get("uploader") is not None
            cols = st.columns(len(examples), gap="small")
            for i, (col, ex) in enumerate(zip(cols, examples)):
                with col:
                    st.image(ex.getvalue(), use_container_width=True)
                    st.markdown(
                        f'<div class="example-caption">{ex.description}</div>',
                        unsafe_allow_html=True,
                    )

                    is_current = (current_name == ex.name)
                    label = "Using" if is_current else "Use it"
                    btype = "primary" if is_current else "secondary"

                    if st.button(
                        label,
                        type=btype,
                        key=f"use_example_{ex.name}",
                        use_container_width=True,
                        disabled=uploader_active,
                    ):
                        st.session_state.selected_example_name = ex.name
                        st.session_state.pop("lastUploadedFile", None)
                        st.rerun()

