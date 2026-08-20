# === LIBRARIES GENERAL ===
import streamlit as st
from typing import Dict

# === PROJECT SCRIPTS ===
from models.configs import ModelConfig

class ModelUIFactory:
    """Фабрика для создания UI-элементов специфичных для модели"""
    
    @staticmethod
    def _on_slider_change(param_name: str, state: Dict, scale: float):
        
        """Callback для обновления параметров сразу при изменении слайдера"""
        def callback():
            # Получаем текущее значение слайдера из session_state
            slider_key = f"slider_{param_name}_{hash(str(state.get('predictedObjects')))}"
            if slider_key in st.session_state:
                min_val, max_val = st.session_state[slider_key]
                if "filtration_params" in state:
                    state["filtration_params"][param_name] = {
                        'min': float(min_val),
                        'max': float(max_val)
                    }
                    state["filters_dirty"] = True
        return callback


    @staticmethod
    def _area_px_to_um2(value, scale):
        if scale is None:
            return value
        return value * scale ** 2


    @staticmethod
    def _area_um2_to_px(value, scale):
        if scale is None or scale == 0:
            return value
        return value / scale ** 2
    

    @staticmethod
    def create_filtration_ui(config: ModelConfig, state: Dict, scale: float) -> Dict:
        """Динамическое создание UI фильтрации на основе конфига."""

        params = {}
        css_rules = []

        if "slider_ranges" not in state:
            state["slider_ranges"] = {}

        if "filtration_params" not in state:
            state["filtration_params"] = {}

        css_rules.append("""
            div[data-testid="stSlider"] {
                pointer-events: auto !important;
            }

            div[data-testid="stSlider"] > div {
                pointer-events: auto !important;
            }

            div[data-testid="stSlider"] [role="slider"] {
                pointer-events: auto !important;
                cursor: pointer !important;
            }
        """)

        for param_name, (min_default, max_default) in config.filtration_params.items():

            # ---------------------------------------------------------
            # Диапазон фильтра в пикселях
            # ---------------------------------------------------------
            if param_name in state["slider_ranges"]:
                slider_min, slider_max = state["slider_ranges"][param_name]

                if slider_min >= slider_max:
                    slider_min, slider_max = min_default, max_default
            else:
                slider_min, slider_max = min_default, max_default

            slider_min = float(slider_min)
            slider_max = float(slider_max)

            # ---------------------------------------------------------
            # Текущее значение фильтра в пикселях
            # ---------------------------------------------------------
            current = state["filtration_params"].get(param_name)

            if isinstance(current, dict):
                current_min = current.get("min", slider_min)
                current_max = current.get("max", slider_max)
            else:
                current_min = slider_min
                current_max = slider_max

            current_min = float(current_min)
            current_max = float(current_max)

            # Ограничиваем текущие значения диапазоном
            current_min = max(slider_min, min(current_min, slider_max))
            current_max = max(current_min, min(current_max, slider_max))

            # ---------------------------------------------------------
            # Тип параметра
            # ---------------------------------------------------------
            is_area = "area" in param_name

            # ---------------------------------------------------------
            # Перевод px² → μm² только для отображения
            # ---------------------------------------------------------
            if is_area and scale is not None and scale > 0:
                factor = float(scale ** 2)

                display_min = slider_min * factor
                display_max = slider_max * factor

                display_current_min = current_min * factor
                display_current_max = current_max * factor

                step = factor
            else:
                display_min = slider_min
                display_max = slider_max

                display_current_min = current_min
                display_current_max = current_max

                step = 1.0 if is_area else 0.01


            # Цвет
            color = "#24b353"

            for class_name, (title, hex_color) in config.class_titles.items():
                if class_name in param_name:
                    color = hex_color
                    break

            # Уникальный ключ
            key = f"slider_{param_name}_{state.get('modelType', '')}"

            css_rules.append(f"""
                div[data-testid="stSlider"][data-key="{key}"] [role="slider"] {{
                    background-color: {color} !important;
                    box-shadow: 0 0 0 2px {color} !important;
                }}

                div[data-testid="stSlider"][data-key="{key}"]
                [data-testid="stSliderThumbValue"] {{
                    color: {color} !important;
                }}
            """)

            # Название

            display_name = param_name.replace("_", " ").title()

            if is_area:
                display_name += f" ({state.get("scale_unit", "μm")}²)"
            # Slider
            slider_value = st.slider(
                display_name,
                min_value=float(display_min),
                max_value=float(display_max),
                value=(
                    float(display_current_min),
                    float(display_current_max)
                ),
                step=float(step),
                key=key,
                disabled=state.get("predictedObjects") is None,
            )

            # ---------------------------------------------------------
            # ВАЖНО:
            # переводим результат UI обратно в px²
            # и сохраняем КАЖДЫЙ параметр
            # ---------------------------------------------------------
            if is_area and scale is not None and scale > 0:
                factor = float(scale ** 2)

                params[param_name] = {
                    "min": float(slider_value[0]) / factor,
                    "max": float(slider_value[1]) / factor,
                }
            else:
                params[param_name] = {
                    "min": float(slider_value[0]),
                    "max": float(slider_value[1]),
                }

        # -------------------------------------------------------------
        # CSS
        # -------------------------------------------------------------
        if css_rules:
            st.markdown(
                f"<style>{''.join(css_rules)}</style>",
                unsafe_allow_html=True
            )

        return params

    @staticmethod
    def create_statistics_ui(config: ModelConfig, result_info: Dict, img_area: float):
        """Динамическое создание UI статистики"""
        bacillus_classes = [c for c in config.class_names if "Bacillus" in c]
        coccus_classes = [c for c in config.class_names if "Coccus" in c]
        other_classes = [c for c in config.class_names if c not in bacillus_classes + coccus_classes and c not in ["bg", "background"]]
        
        if bacillus_classes:
            st.markdown("#### 🎯 Bacillus")
            for class_name in bacillus_classes:
                if class_name in result_info:
                    ModelUIFactory._render_statistic_card(
                        class_name, result_info[class_name], config, img_area
                    )
        
        if coccus_classes:
            st.markdown("#### 🧫 Coccus")
            for class_name in coccus_classes:
                if class_name in result_info:
                    ModelUIFactory._render_statistic_card(
                        class_name, result_info[class_name], config, img_area
                    )
        
        for class_name in other_classes:
            if class_name in result_info:
                ModelUIFactory._render_statistic_card(
                    class_name, result_info[class_name], config, img_area
                )
        st.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)
    @staticmethod
    def _render_statistic_card(class_name: str, stats: Dict, config: ModelConfig, img_area: float):
        """Универсальная отрисовка карточки статистики"""
        title, text_color = config.class_titles.get(
            class_name, 
            (class_name.replace("Bacillus", "").replace("Coccus", "").capitalize(), "#ffffff")
        )
        
        st.markdown(f"""
            <div style="font-size: 0.9rem; margin-top: 0.5rem; color: {text_color};">
                {title} (counted {stats['count']})<br>
                Total area: {stats['total_area_mkm']:.2f} μm² ({((stats['total_area_mkm'] / img_area) * 100):.2f}%) 
            </div>
        """, unsafe_allow_html=True)
        
