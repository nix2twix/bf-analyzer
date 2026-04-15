# === LIBRARIES GENERAL ===
import streamlit as st
from typing import Dict

# === PROJECT SCRIPTS ===
from .modelConfigs import ModelConfig

class ModelUIFactory:
    """Фабрика для создания UI-элементов специфичных для модели"""
    
    @staticmethod
    def create_filtration_ui(config: ModelConfig, state: Dict) -> Dict:
        """Динамическое создание UI фильтрации на основе конфига"""
        params = {}
        css_rules = []

        if "slider_ranges" not in state:
            state["slider_ranges"] = {}
        if "filtration_params" not in state:  
            state["filtration_params"] = {}

        # Добавляем CSS для улучшения отзывчивости слайдеров
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
            # Получаем диапазон слайдера из состояния
            if param_name in state["slider_ranges"]:
                slider_min, slider_max = state["slider_ranges"][param_name]
                if slider_min >= slider_max:
                    slider_min, slider_max = min_default, max_default
            else:
                slider_min, slider_max = min_default, max_default

            # Получаем текущее значение из filtration_params
            if param_name in state["filtration_params"]:
                current = state["filtration_params"][param_name]
                if isinstance(current, dict):
                    current_min = current.get('min', slider_min)
                    current_max = current.get('max', slider_max)
                else:
                    current_min = slider_min
                    current_max = current
            else:
                current_min = slider_min
                current_max = slider_max

            # Корректируем значения
            current_min = max(slider_min, min(current_min, slider_max))
            current_max = min(slider_max, max(current_min, current_max))

            # Определяем тип параметра
            is_area = "area" in param_name
            step = 1.0 if is_area else 0.01

            # Красивое имя
            display_name = param_name.replace("_", " ").title()

            # Определяем цвет
            color = "#24b353"
            for class_name, (title, hex_color) in config.class_titles.items():
                if class_name in param_name:
                    color = hex_color
                    break

            # Уникальный ключ с timestamp для избежания конфликтов
            import time
            key = f"slider_{param_name}_{hash(str(state.get('predictedObjects')))}"

            # Добавляем CSS правило для цвета
            css_rules.append(f"""
                div[data-testid="stSlider"][data-key="{key}"] [role="slider"] {{
                    background-color: {color} !important;
                    box-shadow: 0 0 0 2px {color} !important;
                }}
                div[data-testid="stSlider"][data-key="{key}"] [data-testid="stSliderThumbValue"] {{
                    color: {color} !important;
                }}
            """)

            # Создаем слайдер
            params[param_name] = st.slider(
                display_name,
                min_value=float(slider_min),
                max_value=float(slider_max),
                value=(float(current_min), float(current_max)),
                step=step,
                key=key, 
                disabled=state.get("predictedObjects") is None
            )

        # Применяем стили
        if css_rules:
            st.markdown(f"<style>{''.join(css_rules)}</style>", unsafe_allow_html=True)

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
    
    @staticmethod
    def _render_statistic_card(class_name: str, stats: Dict, config: ModelConfig, img_area: float):
        """Универсальная отрисовка карточки статистики"""
        title, text_color = config.class_titles.get(
            class_name, 
            (class_name.replace("Bacillus", "").replace("Coccus", "").capitalize(), "#ffffff")
        )
        
        st.markdown(f"""
            <div style="
                background-color: rgba(255, 255, 255, 0.05); 
                border-radius: 4px; 
                border: 1.5px solid {text_color}; 
                height: 2rem;
                display: flex; 
                justify-content: center;
                align-items: center;
                font-size: 1.25rem;
                margin: 0;
                padding: 0;
                color: {text_color};">
                {title}
            </div>
            <div style="font-size: 0.9rem; margin-top: 0.5rem;">
                Count: {stats['count']}<br>
                Area (μm²): {stats['total_area_mkm']:.2f}<br> 
                Area (%): {((stats['total_area_mkm'] / img_area) * 100):.2f}  
            </div>
        """, unsafe_allow_html=True)
        st.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)