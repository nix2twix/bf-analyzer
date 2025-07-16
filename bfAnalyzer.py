# === LIBRARIES GENERAL ===
import streamlit as st
import json

# === PROJECT SCRIPTS ===
from processHandler import startProcessing

# === PAGE CONFIGURATION ===
st.set_page_config(
    page_title="Biofilm Analyzer",
    layout="wide"
)

st.markdown("""
    <style>
        .stApp {
            padding: 1rem 1rem 1rem 1rem !important;
            margin: 0 !important;
        }
        
        .stMarkdown h1 {
            margin-top: 0.5rem !important;
            padding-top: 0 !important;
        }
        
        .stMarkdown p {
            margin-bottom:0;
        }
        
        div.stContainer > div:first-child {
            padding-top: 0.5rem !important;
        }
        
        .stSlider, .stSelectbox, .stButton {
            margin-top: 0.2rem !important;
            margin-bottom: 0.2rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# === SESSION STATE ===
if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None
if "is_image_uploaded" not in st.session_state:
    st.session_state.is_image_uploaded = False
if "image_name" not in st.session_state:
    st.session_state.image_name = ""
if "processed_image_bytes" not in st.session_state:
    st.session_state.processed_image = None
if "is_image_processed" not in st.session_state:
    st.session_state.is_image_processed = False   
if "area_range" not in st.session_state:
    st.session_state.area_range = (50, 1155)
if "min_ecc" not in st.session_state:
    st.session_state.min_ecc = 0.85


# ============== Блок 1: Заголовок и описание ==============
with st.container():
    col1, col2 = st.columns([1, 4])

    with col1:
        st.markdown("## 🧪 Biofilm Analyzer")

    with col2:
        st.markdown("### ℹ️ Info")
        st.markdown("""
This tool is designed for processing SEM images of biofilms. The supported image format is .bmp, .png, and .jpg. 
Set the analysis parameters on the left, upload the image, and get the processing result.
        """)

st.markdown('<hr style="margin: 0.5rem 0;">', unsafe_allow_html=True)

# ============== Блок 2: Интерфейс ==============
col_settings, col_workspace, col_tools = st.columns([1, 2.5, 1])

# === Левая панель: Settings ===
with col_settings:
    st.markdown("### ⚙️ Settings")

    # Cлайдеры
    area_range = st.slider(
        "Area range (px)",
        min_value=0, 
        max_value=2000,
        value=st.session_state.area_range,  
        key="area_slider"
    )

    min_ecc = st.slider(
        "Minimum eccentricity",
        min_value=0.0, 
        max_value=1.0,
        value=st.session_state.min_ecc, 
        key="ecc_slider"
    )

    st.session_state.area_range = area_range
    st.session_state.min_ecc = min_ecc
    
    # Статистики
    if st.session_state.get("biofilm_mkm_area") is not None:
        st.markdown("### 📊 Statistics")
        st.markdown(f"Biofilm area: {st.session_state.biofilm_mkm_area:g} (μm<sup>2</sup>)", unsafe_allow_html=True)
        st.markdown(f"Single bacterias count: {st.session_state.bacteria_count}")
        st.markdown(f"Single bacterias area: {st.session_state.bacteries_area:g} (μm<sup>2</sup>)", unsafe_allow_html=True)
        

# === Центральная панель: Workflow ===
with col_workspace:
    st.markdown("### 🔬 Workflow")
    
    if st.session_state.is_image_processed and st.session_state.processed_image_bytes != None:
        st.image(image = st.session_state.processed_image_bytes, 
                 caption = f"Processing {st.session_state.image_name} result", 
                 use_container_width=True)
        
    elif st.session_state.is_image_uploaded and st.session_state.image_bytes != None:
        st.image(image = st.session_state.image_bytes, 
                 caption = f"Loaded SEM-image {st.session_state.image_name}", 
                 use_container_width=True)
    else:
        st.info("SEM-image wasn't uploaded")

# === Правая панель: Инструменты ===
with col_tools:
    st.markdown("### 🛠 Tools")

    # Загрузка изображения
    uploaded_file = st.file_uploader("Load image", type=["bmp", "png", "jpg"], key="uploader")

    if uploaded_file is not None and uploaded_file.name != st.session_state.image_name:
        st.session_state.image_bytes = uploaded_file.read()
        st.session_state.processed_image_bytes = None
        st.session_state.is_image_processed = False
        st.session_state.is_image_uploaded = True
        st.session_state.image_name = uploaded_file.name
        st.rerun()
        
    # Сброс изображения
    elif uploaded_file is None and st.session_state.is_image_uploaded:
        st.session_state.image_bytes = None
        st.session_state.processed_image_bytes = None
        st.session_state.is_image_processed = False
        st.session_state.is_image_uploaded = False
        st.session_state.image_name = ""
        st.rerun()
        
    # --- Инструменты ---
    seg_button_clicked = st.button("🧪 Start segmentation", disabled=st.session_state.image_bytes is None)
    #st.button("🔍 Zoom (see later)")
    #st.button("💾 Save results (see later)")

    if seg_button_clicked:
        with st.spinner("⏳ Image processing..."):

            params = {
                "min_area": st.session_state.area_range[0],
                "max_area": st.session_state.area_range[1],
                "min_ecc": st.session_state.min_ecc,
            }
            
            result = startProcessing(uploaded_file,
                                     st.session_state.image_name,
                                     params["min_area"], 
                                     params["max_area"], 
                                     params["min_ecc"])
            
            st.write("📤 Processing finished")
            print("[INFO] Processed finished successfully!")
            
            if result != None:
                st.session_state.processed_image_bytes = result[0]
                st.session_state.is_image_processed = True
                st.rerun()
                
    