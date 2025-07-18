# === LIBRARIES GENERAL ===
import streamlit as st
from PIL import Image

# === PROJECT SCRIPTS ===
from processHandler import startProcessing, startFiltration
from processingFunctions import cropLineBelow
from styles import loadStyles

# === PAGE CONFIGURATION ===
st.set_page_config(
    page_title="Biofilm Analyzer",
    layout="wide"
)
loadStyles()

# === SESSION STATE ===
if "imageName" not in st.session_state:
    st.session_state.imageName = ""
if "singleBacteriesAreaRange" not in st.session_state:
    st.session_state.singleBacteriesAreaRange = (50, 3000)
if "singleBacteriesMinEcc" not in st.session_state:
    st.session_state.singleBacteriesMinEcc = 0.85
if "bfAreaRange" not in st.session_state:
    st.session_state.bfAreaRange = (0, 100)
if "uploadedFile" not in st.session_state:
    st.session_state.uploadedFile = None
if "imgWidth" not in st.session_state:
    st.session_state.imgWidth = 0
if "imgHeight" not in st.session_state:
    st.session_state.imgHeight = 0
if "processedImageBytes" not in st.session_state:
    st.session_state.processedImageBytes = None
if "predictedLabels" not in st.session_state:
    st.session_state.predictedLabels = None
if "resultInfo" not in st.session_state:
    st.session_state.resultInfo = None
    
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
blockSettings, blockWorkspace, blockTools = st.columns([1, 2.8, 1])

# === Левая панель: Settings ===
with blockSettings:
    st.markdown("### ⚙️ Filtration sttings")
    # Cлайдеры
    singleBacteriesAreaRange = st.slider(
        "Single bacteria area range (px)",
        min_value=0, 
        max_value=5000,
        value=st.session_state.singleBacteriesAreaRange,  
        key="singleBacteriesArea_slider",
        help="Area of single microorganism"
    )

    singleBacteriesMinEcc = st.slider(
        "Single bacteria minimum eccentricity",
        min_value=0.0, 
        max_value=1.0,
        value=st.session_state.singleBacteriesMinEcc, 
        key="singleBacteriesEcc_slider",
        help="An eccentricity equal to zero corresponds to a perfect circle, and equal to 1 corresponds to a parabola"
    )
    
    bfAreaRange = st.slider(
            "Biofilm area range (%)",
            min_value=0, 
            max_value=100,
            value=st.session_state.bfAreaRange,  
            key="bfArea_slider",
            help="Biofilm area range as a percentage of the image size"
        )
    
    # Фильтрация результатов обработки
    if (st.session_state.processedImageBytes != None):
        
        width = st.session_state.imgWidth
        height = st.session_state.imgHeight
        imgSize = width * height    
        
        filteredResult = startFiltration(st.session_state.processedImageBytes,
                                        st.session_state.predictedLabels,
                                        singleBacteriesAreaRange[0], 
                                        singleBacteriesAreaRange[1], 
                                        singleBacteriesMinEcc,
                                        bfAreaRange[0],
                                        bfAreaRange[1],
                                        imgSize,
                                        0.05)
        
        st.session_state.processedImageBytes = filteredResult[0]
        st.session_state.resultsInfo = filteredResult[1]
        biofilmArea = filteredResult[1]["biofilm_mkm_area"]
        bacteriesCount = filteredResult[1]["bacteria_count"]
        bacteriesArea = filteredResult[1]["bacteries_mkm_area"]
        st.session_state.singlePredictions = filteredResult[2]["single"]
        st.session_state.bfPredictions = filteredResult[2]["bf"]
        
        st.markdown("### 📊 Statistics")
        st.markdown(f"Biofilm area: {biofilmArea:g} (μm<sup>2</sup>, {(biofilmArea / (width * height + 1e-6)):g}%)", unsafe_allow_html=True)
        st.markdown(f"Single bacterias count: {bacteriesCount}")
        st.markdown(f"Single bacterias area: {(bacteriesArea):g} (μm<sup>2</sup>, {(bacteriesArea / (width * height + 1e-6)):g}%)", unsafe_allow_html=True)
 

# === Центральная панель: Workflow ===
with blockWorkspace:
    st.markdown("### 🔬 Workflow")
    
    if st.session_state.processedImageBytes != None:
        st.image(image = st.session_state.processedImageBytes, 
                 caption = f"Processing {st.session_state.imageName} result", 
                 use_container_width=True)
        
    elif st.session_state.uploadedFile != None:
        st.image(image = st.session_state.imageBytes, 
                 caption = f"Loaded SEM-image {st.session_state.imageName}", 
                 use_container_width=True)
    else:
        st.info("SEM-image wasn't uploaded")

# === Правая панель: Инструменты ===
with blockTools:
    st.markdown("### 🛠 Tools")

    # Загрузка изображения
    uploadedFile = st.file_uploader("Load image", type=["bmp", "png", "jpg"], key="uploader")
    
    if uploadedFile is not None and uploadedFile.name != st.session_state.imageName:
        st.session_state.uploadedFile = uploadedFile
        st.session_state.processedImageBytes = None
        st.session_state.imageName = uploadedFile.name
        
        image = Image.open(uploadedFile)
        image = cropLineBelow(image, 128)
        st.session_state.imgWidth, st.session_state.imgHeight = image.size
        st.rerun()
        
    # Сброс изображения
    elif uploadedFile is None:
        st.session_state.uploadedFile = None
        st.session_state.processedImageBytes = None
        st.session_state.imageName = ""
        st.session_state.imgWidth, st.session_state.imgHeight = 0, 0
        st.rerun()
        
    # --- Инструменты ---
    seg_button_clicked = st.button("🧪 Start segmentation", disabled=st.session_state.uploadedFile is None)
    #st.button("🔍 Zoom (see later)")
    #st.button("💾 Save results (see later)")

    if seg_button_clicked:
        with st.spinner("⏳ Image processing..."):
            params = {
                "minSingleArea": st.session_state.singleBacteriesAreaRange[0],
                "maxSingleArea": st.session_state.singleBacteriesAreaRange[1],
                "minSingleEcc": st.session_state.singleBacteriesMinEcc,
                "minBfArea": st.session_state.bfAreaRange[0],
                "maxBfArea": st.session_state.bfAreaRange[1]
            }
            
            result = startProcessing(uploadedFile,
                                     st.session_state.imageName)
                        
            filteredResult = startFiltration(result,
                                     params["minSingleArea"], 
                                     params["maxSingleArea"], 
                                     params["minSingleEcc"],
                                     params["minBfArea"],
                                     params["maxBfArea"],
                                     width * height,
                                     0.05)     
            
            st.session_state.processedImageBytes = filteredResult[0]
            st.session_state.biofilm_mkm_area = filteredResult[1]["biofilm_mkm_area"]
            st.session_state.bacteriaCount = filteredResult[1]["bacteria_count"]
            st.session_state.bacteries_mkm_area = filteredResult[1]["bacteries_mkm_area"]
            st.session_state.singlePredictions = filteredResult[2]["single"]
            st.session_state.bfPredictions = filteredResult[2]["bf"]
            
            st.write("📤 Processing finished")
            print("[INFO] Processed finished successfully!")
            st.rerun()
                
    