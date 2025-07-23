# === LIBRARIES GENERAL ===
from cv2 import threshold
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# === PROJECT SCRIPTS ===
from processHandler import calculateStatistics, detectBiofilm, detectSingleBacteries
from processHandler import filtrationObjects, drawPicture, makeBacteriaInfo
from processingFunctions import cropLineBelow
from styles import loadStyles

def loadDefaultSession():
        # IMAGE    
        st.session_state.imageName = None
        st.session_state.uploadedImage = None
        
        st.session_state.predictedLabels = None
        st.session_state.filteredLabels = None
        
        st.session_state.statisticsInfo = None
                
        st.session_state.imgWidth = 0
        st.session_state.imgHeight = 0
        
        # FILTRATION SETTINGS
        st.session_state.singleBacteriesAreaRange = (0, 5000)
        st.session_state.singleBacteriesMinEcc = 0.85
        st.session_state.bfAreaRange = (0, 100)
        
        st.session_state.filtrationParams = {
                "minSingleArea": st.session_state.singleBacteriesAreaRange[0],
                "maxSingleArea": st.session_state.singleBacteriesAreaRange[1],
                "minSingleEcc": st.session_state.singleBacteriesMinEcc,
                "minBfArea": st.session_state.bfAreaRange[0],
                "maxBfArea": st.session_state.bfAreaRange[1]
        }
        
        # SESSION
        st.session_state.showNumbers = False
           

# === PAGE CONFIGURATION ===
st.set_page_config(
    page_title="Biofilm Analyzer",
    layout="wide"
)

loadStyles()

if "imageName" not in st.session_state:
    loadDefaultSession()
    
# === HEADER ===
with st.container():
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("## 🧪 Biofilm Analyzer")

    with col2:
        st.markdown("### ℹ️ Info")
        st.markdown("""
            This tool is designed for processing SEM images of biofilms.  
                Set the analysis parameters on the left, upload the image, and get the processing result.
        """)

st.markdown('<hr style="margin: 0.5rem 0;">', unsafe_allow_html=True)

# === INTERFACE ===
blockSettings, blockWorkspace, blockTools = st.columns([1, 2.8, 1])

# === Правая панель: Инструменты ===
with blockTools:
    st.markdown("### 🛠 Tools")

    # Загрузка изображения
    uploadedFile = st.file_uploader("Load image", type=["bmp", "png", "jpg"], key="uploader")
    
    if uploadedFile is None:
        loadDefaultSession()
        
    if (uploadedFile is not None) and (uploadedFile.name != st.session_state.imageName):
        loadDefaultSession()
        st.session_state.imageName = uploadedFile.name
        st.session_state.uploadedImage = Image.open(uploadedFile)       

    # --- Инструменты ---
    seg_button_clicked = st.button("🧪 Start segmentation",disabled = st.session_state.uploadedImage is None, use_container_width=True)
    #st.button("🔍 Zoom (see later)")
    #st.button("💾 Save results (see later)")
    st.session_state.showNumbers = st.toggle("Show bacteries numbers", disabled = True)
    
    if seg_button_clicked:
        with st.spinner("⏳ Image processing..."):
            
            tempCropedImage = st.session_state.uploadedImage
            tempCropedImage = cropLineBelow(tempCropedImage, 120)    
            
            st.session_state.imgWidth, st.session_state.imgHeight = tempCropedImage.size
            
            # PREDICTION CLASS: BIOFILM
            biofilmPredictions = detectBiofilm(
                np.asarray(tempCropedImage),
                tempCropedImage.size,
                threshold = 0.5
            )
            # PREDICTION CLASS: SINGLE
            singlePredictions = detectSingleBacteries(
                np.asarray(tempCropedImage),
                biofilmPredictions,
                cellposeParams = [0.4, 0.0]
            )
            
            st.session_state.predictedLabels = {
                "bf": biofilmPredictions,
                "single": singlePredictions
            }
            # посчитать статистики! и объекты
            st.session_state.statisticsInfo = calculateStatistics(st.session_state.predictedLabels, 
                                                    scale = 0.05)
            
            if (st.session_state.filteredLabels is not None):
                st.write("📤 Processing finished") 
                print("[INFO] Processed finished successfully!")

# === Левая панель: Settings ===
with blockSettings:
    st.markdown("### ⚙️ Filtration settings")
    # Cлайдеры
    singleBacteriesAreaRange = st.slider(
        "Single bacteria area (px)",
        min_value=0, 
        max_value=5000,
        value=st.session_state.singleBacteriesAreaRange,  
        key="singleBacteriesAreaRange",
        help="Area of single microorganism in pixels of image",
        disabled=st.session_state.predictedLabels is None
    )
     
    singleBacteriesMinEcc = st.slider(
        "Single bacteria eccentricity",
        min_value=0.0, 
        max_value=1.0,
        value=st.session_state.singleBacteriesMinEcc, 
        key="singleBacteriesMinEcc",
        help="An eccentricity equal to zero corresponds to a perfect circle, and equal to one corresponds to a ellipse",
        disabled=st.session_state.predictedLabels is None
        )
    
    bfAreaRange = st.slider(
        "Biofilm area (%)",
        min_value=0, 
        max_value=100,
        value=st.session_state.bfAreaRange,  
        key="bfAreaRange",
        help="Biofilm area range as a percentage of the image size",
        disabled=st.session_state.predictedLabels is None
    )
    
    st.session_state.filtrationParams = {
                "minSingleArea": st.session_state.singleBacteriesAreaRange[0],
                "maxSingleArea": st.session_state.singleBacteriesAreaRange[1],
                "minSingleEcc": st.session_state.singleBacteriesMinEcc,
                "minBfArea": st.session_state.bfAreaRange[0],
                "maxBfArea": st.session_state.bfAreaRange[1]
    }   

with blockSettings:
    # Фильтрация результатов обработки
    if (st.session_state.predictedLabels is not None):
        
        st.session_state.filteredLabels = filtrationObjects(st.session_state.uploadedImage,
                                        st.session_state.predictedLabels,
                                        st.session_state.filtrationParams)
        # фильтрация посчитанных статистик из st.session_state.statisticsInfo
        # filtered labels должна вернуть номера отфильтрованных оставшихся меток, по ним считать статистику

        #типо сумма масок по номерам
        biofilmArea = resultInfo["biofilm_mkm_area"]
        bacteriesCount = resultInfo["bacteria_count"]
        bacteriesArea = resultInfo["bacteries_mkm_area"]
        imgArea = st.session_state.imgWidth * st.session_state.imgHeight * (0.05**2) #scale square
        
        st.markdown("### 📊 Statistics")
        st.markdown(f"Single bacterias count: {bacteriesCount}")
        st.markdown(f"Single bacterias area: {(bacteriesArea):.1f} μm<sup>2</sup> ({(100*bacteriesArea / imgArea):.1f}%)", unsafe_allow_html=True)
        st.markdown(f"Biofilm area: {biofilmArea:.1f} μm<sup>2</sup> ({(100*biofilmArea / imgArea):.1f}%)", unsafe_allow_html=True)
        
# === Центральная панель: Workflow ===
with blockWorkspace:
    st.markdown("### 🔬 Workflow")
    
    if st.session_state.uploadedImage is None:        
        st.info("SEM-image wasn't uploaded")
    else:
        if st.session_state.filteredLabels is None:
            st.image(image = st.session_state.uploadedImage, 
                     caption = f"Loaded SEM-image {st.session_state.imageName}", 
                     use_container_width=True)
            
        else:
            cropedOrigImage = cropLineBelow(st.session_state.uploadedImage, 120)    
            processedImage = drawPicture(cropedOrigImage, 
                st.session_state.filteredLabels)

            if st.session_state.showNumbers:
                singleInfo = makeBacteriaInfo(st.session_state.filteredLabels)   
                draw = ImageDraw.Draw(processedImage)

                for bacteria in singleInfo:
                    x = bacteria["centroidCoords"][0]
                    y = bacteria["centroidCoords"][1]
                    draw.text((y, x),
                              text = str(bacteria["maskNum"]),
                              fill = (255, 255, 255),
                              font = ImageFont.load_default(size = 30)
                              )
            st.image(image = processedImage, 
                         caption = f"Processing {st.session_state.imageName} result", 
                         use_container_width=True)
