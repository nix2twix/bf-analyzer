# === LIBRARIES GENERAL ===
import streamlit as st
import numpy as np
from PIL import Image

# === PROJECT SCRIPTS ===
from processHandler import (
    segmentationImage,
    getPredictedObjects,
    prepareObjectInfo,
    filtrationObjects,
    drawPicture,
    calculateStatistics
)
from autoScale import estimateScale

from resultsHandler import makeCVATbackupPolygons
from resultsHandler import loadMasksFromZip
from styles import loadStyles
from datetime import datetime

# === DEFAULT SESSION ===
def loadDefaultSession():
    # IMAGE    
    st.session_state.imageName = None
    st.session_state.uploadedImage = None
        
    st.session_state.imgWidth = 0
    st.session_state.imgHeight = 0
    st.session_state.imgScale = None
    st.session_state.scaleMkm = None
    st.session_state.infoLineHeight = None
    
    # PREDS
    st.session_state.predictedObjects = None
    st.session_state.filteredObjects = None
    st.session_state.objectsInfo = None
    st.session_state.areaStats = {
      "biofilm": {"min_area": 0, "max_area": 10000},
      "intermediate": {"min_area": 0, "max_area": 10000},
      "single": {"min_area": 0, "max_area": 5000}
    }
     
    # FILTRATION SETTING
    st.session_state.singleBacteriesAreaRange = (0, 5000)
    st.session_state.biofilmAreaRange = (0, 10000)
    st.session_state.singleBacteriesEccRange = (0.0, 1.0)
    st.session_state.intermediateAreaRange = (0, 10000)
    
    st.session_state.filtrationParams = {
        "minSingleArea": st.session_state.singleBacteriesAreaRange[0],
        "maxSingleArea": st.session_state.singleBacteriesAreaRange[1],
        "minSingleEcc": st.session_state.singleBacteriesEccRange[0],
        "maxSingleEcc": st.session_state.singleBacteriesEccRange[1],
        "minBfArea": st.session_state.biofilmAreaRange[0],
        "maxBfArea": st.session_state.biofilmAreaRange[1],
        "minIntermediateArea": st.session_state.intermediateAreaRange[0],
        "maxIntermediateArea": st.session_state.intermediateAreaRange[1]
    }
    
    # VIZUALIZATION
    st.session_state.classColors = {  
                "biofilm": (36, 179, 83, 178), 
                "intermediate": (221, 255, 51, 178),  # 221, 255, 51
                "single": (184, 61, 245, 178) 
    }
    st.session_state.isShowIntermediate = False
    
    # EXPORT
    st.session_state.polygonsCVAT = ""

# === PROCESSING CALLBACKS ===
def updateFilteredObjects():
    params = {
        "minSingleArea": st.session_state.singleBacteriesAreaRange[0],
        "maxSingleArea": st.session_state.singleBacteriesAreaRange[1],
        "minSingleEcc": st.session_state.singleBacteriesEccRange[0],
        "maxSingleEcc": st.session_state.singleBacteriesEccRange[1],
        "minBfArea": st.session_state.biofilmAreaRange[0],
        "maxBfArea": st.session_state.biofilmAreaRange[1]
    }

    if st.session_state.isShowIntermediate:
        params["minIntermediateArea"] = st.session_state.intermediateAreaRange[0]
        params["maxIntermediateArea"] = st.session_state.intermediateAreaRange[1]

    st.session_state.filtrationParams = params

    if st.session_state.predictedObjects is not None:
        st.session_state.filteredObjects = filtrationObjects(
            st.session_state.objectsInfo,
            st.session_state.predictedObjects,
            st.session_state.filtrationParams,
            showIntermedAsBf = not st.session_state.isShowIntermediate   
        )
        

# === PAGE CONFIGURATION ===
st.set_page_config(page_title="Biofilm Analyzer", layout="wide")
loadStyles()

if "imageName" not in st.session_state:
    loadDefaultSession()

# === HEADER ===
st.header("🧪 Biofilm Analyzer")
st.markdown("This tool is designed for processing SEM images of biofilms")
st.markdown('<hr style="margin: 0.5rem 0;">', unsafe_allow_html=True)

# === INTERFACE ===
blockWorkspace, space, blockTools = st.columns([2.2, 0.05, 1.4])

with blockTools:
    tabsTools = st.tabs(["🛠 Tools", "⚙️ Filtration"])
    # === Вкладка Tools ===
    with tabsTools[0]:      
        with st.expander("⬇️ Upload SEM-image", expanded=True):
            uploadedFile = st.file_uploader("Choose SEM-image file", type=["bmp", "png", "jpg"], key="uploader")
            
        if uploadedFile is None:
            loadDefaultSession()
            
        if (uploadedFile is not None) and (uploadedFile.name != st.session_state.imageName):
            loadDefaultSession()
            st.session_state.imageName = uploadedFile.name
            st.session_state.uploadedImage = Image.open(uploadedFile)
            st.session_state.imgWidth, st.session_state.imgHeight = st.session_state.uploadedImage.size

        manualScaleEnabled = st.toggle(
            "Enter the scale manually", 
            value=False,
            disabled=st.session_state.uploadedImage is None
        )
        
        if uploadedFile is not None:  
            tempArrImg = np.array(st.session_state.uploadedImage, dtype='uint8')
            autoScale, scaleData = estimateScale(tempArrImg)
            autoScaleText = scaleData[4] if scaleData else 1
            infoLineHeight = st.session_state.imgHeight - scaleData[0] if scaleData else 0
        
            st.session_state.imgScale = autoScale if autoScale else 1.0
            st.session_state.infoLineHeight = infoLineHeight
            st.session_state.scaleMkm = autoScaleText
        
            if manualScaleEnabled:
                manualScaleValue = st.number_input(
                    "Enter the scale μm:",
                    min_value=0.001,
                    max_value=1000.0,
                    value=float(autoScale) if autoScale else 1.0,
                    step=0.1,
                    format="%.3f"
                )
                st.session_state.imgScale = manualScaleValue / 1000
                st.session_state.scaleMkm = manualScaleValue
            
        segButtonClicked = st.button(
            "🧪 Start segmentation", 
            disabled=st.session_state.uploadedImage is None,
            use_container_width=True)     
        
        if segButtonClicked:
            with st.spinner("⏳ Image processing..."):
           
                predictedLabels = segmentationImage(
                    uploadedFile,
                    st.session_state.infoLineHeight,
                    st.session_state.imgWidth, 
                    st.session_state.imgHeight,
                    st.session_state.imageName
                )
                
                st.session_state.predictedObjects = getPredictedObjects(
                    predictedLabels
                )
            
        st.markdown('<div style="margin: 0rem;"></div>', unsafe_allow_html=True)
        st.markdown("#### 📁 Masks uploader", unsafe_allow_html=True)
        with st.expander("⬇️ Upload mask from CVAT", expanded=False):
            uploaded_zip = st.file_uploader("Choose ZIP file", type=['zip'], 
                                            disabled=st.session_state.uploadedImage is None)
            if uploaded_zip is not None:
                st.session_state.predictedObjects = loadMasksFromZip(
                    uploaded_zip, 
                    st.session_state.imgWidth, 
                    st.session_state.imgHeight)
                st.session_state.objectsInfo = None

        if st.session_state.predictedObjects is not None and st.session_state.objectsInfo is None:
            st.session_state.objectsInfo, st.session_state.areaStats = prepareObjectInfo(st.session_state.predictedObjects)
            st.session_state.singleBacteriesAreaRange = (
                st.session_state.areaStats["single"]["min_area"],
                st.session_state.areaStats["single"]["max_area"]
            )
            st.session_state.biofilmAreaRange = (
                st.session_state.areaStats["biofilm"]["min_area"],
                st.session_state.areaStats["biofilm"]["max_area"]
            )
            st.session_state.intermediateAreaRange = (
                st.session_state.areaStats["intermediate"]["min_area"],
                st.session_state.areaStats["intermediate"]["max_area"]
            )
            st.session_state.filteredObjects = None     
   
    with tabsTools[0]:
        if st.session_state.filteredObjects is not None:
            st.session_state.polygonsCVAT = makeCVATbackupPolygons(
                st.session_state.uploadedImage,
                st.session_state.imageName,
                st.session_state.filteredObjects,
                st.session_state.imgWidth, 
                st.session_state.imgHeight
            )
        st.download_button(
            label="📥 Export mask for CVAT",
            data=st.session_state.polygonsCVAT,
            file_name=f"backup-{datetime.now().strftime('%d-%m-%Y-%H-%M')}.zip",
            mime="application/zip",
            disabled=st.session_state.filteredObjects is None,
            use_container_width=True
        )
    # === Раздел Filtration ===
    with tabsTools[1]: 
        filtrationCol, gap, statCol = st.columns([1.5, 0.01, 1.6])
        with filtrationCol:
            st.markdown("### ⚙️ Filtration")
            st.session_state.isShowIntermediate = st.toggle(
                "Show intermediate", 
                value = st.session_state.isShowIntermediate)  
        
            singleBacteriesAreaRange = st.slider(
                "Single bacteria area (px)",
                min_value=st.session_state.areaStats["single"]["min_area"], 
                max_value=st.session_state.areaStats["single"]["max_area"],
                value=st.session_state.singleBacteriesAreaRange,  
                key="singleBacteriesAreaRange",
                on_change=updateFilteredObjects
            )

            singleBacteriesEccRange = st.slider(
                "Single bacteria eccentricity",
                min_value=0.0, 
                max_value=1.0,
                step=0.1,
                value=st.session_state.singleBacteriesEccRange, 
                key="singleBacteriesEccRange",
                on_change=updateFilteredObjects
            )

            biofilmAreaRange = st.slider(
                "Biofilm area (px)",
                min_value=st.session_state.areaStats["biofilm"]["min_area"],
                max_value=st.session_state.areaStats["biofilm"]["max_area"],
                value=st.session_state.biofilmAreaRange,
                step=1,
                key="biofilmAreaRange",
                on_change=updateFilteredObjects
            )
        
            if st.session_state.isShowIntermediate:
                    intermediateAreaRange = st.slider(
                        "Intermediate area (px)",
                        min_value=st.session_state.areaStats["intermediate"]["min_area"],
                        max_value=st.session_state.areaStats["intermediate"]["max_area"],
                        value=st.session_state.intermediateAreaRange,
                        step=1,
                        key="intermediateAreaRange",
                        on_change=updateFilteredObjects
                    )
            if st.session_state.predictedObjects is not None:
                updateFilteredObjects() 
        with statCol:
            st.markdown("### 📊 Statistics")
            
            if st.session_state.filteredObjects is not None:
                resultInfo = calculateStatistics(
                    st.session_state.objectsInfo, 
                    scale=st.session_state.imgScale)
            
                imgArea = st.session_state.imgWidth * (st.session_state.imgHeight - st.session_state.infoLineHeight)
                imgArea = imgArea * (st.session_state.imgScale**2)

                classTitles = {
                    "biofilm": ("Biofilms", "#24b353"),          
                    "intermediate": ("Intermediate", "#ddff33"), 
                    "single": ("Single", "#b83df5")             
                } 
                if not st.session_state.isShowIntermediate and "intermediate" in resultInfo:
                        resultInfo["biofilm"]["count"] += resultInfo["intermediate"]["count"]
                        resultInfo["biofilm"]["total_area_px"] += resultInfo["intermediate"]["total_area_px"]
                        resultInfo["biofilm"]["total_area_mkm"] += resultInfo["intermediate"]["total_area_mkm"]
                        resultInfo.pop("intermediate", None)

                
                for className, stats in resultInfo.items():
                    title, textColor = classTitles[className]

                    st.markdown(f"""
                        <div style="
                            background-color: rgba(255, 255, 255, 0.05); 
                            border-radius: 4px; 
                            border: 1.5px solid {textColor}; 
                            height: 2rem;
                            display: flex; 
                            justify-content: center;
                            align-items: center;
                            font-size: 1.25rem;
                            margin: 0;
                            padding: 0;
                            color: {textColor};">
                            {title}
                        </div>
                        <div style="font-size: 0.9rem; margin-top: 0.5rem;">
                            Count: {stats['count']}<br>
                            Area: {stats['total_area_mkm']:.2f} μm²
                        </div>
                    """, unsafe_allow_html=True)
                    st.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)
            else:
                st.info("No statistics was calculated.")

# === Левая панель: Workflow ===
with blockWorkspace:
    workFlowCol, scaleCol = st.columns([2.4, 1.6])
    with workFlowCol:
        st.markdown("### 🔬 Workflow")  
    with scaleCol:
        if st.session_state.uploadedImage is not None:
            st.markdown(f"### 🔎 Scale: {st.session_state.scaleMkm:.1f} μm")
        else:
            st.markdown("")
        
    if st.session_state.uploadedImage is not None:
        if st.session_state.filteredObjects is None:
            st.image(
                st.session_state.uploadedImage, 
                caption=f"Loaded SEM-image {st.session_state.imageName}", 
                use_container_width=True)
        else:
            processedImage = drawPicture(
                uploadedFile,
                st.session_state.filteredObjects,
                st.session_state.classColors,
                st.session_state.isShowIntermediate
            )
            st.image(processedImage, caption=f"Processing {st.session_state.imageName} result", use_container_width=True)
    else:
        st.info("SEM-image was not uploaded.")
    
    helpTab =  st.expander("❓ Help", expanded=False)
    with helpTab:
        st.markdown("If you have any problems, you can try to clear cash memory:")
        st.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)
        if st.button("♻ Clear cache"):
            st.cache_data.clear()

