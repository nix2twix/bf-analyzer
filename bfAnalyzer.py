# === LIBRARIES GENERAL ===
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# === PROJECT SCRIPTS ===
from processHandler import (
    segmentationImage,
    prepareObjectInfo,
    filtrationObjects,
    drawPicture,
    calculateStatistics,
    bytesToImg,
    imgToBytes
)
from autoScale import (
    estimateScale,
    findBorder
    )
from resultsHandler import makeCVATbackupPolygons
from resultsHandler import loadMasksFromZip
from styles import loadStyles
from datetime import datetime

# === DEFAULT SESSION ===
def loadDefaultSession():
    # IMAGE    
    st.session_state.imageName = None
    st.session_state.uploadedImage = None
    st.session_state.predictedLabels = None
    st.session_state.filteredLabels = None
    
    st.session_state.bacteriaInfo = None
    st.session_state.singleMinArea = 0
    st.session_state.singleMaxArea = 10000    

    st.session_state.biofilmRegions = None
    st.session_state.biofilmMinArea = 0
    st.session_state.biofilmMaxArea = 10000
    
    st.session_state.imgWidth = 0
    st.session_state.imgHeight = 0
    st.session_state.imgScale = 0.05
    st.session_state.scaleMkm = 50
    st.session_state.infoLineHeight = 120
    
    # FILTRATION SETTINGS
    st.session_state.singleBacteriesMinEcc = 0.85
    
    st.session_state.singleBacteriesAreaRange = (st.session_state.singleMinArea, st.session_state.singleMaxArea)
    st.session_state.bfAreaRange = (st.session_state.biofilmMinArea, st.session_state.biofilmMaxArea)
    
    st.session_state.filtrationParams = {
        "minSingleArea": st.session_state.singleMinArea,
        "maxSingleArea": st.session_state.singleMaxArea,
        "minSingleEcc": st.session_state.singleBacteriesMinEcc,
        "minBfArea": st.session_state.biofilmMinArea,
        "maxBfArea": st.session_state.biofilmMaxArea
    }
    
    # SESSION
    st.session_state.showNumbers = False
    st.session_state.polygonsCVAT = ""


# === PAGE CONFIGURATION ===
st.set_page_config(
    page_title="Biofilm Analyzer",
    layout="wide"
    )

loadStyles()

if "imageName" not in st.session_state:
    loadDefaultSession()

# === HEADER ===
st.header("🧪 Biofilm Analyzer")
st.markdown("This tool is designed for processing SEM images of biofilms")
st.markdown('<hr style="margin: 0.5rem 0;">', unsafe_allow_html=True)

# === INTERFACE ===
blockWorkspace, space, blockTools = st.columns([2.5, 0.05, 1.2])

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
            
        tempArrImg = np.array(st.session_state.uploadedImage, dtype='uint8')

        autoScale, scaleData = estimateScale(tempArrImg)
        autoScaleText = scaleData[4] if scaleData else None
        infoLineHeight = st.session_state.imgHeight - scaleData[0] if scaleData else None
        
        st.session_state.imgScale = autoScale
        st.session_state.infoLineHeight = infoLineHeight
        st.session_state.scaleMkm = autoScaleText

        manualScaleEnabled = st.toggle(
            "Enter the scale manually", value=False
        )
        if manualScaleEnabled:
            manualScaleValue = st.number_input(
                "Enter the scale (μm/px):",
                min_value=0.001,
                max_value=1000.0,
                value=float(autoScale) if autoScale else 1.0,
                step=0.01,
                format="%.3f"
            )
            st.session_state.imgScale = manualScaleValue
        else:
            st.session_state.imgScale = autoScale
            
        segButtonClicked = st.button(
            "🧪 Start segmentation", 
            disabled=st.session_state.uploadedImage is None,
            use_container_width=True)     
        
        if segButtonClicked:
            with st.spinner("⏳ Image processing..."):
           
                st.session_state.predictedLabels = segmentationImage(
                    uploadedFile,
                    st.session_state.infoLineHeight,
                    st.session_state.imgWidth, 
                    st.session_state.imgHeight,
                    st.session_state.imageName
                )

        if st.session_state.filteredLabels is not None:
            st.session_state.polygonsCVAT = makeCVATbackupPolygons(
                st.session_state.uploadedImage,
                st.session_state.imageName,
                st.session_state.filteredLabels,
                st.session_state.imgWidth, 
                st.session_state.imgHeight
            )
            
        st.markdown('<div style="margin: 0rem;"></div>', unsafe_allow_html=True)
        st.markdown("#### 📁 Masks uploader", unsafe_allow_html=True)
        with st.expander("⬇️ Upload mask from CVAT", expanded=False):
            uploaded_zip = st.file_uploader("Choose ZIP file", type=['zip'], 
                                            disabled=st.session_state.uploadedImage is None)
            if uploaded_zip is not None:
                labeledMasks = loadMasksFromZip(
                    uploaded_zip, 
                    st.session_state.imgWidth, 
                    st.session_state.imgHeight)
                if labeledMasks is not None:
                    st.session_state.predictedLabels = labeledMasks
                        
            if st.session_state.predictedLabels is not None:
                st.session_state.objectsInfo = prepareObjectInfo(st.session_state.predictedLabels)
                
                st.session_state.bacteriaInfo = st.session_state.objectsInfo[0]
                st.session_state.singleMinArea = st.session_state.objectsInfo[1]
                st.session_state.singleMaxArea = st.session_state.objectsInfo[2]
                
                st.session_state.biofilmRegions = st.session_state.objectsInfo[3]
                st.session_state.biofilmMinArea = st.session_state.objectsInfo[4]
                st.session_state.biofilmMaxArea = st.session_state.objectsInfo[5]
                
                st.session_state.singleBacteriesAreaRange = (st.session_state.singleMinArea, st.session_state.singleMaxArea)
                st.session_state.bfAreaRange = (st.session_state.biofilmMinArea, st.session_state.biofilmMaxArea)

                st.session_state.filteredLabels = None     

    # === Вкладка Filtration ===
    with tabsTools[1]:
        st.markdown("### ⚙️ Filtration")

        singleBacteriesAreaRange = st.slider(
            "Single bacteria area (px)",
            min_value=st.session_state.singleMinArea, 
            max_value=st.session_state.singleMaxArea,
            value=st.session_state.singleBacteriesAreaRange,  
            key="singleBacteriesAreaRange",
            disabled=st.session_state.predictedLabels is None
        )
        
        singleBacteriesMinEcc = st.slider(
            "Single bacteria eccentricity",
            min_value=0.0, 
            max_value=1.0,
            value=st.session_state.singleBacteriesMinEcc, 
            key="singleBacteriesMinEcc",
            disabled=st.session_state.predictedLabels is None
        )
            
        bfAreaRange = st.slider(
            "Biofilm area (px)",
            min_value=st.session_state.biofilmMinArea,
            max_value=st.session_state.biofilmMaxArea,
            value=st.session_state.bfAreaRange,
            step=1,
            key="bfAreaRange",
            disabled=st.session_state.predictedLabels is None
        )
        
        st.session_state.filtrationParams = {
            "minSingleArea": singleBacteriesAreaRange[0],
            "maxSingleArea": singleBacteriesAreaRange[1],
            "minSingleEcc": singleBacteriesMinEcc,
            "minBfArea": bfAreaRange[0],
            "maxBfArea": bfAreaRange[1]
        }

        if st.session_state.predictedLabels is not None:
            st.session_state.filteredLabels = filtrationObjects(
                st.session_state.bacteriaInfo,
                st.session_state.predictedLabels["bf"],
                st.session_state.filtrationParams
            )
            
    with tabsTools[0]:              
        st.download_button(
            label="📥 Export mask for CVAT",
            data=st.session_state.polygonsCVAT,
            file_name=f"backup-{datetime.now().strftime('%d-%m-%Y-%H-%M')}.zip",
            mime="application/zip",
            disabled=st.session_state.filteredLabels is None,
            use_container_width=True
        )
    # === Раздел Filtration ===
    with tabsTools[1]: 
        st.markdown("### 📊 Statistics")
        if st.session_state.filteredLabels is not None:
            resultInfo = calculateStatistics(
                st.session_state.filteredLabels, 
                scale=st.session_state.imgScale)
            
            biofilmArea = resultInfo["biofilm_mkm_area"]
            bacteriesCount = resultInfo["bacteria_count"]
            bacteriesArea = resultInfo["bacteries_mkm_area"]
            imgArea = st.session_state.imgWidth * (st.session_state.imgHeight - st.session_state.infoLineHeight)
            imgArea = imgArea * (st.session_state.imgScale**2)
                      
            st.markdown("""
                <div style="
                    background-color: rgba(36, 179, 83, 0.3); 
                    border-radius: 4px; 
                    height: 2rem;
                    display: flex; 
                    justify-content: center;
                    align-items: center;
                    font-size: 1.25rem;
                    margin: 0;
                    padding: 0;">
                    Biofilms
                </div>
            """, unsafe_allow_html=True)

            st.markdown(f"Biofilm area: {biofilmArea:.1f} μm<sup>2</sup> ({(100*biofilmArea / imgArea):.1f}%)", 
                        unsafe_allow_html=True)
            st.markdown('<div style="margin-bottom: 1.5rem;"></div>', unsafe_allow_html=True)
            st.markdown("""
                <div style="
                    background-color: rgba(184, 61, 245, 0.3); 
                    border-radius: 4px; 
                    height: 2rem;
                    display: flex; 
                    justify-content: center;
                    align-items: center;
                    font-size: 1.25rem;
                    margin: 0;
                    padding: 0;">
                    Microorganisms
                </div>
            """, unsafe_allow_html=True)

            st.markdown(f"Single bacterias count: {bacteriesCount}")
            st.markdown(f"Single bacterias area: {(bacteriesArea):.1f} μm<sup>2</sup> ({(100*bacteriesArea / imgArea):.1f}%)", unsafe_allow_html=True)
        else:
            st.info("No statistics was calculated.")

# === Левая панель: Workflow ===
with blockWorkspace:
    workFlowCol, scaleCol = st.columns([2.4, 1.6])
    with workFlowCol:
        st.markdown("### 🔬 Workflow")  
    with scaleCol:
        scaleInfoText = f"""
        ### 🔎 Scale: {st.session_state.scaleMkm:.1f} μ/px  
        """
        st.markdown(scaleInfoText)
    if st.session_state.uploadedImage is not None:
        if st.session_state.filteredLabels is None:
            st.image(
                st.session_state.uploadedImage, 
                caption=f"Loaded SEM-image {st.session_state.imageName}", 
                use_container_width=True)
        else:
            processedImage = drawPicture(uploadedFile, st.session_state.filteredLabels)

            

            if st.session_state.showNumbers:
                draw = ImageDraw.Draw(processedImage)
                for b in st.session_state.bacteriaInfo:
                    if b["maskNum"] in np.unique(st.session_state.filteredLabels["single"]):
                        y, x = b["centroidCoords"]
                        draw.text((x, y), text=str(b["maskNum"]), fill=(255, 255, 255), font=ImageFont.load_default(size=30))

            st.image(processedImage, caption=f"Processing {st.session_state.imageName} result", use_container_width=True)
    else:
        st.info("SEM-image was not uploaded.")
    
    helpTab =  st.expander("❓ Help", expanded=False)
    with helpTab:
        st.markdown("If you have any problems, you can try to clear cash memory:")
        st.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)
        if st.button("♻ Clear cache"):
            st.cache_data.clear()

