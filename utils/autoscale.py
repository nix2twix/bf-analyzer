import numpy as np
import re
import streamlit as st


@st.cache_resource
def get_ocr_reader():
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return reader

def normalizeScaleBar(c_fullImage, lowerBound = None, bright_threshold=240):
    if lowerBound is None:
        return c_fullImage  
    
    img_normalized = c_fullImage.copy()
    scale_region = img_normalized[lowerBound:, :]

    mask_bright = scale_region > bright_threshold
    scale_region[:] = 0 #обнуляем темное
    scale_region[mask_bright] = c_fullImage[lowerBound:, :][mask_bright]

    return img_normalized


def findBorder(c_fullImage, thr = 0.35):
    if len(c_fullImage.shape) == 3:
        row_sum = np.sum(c_fullImage, axis=(1, 2), dtype=np.int64)
    else:
        row_sum = np.sum(c_fullImage, axis=1, dtype=np.int64)

    for i in range(10, len(row_sum) - 1):
        if np.abs(row_sum[i] - row_sum[i + 1]) >= row_sum[i] * thr:
            return i + 1
    
    return None

def scaleLength(c_fullImage, start_y):
    _, width = c_fullImage.shape
    first_white_index = None
    last_white_index = None

    for x in range(1, width):
        if (c_fullImage[start_y, x] >= 250 ) and (c_fullImage[start_y, x-1] <= 230):
            if first_white_index is None:
                first_white_index = x
            last_white_index = x

    if first_white_index is not None and last_white_index is not None:
        return last_white_index - first_white_index, first_white_index

    return None, None


def findText(c_footnoteImage):
    reader = get_ocr_reader() 
    result = reader.readtext(c_footnoteImage, detail=0, blocklist='SOo')
    return ' '.join(result).lower()  

def increase(c_text):
    try:
        matchesIncrease = re.findall(r'[x][0-9]*\.?[0-9]+[k]', c_text)[0]
        _increase = float(matchesIncrease[1:-1])
    except Exception:
        _increase = None

    return _increase

def scale(c_text):
    try:
        matchesScale = re.findall(r"[0-9]*\.?[0-9]+[nup]m", c_text)[0]
        if matchesScale[-2] == 'n':
            _scale = float(matchesScale[:-2]) / 1000
        elif matchesScale[-2] == 'u' or matchesScale[-2] == 'p':
            _scale = float(matchesScale[:-2])
    except Exception:
        _scale = None
        matchesScale = None

    return _scale, matchesScale

@st.cache_data(show_spinner = False)
def estimateScale(c_image):
    
    lowerBound = findBorder(c_image)
    c_image = normalizeScaleBar(c_image, lowerBound)
    
    if (lowerBound is not None):      
        text = findText(c_image[lowerBound:, :])
        scaleVal, scaleText = scale(text)
            
        scaleLengthVal, startPixelScale = scaleLength(c_image, lowerBound)
        print(f"Start scale pixel: {startPixelScale}")
        print(f"Scale below length: {scaleLengthVal} px")      
        
        if (scaleVal is not None) and (scaleLengthVal is not None):
            print(f"mkm / pixel: {scaleVal} / {scaleLengthVal} = {scaleVal / scaleLengthVal}")
            print(f"pixel / mkm: {scaleLengthVal} / {scaleVal} = {scaleLengthVal / scaleVal}")
            return scaleVal / scaleLengthVal, [lowerBound, startPixelScale, scaleLengthVal, scaleText, scaleVal]
        
    return None, None

