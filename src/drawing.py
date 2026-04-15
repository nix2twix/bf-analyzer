# === LIBRARIES GENERAL ===
import cv2  
from PIL import Image
import numpy as np

def checkSize(img: Image.Image, min_size: int = 512):
    orig_w, orig_h = img.size
    w = max(orig_w, min_size)
    h = max(orig_h, min_size)
    w_pad = ((w + 31) // 32) * 32
    h_pad = ((h + 31) // 32) * 32

    return w_pad, h_pad

def correctSize(img: Image.Image, w_pad, h_pad):
    padded_img = Image.new("L", (w_pad, h_pad))
    padded_img.paste(img, (0, 0))

    return padded_img

#@st.cache_data(show_spinner=False, ttl=6000, max_entries=10)
def drawPicture(uploaded_file, objects, classColors, isShowIntermediate=True):
    origImg = np.array(uploaded_file)
    
    if origImg is None:
        raise ValueError("Could not decode image")
    
    if len(origImg.shape) == 2:  # Grayscale
        origImg = cv2.cvtColor(origImg, cv2.COLOR_GRAY2RGBA)
    elif origImg.shape[2] == 3:  # BGR
        origImg = cv2.cvtColor(origImg, cv2.COLOR_BGR2RGBA)
    
    overlay = np.zeros_like(origImg)
    for className, mask in objects.items():
        if className not in classColors:
            continue
        color = classColors["biofilm"] if (className == "intermediate" and not isShowIntermediate) else classColors[className]
        overlay[mask > 0] = color

    result = origImg.copy()
    alpha = overlay[..., 3] / 255.0
    result[..., :3] = (1 - alpha[..., np.newaxis]) * origImg[..., :3] + alpha[..., np.newaxis] * overlay[..., :3]
    
    return Image.fromarray(result)