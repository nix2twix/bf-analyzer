# === LIBRARIES GENERAL ===
import numpy as np
from PIL import Image

MINIMAGESIZE = 512

def cropLineBelow(imgPIL, countPx=120):
    width, height = imgPIL.size
    if (width >= MINIMAGESIZE + countPx) and (height >= MINIMAGESIZE + countPx):
        cropped_img = imgPIL.crop((0, 0, width, height - countPx))
        print(f"CROPED {countPx} px from below")
        return cropped_img
    else:
        return imgPIL

def makePatches(imgPIL, patch_size=(512, 512), stride=(128, 128)):   
    img_np = np.array(imgPIL)
    img_height, img_width = img_np.shape[:2]

    patch_h, patch_w = patch_size
    stride_y, stride_x = stride

    patch_id = 0
    patch_list = []
    coords = []

    x_coords = list(range(0, img_width - patch_w + 1, stride_x))
    x_coords.append(img_width - patch_w)
    
    y_coords = list(range(0, img_height - patch_h + 1, stride_y))
    y_coords.append(img_height - patch_h)

    for y in y_coords:
        for x in x_coords:
            patch = imgPIL.crop((x, y, x + patch_w, y + patch_h))
            patch_list.append(patch)
            coords.append((x, y, patch_id))
            patch_id += 1
            
    return patch_list, coords

def pad_to_divisible(img, divisor=32):
    w, h = img.size

    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor

    new_h = h + pad_h
    new_w = w + pad_w

    padded_img = Image.new("RGB", (new_w, new_h))
    padded_img.paste(img, (0, 0))

    return padded_img, (h, w)
