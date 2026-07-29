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


def get_objects_hash(objects):
    """Создает хэш для кэширования на основе содержимого масок"""
    if objects is None:
        return None
    
    hash_parts = []
    for class_name, mask in sorted(objects.items()):
        if mask is not None:
            # Уникальные ID объектов в маске
            unique_ids = np.unique(mask)
            obj_count = len(unique_ids[unique_ids > 0])
            # Сумма всех пикселей для проверки изменений
            pixel_sum = np.sum(mask)
            hash_parts.append(f"{class_name}:{obj_count}:{pixel_sum}")
    
    return "|".join(hash_parts)


def drawPicture(uploaded_file, objects, classColors, isShowIntermediate=True):
    """
    Оптимизированная отрисовка масок на изображении.
    Использует векторизованные операции numpy для ускорения.
    """
    origImg = np.array(uploaded_file)
    
    if origImg is None:
        raise ValueError("Could not decode image")
    
    # Единоразовая конвертация в RGBA
    if len(origImg.shape) == 2:
        origImg = cv2.cvtColor(origImg, cv2.COLOR_GRAY2RGBA)
    elif origImg.shape[2] == 3:
        origImg = cv2.cvtColor(origImg, cv2.COLOR_RGB2RGBA)
    elif origImg.shape[2] == 4:
        pass  # Уже RGBA
    else:
        origImg = cv2.cvtColor(origImg, cv2.COLOR_BGR2RGBA)
    
    # Создаем оверлей одним проходом
    base_rgb = origImg[..., :3].copy()
    overlay_rgb = np.zeros_like(base_rgb)
    overlay_alpha = np.zeros(base_rgb.shape[:2], dtype=np.uint8)
    
    for className, mask in objects.items():
        if className not in classColors:
            continue
        
        color = classColors[className]
        # Для intermediate класса с опцией скрытия - используем цвет biofilm
        if className == "intermediate" and not isShowIntermediate:
            if "biofilm" in classColors:
                color = classColors["biofilm"]
            else:
                continue
        
        # Векторизованное применение маски
        mask_bool = mask != 0
        if np.any(mask_bool):
            overlay_rgb[mask_bool] = color[:3]
            overlay_alpha[mask_bool] = color[3]
    
    # Быстрое смешивание через numpy (векторизовано)
    covered = overlay_alpha != 0
    if np.any(covered):
        alpha = overlay_alpha[covered, None].astype(np.uint16)
        base = base_rgb[covered].astype(np.uint16)
        color = overlay_rgb[covered].astype(np.uint16)
        base_rgb[covered] = ((base * (255 - alpha) + color * alpha) // 255).astype(np.uint8)
    
    # Конвертируем обратно в uint8 и добавляем альфа-канал
    return Image.fromarray(base_rgb, mode="RGB")


def resizeForDisplay(image: Image.Image, max_width: int = 2560) -> Image.Image:
    """Уменьшает изображение для ускорения отрисовки в браузере"""
    if image.width > max_width:
        ratio = max_width / image.width
        new_height = int(image.height * ratio)
        return image.resize((max_width, new_height), Image.Resampling.LANCZOS)
    return image
