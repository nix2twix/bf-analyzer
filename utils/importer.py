# === LIBRARIES GENERAL ===
import zipfile
import json
import xml.etree.ElementTree as ET
import numpy as np
from utils.rle import rle_decode

def loadMasksFromZip(uploaded_zip, imgWidth, imgHeight, model_config):

    if uploaded_zip is None:
        return None

    try:
        with zipfile.ZipFile(uploaded_zip) as z:
            archive_type = getArchiveType(z)

            if archive_type == "xml":
                xml_name = next(
                    name for name in z.namelist()
                    if name.lower().endswith(".xml")
                )

                with z.open(xml_name) as f:
                    return loadXMLAnnotations(
                        f.read(),
                        imgWidth,
                        imgHeight,
                        model_config
                    )

            elif archive_type == "backup":
                with z.open("annotations.json") as f:
                    return loadBackupAnnotations(
                        f.read(),
                        imgWidth,
                        imgHeight,
                        model_config
                    )
            else:
                raise ValueError("Unknown archive type.")

    except zipfile.BadZipFile:
        return None

def getArchiveType(zf):
    names = set(zf.namelist())

    if "annotations.json" in names:
        return "backup"

    if any(name.endswith(".xml") for name in names):
        return "xml"

    return None


def decodeAnnotationObjects(
    objects,
    predictedObjects,
    model_config,
    source: str = "xml"
):
    """
    Restore masks from CVAT XML annotations or CVAT backup JSON.
    """

    if model_config.cvat_labels:
        class_map = {
            cvat_name.lower(): class_name
            for class_name, cvat_name in model_config.cvat_labels.items()
        }
    else:
        class_map = {
            cls.lower(): cls
            for cls in model_config.class_names
        }

    for obj in objects:

        if source == "xml":

            label = obj.attrib["label"]
            rle = obj.attrib["rle"]

            left = int(float(obj.attrib["left"]))
            top = int(float(obj.attrib["top"]))
            width = int(float(obj.attrib["width"]))
            height = int(float(obj.attrib["height"]))

        else:

            label = obj["label"]
            rle = obj["points"]

            left = int(float(rle[-4]))
            top = int(float(rle[-3]))

            right = int(float(rle[-2]))
            bottom = int(float(rle[-1]))

            width = right - left + 1
            height = bottom - top + 1

        class_name = class_map.get(label.lower())

        if class_name is None:
            continue

        crop = rle_decode(rle, (height, width))

        mask = predictedObjects[class_name]

        object_id = mask.max() + 1

        region = mask[top:top + height, left:left + width]

        region[crop == 1] = object_id

        mask[top:top + height, left:left + width] = region

def loadXMLAnnotations(xml_bytes, imgWidth, imgHeight, model_config):

    predictedObjects = {
        cls: np.zeros((imgHeight, imgWidth), dtype=np.int32)
        for cls in model_config.class_names
    }

    root = ET.fromstring(xml_bytes)

    objects = root.findall("image/mask")

    decodeAnnotationObjects(
        objects,
        predictedObjects,
        model_config,
        source="xml"
    )

    return predictedObjects

def loadBackupAnnotations(json_bytes, imgWidth, imgHeight, model_config):

    predictedObjects = {
        cls: np.zeros((imgHeight, imgWidth), dtype=np.int32)
        for cls in model_config.class_names
    }

    annotations = json.loads(json_bytes)

    objects = annotations[0]["shapes"]

    decodeAnnotationObjects(
        objects,
        predictedObjects,
        model_config,
        source="backup"
    )

    return predictedObjects

def checkAnnotationSize(uploaded_zip, imgWidth, imgHeight, minimumDiff = 0):
    """Проверяет соответствие размеров аннотации и изображения"""
    try:
        with zipfile.ZipFile(uploaded_zip) as z:
            archive_type = getArchiveType(z)
            
            if archive_type == "xml":
                xml_name = next(
                    name for name in z.namelist()
                    if name.lower().endswith(".xml")
                )
                with z.open(xml_name) as f:
                    return checkXMLSize(f.read(), imgWidth, imgHeight, minimumDiff)
                    
            elif archive_type == "backup":
                manifest_path = "data/manifest.jsonl"
                if manifest_path in z.namelist():
                    with z.open(manifest_path) as f:
                        return checkBackupSizeFromManifest(f.read(), imgWidth, imgHeight, minimumDiff)
                else:
                    return False, 0, 0
            else:
                return False, 0, 0
                
    except Exception as e:
        print(f"Ошибка при проверке размеров: {e}")
        return False, 0, 0

def checkXMLSize(xml_content, imgWidth, imgHeight, minimumDiff):
    """Проверяет размеры из XML"""
    import xml.etree.ElementTree as ET
    
    try:
        root = ET.fromstring(xml_content)
        image_element = root.find(".//image")
        if image_element is not None:
            width = int(image_element.attrib.get("width", 0))
            height = int(image_element.attrib.get("height", 0))
            if width > 0 and height > 0:
                is_valid = (np.abs(width-imgWidth) <= minimumDiff) and (np.abs(height-imgHeight) <= minimumDiff)
                return is_valid, width, height
            
        meta = root.find("meta")
        if meta is not None:
            size = meta.find(".//original_size")
            if size is not None:
                width = int(size.attrib.get("width", 0))
                height = int(size.attrib.get("height", 0))
                if width > 0 and height > 0:
                    is_valid = (np.abs(width-imgWidth) <= minimumDiff) and (np.abs(height-imgHeight) <= minimumDiff)
                    return is_valid, width, height
        
        images = root.findall(".//image")
        for img in images:
            width = int(img.attrib.get("width", 0))
            height = int(img.attrib.get("height", 0))
            if width > 0 and height > 0:
                is_valid = (np.abs(width-imgWidth) <= minimumDiff) and (np.abs(height-imgHeight) <= minimumDiff)
                return is_valid, width, height
        
        return False, 0, 0
        
    except Exception as e:
        print(f"Ошибка при парсинге XML: {e}")
        return False, 0, 0

def checkBackupSizeFromManifest(manifest_content, imgWidth, imgHeight, minimumDiff):
    """Проверяет размеры из manifest.jsonl"""
    import json
    
    try:
        lines = manifest_content.decode('utf-8').strip().split('\n')
        
        for line in lines:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if data.get("type") == "images":
                    continue
                elif "name" in data and "width" in data and "height" in data:
                    width = data.get("width", 0)
                    height = data.get("height", 0)
                    if width > 0 and height > 0:
                        is_valid = (np.abs(width-imgWidth) <= minimumDiff) and (np.abs(height-imgHeight) <= minimumDiff)
                        return is_valid, width, height
            except json.JSONDecodeError:
                continue
        
        return False, 0, 0
        
    except Exception as e:
        print(f"Ошибка при парсинге manifest.jsonl: {e}")
        return False, 0, 0

