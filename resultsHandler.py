import io
import zipfile
import skimage.measure as measure
from datetime import datetime
import json
import os
from PIL import Image
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import re

def rle_decode(rle_str, shape):
    rle_numbers = [int(num) for num in re.findall(r'\d+', rle_str)]
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    index = 0
    i = 0
    n = len(rle_numbers)

    while i < n:
        start = rle_numbers[i]
        i += 1
        length = 0
        if i < n:
            length = rle_numbers[i]
            i += 1
        # Сдвигаемся
        index += start
        # Если длина > 0, заливаем участок
        if length > 0:
            if index + length > img.size:
                raise ValueError(f"RLE segment exceeds mask size: index {index} length {length} size {img.size}")
            img[index:index+length] = 1
            index += length
    return img.reshape(shape)

def rle_encode(mask: np.ndarray) -> str:
    # Flatten по строкам (row-major)
    pixels = mask.flatten(order="C")
    # Список длин чередующихся серий (0,1,0,1,...)
    counts = []
    current_pixel = pixels[0]
    count = 1
    for p in pixels[1:]:
        if p == current_pixel:
            count += 1
        else:
            counts.append(count)
            count = 1
            current_pixel = p
    counts.append(count)
    return ", ".join(map(str, counts))

def loadMasksFromZip(uploaded_zip, imgWidth, imgHeight):
    if uploaded_zip is None:
        return None

    try:
        with zipfile.ZipFile(uploaded_zip) as z:
            xml_filename = next((name for name in z.namelist() if name.lower().endswith('.xml')), None)
            if xml_filename is None:
                print("No XML file found in the uploaded ZIP archive.")
                return None
            
            with z.open(xml_filename) as xml_file:
                xml_bytes = xml_file.read()
                masks = updateMaskCVAT(xml_bytes, imgWidth, imgHeight)
                
                singleBinary = masks.get("single", np.zeros((imgHeight, imgWidth), dtype=np.uint8))
                labeledSingle = measure.label(singleBinary)
                
                labeledBf = masks.get("bf", np.zeros((imgHeight, imgWidth), dtype=np.uint8))
                labeledMasks = {
                    "single": labeledSingle,
                    "bf": labeledBf
                }
                return labeledMasks

    except zipfile.BadZipFile:
        print("Uploaded file is not a valid ZIP archive.")
        return None


def updateMaskCVAT(xml_path, width, height):
    xml_io = io.BytesIO(xml_path)
    tree = ET.parse(xml_io)
    root = tree.getroot()

    bf_mask = np.zeros((height, width), dtype=np.uint8)
    single_mask = np.zeros((height, width), dtype=np.uint8)

    for image in root.findall('image'):
        for mask in image.findall('mask'):
            label = mask.attrib['label']
            rle = mask.attrib['rle']
            left = int(mask.attrib.get('left', 0))
            top = int(mask.attrib.get('top', 0))
            w = int(mask.attrib.get('width', 0))
            h = int(mask.attrib.get('height', 0))
            z_order = int(mask.attrib.get('z_order', 0))

            decoded_mask = rle_decode(rle, (h, w))

            if top + h > height or left + w > width:
                raise ValueError(f"Mask with position ({left},{top}) and size ({w}x{h}) exceeds image bounds ({width}x{height})")

            if label == "Biofilm":
                region = bf_mask[top:top+h, left:left+w]
                bf_mask[top:top+h, left:left+w] = np.maximum(region, decoded_mask * (z_order + 1))
            elif label == "Microorganisms":
                region = single_mask[top:top+h, left:left+w]
                single_mask[top:top+h, left:left+w] = np.maximum(region, decoded_mask * (z_order + 1))

    return {"bf": bf_mask, "single": single_mask}

def get_contours_as_points(mask: np.ndarray, width: int, height: int):
    bin_mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours_points = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue

        contour = contour.reshape(-1, 2)
        points = contour.flatten().astype(float).tolist()

        if len(points) < 6:
            continue
        if len(points) % 2 != 0:
            continue
        if any(p < 0 for p in points):
            continue
        if any(p > max(width, height) for p in points):
            continue

        # Замыкаем контур
        if points[0] != points[-2] or points[1] != points[-1]:
            points += points[:2]

        all_contours_points.append(points)
    return all_contours_points

def makeCVATbackupPolygons(
    image: Image.Image,
    original_filename: str,
    predicted_labels: dict,
    width, height
):
    base_name, ext = os.path.splitext(original_filename)
    archive_name = f"backup-{datetime.now().strftime('%d-%m-%Y-%H-%M')}.zip"

    # manifest.jsonl
    manifest = [
        {"version": "1.1"},
        {"type": "images"},
        {"name": base_name, "extension": ext, "width": width, "height": height, "meta": {"related_images": []}},
    ]
    manifest_content = "\n".join(json.dumps(line, ensure_ascii=False) for line in manifest)

    # task.json
    task = {
        "name": archive_name,
        "bug_tracker": "",
        "status": "annotation",
        "subset": "Train",
        "labels": [
            {"name": "Microorganisms", "color": "#b83df5", "attributes": [], "type": "any", "sublabels": []},
            {"name": "Biofilm", "color": "#24b353", "attributes": [], "type": "any", "sublabels": []},
            {"name": "Defect", "color": "#ff0000", "attributes": [], "type": "any", "sublabels": []},
            {"name": "intermediate-stage", "color": "#ddff33", "attributes": [], "type": "any", "sublabels": []},
        ],
        "version": "1.0",
        "data": {
            "chunk_size": 15,
            "image_quality": 100,
            "start_frame": 0,
            "stop_frame": 0,
            "storage_method": "cache",
            "storage": "local",
            "sorting_method": "lexicographical",
            "chunk_type": "imageset",
            "deleted_frames": [],
        },
        "jobs": [{"status": "annotation", "type": "annotation", "start_frame": 0, "stop_frame": 0, "frames": []}],
    }
    task_content = json.dumps(task, ensure_ascii=False, indent=2)

    # annotations.json — один кадр
    frame_obj = {"version": 0, "tags": [], "shapes": [], "tracks": []}

    # Microorganisms
    if "single" in predicted_labels and predicted_labels["single"] is not None:
        single_mask = np.asarray(predicted_labels["single"])
        unique_ids = np.unique(single_mask)
        unique_ids = unique_ids[unique_ids != 0]  # убираем фон
        for obj_id in unique_ids:
            obj_mask = (single_mask == obj_id).astype(np.uint8)
            contours_points_list = get_contours_as_points(obj_mask, width, height)
            for points in contours_points_list:
                shape = {
                    "type": "polygon",
                    "occluded": False,
                    "z_order": 0,
                    "rotation": 0.0,
                    "points": points,
                    "frame": 0,
                    "group": 0,
                    "source": "manual",
                    "attributes": [],
                    "label": "Microorganisms",
                }
                frame_obj["shapes"].append(shape)

    # Biofilm
    if "bf" in predicted_labels and predicted_labels["bf"] is not None:
        bf_mask = np.asarray(predicted_labels["bf"])
        if bf_mask.dtype != np.uint8:
            bf_mask = (bf_mask > 0.5).astype(np.uint8)
        contours_points_list = get_contours_as_points(bf_mask, width, height)
        for points in contours_points_list:
            shape = {
                "type": "polygon",
                "occluded": False,
                "z_order": 0,
                "rotation": 0.0,
                "points": points,
                "frame": 0,
                "group": 0,
                "source": "manual",
                "attributes": [],
                "label": "Biofilm",
            }
            frame_obj["shapes"].append(shape)

    annotations = [frame_obj]
    annotations_content = json.dumps(annotations, ensure_ascii=False, separators=(',', ':'))

    # Создаем ZIP в памяти
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Картинка
        img_bytes = io.BytesIO()
        image.save(img_bytes, format=image.format or ext.replace(".", "").upper())
        img_bytes.seek(0)
        zf.writestr(f"data/{original_filename}", img_bytes.read())

        # manifest.jsonl
        zf.writestr("data/manifest.jsonl", manifest_content)

        # task.json и annotations.json
        zf.writestr("task.json", task_content)
        zf.writestr("annotations.json", annotations_content)

    zip_buffer.seek(0)
    return zip_buffer

def makeXMLforCVAT(image_name, image_width, image_height, labels, masks, crop_bottom=120):
    """
    labels: [{"name": str, "color": str}]
    masks: [{"label": str, "mask": np.ndarray, "z_order": int}]
    crop_bottom: сколько пикселей снизу было обрезано (для восстановления)
    """
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    meta = ET.SubElement(root, "meta")
    job = ET.SubElement(meta, "job")
    ET.SubElement(job, "id").text = "1"
    ET.SubElement(job, "size").text = "1"
    ET.SubElement(job, "mode").text = "annotation"
    ET.SubElement(job, "overlap").text = "0"
    ET.SubElement(job, "created").text = datetime.utcnow().isoformat()
    ET.SubElement(job, "updated").text = datetime.utcnow().isoformat()
    labels_el = ET.SubElement(job, "labels")
    for lbl in labels:
        lbl_el = ET.SubElement(labels_el, "label")
        ET.SubElement(lbl_el, "name").text = lbl["name"]
        ET.SubElement(lbl_el, "color").text = lbl["color"]
        ET.SubElement(lbl_el, "type").text = "any"
        ET.SubElement(lbl_el, "attributes").text = " "

    img_el = ET.SubElement(root, "image", {
        "id": "0",
        "name": image_name,
        "width": str(image_width),
        "height": str(image_height)
    })

    for m in masks:
        # 1. Восстанавливаем высоту, если маска обрезана
        mask_array = m["mask"].astype(np.uint8)
        if mask_array.shape[0] != image_height:
            pad_height = image_height - mask_array.shape[0]
            if pad_height < 0:
                raise ValueError("Маска больше, чем изображение!")
            mask_array = np.pad(mask_array, ((0, pad_height), (0, 0)), mode='constant', constant_values=0)

        # 2. Разделяем на объекты
        num_labels, labels_im = cv2.connectedComponents(mask_array)
        for obj_id in range(1, num_labels):  # 0 — фон
            obj_mask = (labels_im == obj_id).astype(np.uint8)

            # 3. Bounding box
            ys, xs = np.where(obj_mask > 0)
            if len(xs) == 0:
                continue
            left, top = xs.min(), ys.min()
            width = xs.max() - xs.min() + 1
            height = ys.max() - ys.min() + 1

            cropped_mask = obj_mask[top:top+height, left:left+width]
            rle = rle_encode(cropped_mask)

            ET.SubElement(img_el, "mask", {
                "label": m["label"],
                "source": "manual",
                "occluded": "0",
                "rle": rle,
                "left": str(left),
                "top": str(top),
                "width": str(width),
                "height": str(height),
                "z_order": str(m.get("z_order", 0))
            }).text = " "

    return ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")

# xml_bytes = makeXMLforCVAT(
#     st.session_state.imageName,
#     st.session_state.imgWidth,
#     st.session_state.imgHeight,
#     [
#         {"name": "Microorganisms", "color": "#b83df5"},
#         {"name": "Biofilm", "color": "#24b353"},
#         {"name": "Defect", "color": "#ff0000"},
#         {"name": "intermediate-stage", "color": "#ddff33"}
#     ],
#     [
#         {"label": "Microorganisms", "mask": st.session_state.filteredLabels["single"]},
#         {"label": "Biofilm", "mask": st.session_state.filteredLabels["bf"]}
#     ]
# ).encode("utf-8")

# st.download_button(
#     label="📥 Download mask in XML",
#     data=xml_bytes,
#     file_name="annotations.xml",
#     mime="application/xml"
# )