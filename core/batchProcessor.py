# === LIBRARIES GENERAL ===
import io
import os
import numpy as np
import streamlit as st
from dataclasses import dataclass, field
from PIL import Image
from typing import List, Optional, Dict, Tuple, Callable

# === PROJECT SCRIPTS ===
from segmentation.inference import segmentationImage
from segmentation.objects import prepare_objectInfo, prepare_filteredObjectInfo
from segmentation.statistics import calculateStatistics
from utils.drawing import checkSize, correctSize


@dataclass
class BatchItem:
    """Результат обработки одной картинки в пачке."""
    unique_name: str
    original_bytes: bytes                
    width: int = 0
    height: int = 0
    info_line_height: int = 0
    scale: float = 1.0     
    scale_source: str = "auto"    # "auto", "manual", "first_in_batch", "none"
    scale_ok: bool = False
    exported: bool = False
    filtered_masks: Dict[str, np.ndarray] = field(default_factory=dict)
    objects_info: Dict = field(default_factory=dict)
    filtered_info: Dict = field(default_factory=dict)
    stats: Dict = field(default_factory=dict)        
    img_area_um2: float = 0.0

    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and (bool(self.filtered_masks) or self.exported)
    def safe_stem(self) -> str:
        """Безопасное имя без расширения (для Excel/CVAT)."""
        stem, _ = os.path.splitext(self.unique_name)
        return stem


class BatchProcessor:
    def __init__(self, state_manager):
        self.state = state_manager

    def process_all(
        self,
        images: List[Tuple[str, bytes]],
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
        patch_progress_cb_factory: Optional[Callable[[str], Callable[[int, int], None]]] = None,
        scale_mode="first"
    ) -> List[BatchItem]:
        model_config = self.state.getConfig()
        if model_config is None:
            raise RuntimeError("Model config is not set")

        results: List[BatchItem] = []
        total = len(images)

        for i, (name, data) in enumerate(images, start=1):
            if progress_cb is not None:
                progress_cb(i, total, name)

            patch_cb = patch_progress_cb_factory(name) if patch_progress_cb_factory else None

            item = self.process_one(name, data, model_config,
                            patch_progress_cb=patch_cb,
                            scale_mode=scale_mode,
                            fallback_scale=first_scale if scale_mode == "first" else None,
                            fallback_ok=first_scale_ok if scale_mode == "first" else False)
            if scale_mode == "first" and i == 1:
                first_scale = item.scale
                first_scale_ok = item.scale_ok
            results.append(item)

        return results

    def process_one(
        self,
        unique_name: str,
        image_bytes: bytes,
        model_config,
        patch_progress_cb: Optional[Callable[[int, int], None]] = None,
        scale_mode="first",
        fallback_scale=None, fallback_ok=False
    ) -> BatchItem:
        item = BatchItem(unique_name=unique_name, original_bytes=image_bytes)

        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("L")
            w_pad, h_pad = checkSize(pil_img, min_size=512)
            if (w_pad, h_pad) != pil_img.size:
                pil_img = correctSize(pil_img, w_pad, h_pad)

            item.width, item.height = pil_img.size

            item.info_line_height = 0
            item.scale = 1.0

            arr = np.asarray(pil_img, dtype=np.uint8)
            image_bytes_padded = arr.tobytes()

            postprocessed_masks, raw_masks, _ = segmentationImage(
                image_bytes=image_bytes_padded,
                INFLINEPX=0,
                width=item.width,
                height=item.height,
                imgName=unique_name,
                model_config=model_config,
                threshold=0.1,
                show_progress=False,
                use_cache=False,
                patch_progress_cb=patch_progress_cb,
            )

            predicted = _drop_small_objects(postprocessed_masks, min_area=200)
            item.filtered_masks = predicted
            objects_info, _ = prepare_objectInfo(predicted, model_config)
            item.objects_info = objects_info

            filtered_info = prepare_filteredObjectInfo(predicted)
            item.filtered_info = filtered_info

            item.stats = calculateStatistics(filtered_info, scale=item.scale)
            # --- Масштаб ---
            if scale_mode == "none":
                item.scale = 1.0
                item.scale_ok = False
            elif scale_mode == "first" and fallback_scale is not None:
                item.scale = fallback_scale
                item.scale_ok = fallback_ok
            else:
                # "each" или "first" для первой картинки
                from utils.autoscale import estimateScale
                arr = np.asarray(pil_img, dtype=np.uint8)
                auto_scale, scale_data = estimateScale(arr)
                if auto_scale is not None:
                    item.scale = float(auto_scale)
                    item.scale_ok = True
                else:
                    item.scale = 1.0
                    item.scale_ok = False

            item.img_area_um2 = item.width * item.height * (item.scale ** 2)

        except Exception as e:
            item.error = f"{type(e).__name__}: {e}"

        return item

def _drop_small_objects(labeled_masks: dict, min_area: int = 200) -> dict:
    """
    Убирает объекты площадью < min_area, пересобирая метки.
    Дешёвая операция: один проход np.unique + маска.
    """
    out = {}
    for class_name, labeled in labeled_masks.items():
        if labeled is None or "background" in class_name or class_name == "bg":
            out[class_name] = labeled
            continue

        ids, counts = np.unique(labeled, return_counts=True)
        keep_ids = ids[(ids != 0) & (counts >= min_area)]

        if len(keep_ids) == 0:
            out[class_name] = np.zeros_like(labeled, dtype=labeled.dtype)
            continue

        new_labeled = np.zeros_like(labeled, dtype=labeled.dtype)
        for new_id, old_id in enumerate(keep_ids, start=1):
            new_labeled[labeled == old_id] = new_id
        out[class_name] = new_labeled

    return out