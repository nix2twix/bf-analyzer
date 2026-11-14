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

            item = self.process_one(
                name, data, model_config,
                patch_progress_cb=patch_cb,
            )
            results.append(item)

        return results

    def process_one(
    self,
    unique_name: str,
    image_bytes: bytes,
    model_config,
    patch_progress_cb: Optional[Callable[[int, int], None]] = None,
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

            # В batch фильтрацию НЕ применяем: пользователь не может её настроить,
            # поэтому показываем все объекты, которые нашла модель.
            predicted = postprocessed_masks
            item.filtered_masks = predicted  # имя поля оставляем — CVAT/Excel ниже читают его

            objects_info, _ = prepare_objectInfo(predicted, model_config)
            item.objects_info = objects_info

            filtered_info = prepare_filteredObjectInfo(predicted)
            item.filtered_info = filtered_info

            item.stats = calculateStatistics(filtered_info, scale=item.scale)
            item.img_area_um2 = item.width * item.height * (item.scale ** 2)

        except Exception as e:
            item.error = f"{type(e).__name__}: {e}"

        return item