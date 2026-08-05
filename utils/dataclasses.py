from dataclasses import dataclass
from typing import Optional

from PIL import Image


@dataclass
class AnnotationData:

    image: Optional[Image.Image]
    image_name: str
    width: int
    height: int
    masks: dict
    source: str
    metadata: dict