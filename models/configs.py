# === LIBRARIES GENERAL ===
import os

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Пути к моделям
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

@dataclass
class ModelConfig:
    name: str
    display_name: str
    class_names: List[str]
    class_colors: Dict[str, Tuple[int, int, int, int]]
    checkpoint_path: str
    num_classes: int

    class_labels: Dict[str, int]
    filtration_params: Dict[str, Tuple[float, float]]
   
    # Для статистики
    class_titles: Dict[str, Tuple[str, str]]

    # postsegmentation
    postprocess_params: Optional[Dict[str, Dict]] = None
    
    # Для CVAT 
    cvat_labels: Optional[Dict[str, str]] = None  # class_name -> cvat_label_name
    cvat_label_colors: Optional[Dict[str, str]] = None  # cvat_label_name -> color
    
    # Параметры модели
    encoder_name: str = "resnet34"
    encoder_weights: str = "imagenet"
    activation: Optional[str] = None
    class_weights: Optional[Dict[str, float]] = None
    
    def adjust_area_stats(self, areaStats):
        """Специфичная для модели обработка статистик"""
        return areaStats  # по умолчанию без изменений

BACILLUS_CONFIG = ModelConfig(
    name="Bacillus",
    display_name="🦠 Bacillus",
    class_names=["bg", "biofilm", "intermediate", "single"],
    num_classes=4,
    class_colors={
        "biofilm": (36, 179, 83, 90),
        "intermediate": (221, 255, 51, 90),
        "single": (184, 61, 245, 90)
    },
    checkpoint_path=os.path.join(MODELS_DIR, "bacillus.pth"),
    class_labels={"bg": 0, "biofilm": 1, "intermediate": 2, "single": 3},
    postprocess_params={
        "single": {
            "area": (150, 20000),
            "ecc": (0.35, 1.0)
        },
        "biofilm": {
            "area": (3000, 4587520)
        },
        "intermediate": {
            "area": (1250, 458752)
        }
    },
    filtration_params={
        "single_area": (100, 5000),
        "single_ecc": (0.0, 1.0),
        "biofilm_area": (100, 10000),
        "intermediate_area": (100, 10000)
    },
    class_titles={
        "single": ("Single", "#b83df5"),
        "biofilm": ("Biofilms", "#24b353"),
        "intermediate": ("Intermediate", "#ddff33")
    },

    cvat_labels={
        "single": "Microorganisms",
        "biofilm": "Biofilm",
        "intermediate": "intermediate-stage"
    },
    cvat_label_colors={
        "Microorganisms": "#b83df5",
        "Biofilm": "#24b353",
        "intermediate-stage": "#ddff33",
        "Defect": "#ff0000"  
    }
)

COCCUS_CONFIG = ModelConfig(
    name="Coccus",
    display_name="🧫 Coccus",
    class_names=["background", "biofilm", "planktonic"],
    num_classes=3,
    class_colors={
        "biofilm": (18, 225, 249, 90),
        "planktonic": (14, 101, 235, 90)
    },
    checkpoint_path=os.path.join(MODELS_DIR, "coccus.pth"),
    class_labels={"background": 0, "biofilm": 1, "planktonic": 2},
    postprocess_params={
        "biofilm": {
            "area": (100, 4587520)
        },
        "planktonic": {
            "area": (100, 4587520)
        }
    },
    filtration_params={
        "biofilm_area": (100, 100000),
        "planktonic_area": (100, 100000)
    },
    class_titles={
        "biofilm": ("Biofilms", "#12e1f9"),
        "planktonic": ("Planktonic", "#0e65eb")
    },
    class_weights={
        "background": 0.1,
        "biofilm": 1.0,
        "planktonic": 10.0
    },

    cvat_labels={
        "biofilm": "Biofilm",
        "planktonic": "Planktonic"
    },
    cvat_label_colors={
        "Biofilm": "#12e1f9",
        "Planktonic": "#0e65eb",
        "Defect": "#ff0000"
    }
)

COMBINED_CONFIG = ModelConfig(
    name="Combined",
    display_name="🔬 Combined (Bacillus + Coccus)",
    class_names=["background", "biofilmBacillus", "biofilmCoccus", "singleBacillus", 
                 "intermediateBacillus", "planktonicCoccus"],
    num_classes=6,
    class_colors={
        "biofilmBacillus": (36, 179, 83, 178),
        "biofilmCoccus": (18, 225, 249, 178),
        "singleBacillus": (184, 61, 245, 178),
        "intermediateBacillus": (221, 255, 51, 178),
        "planktonicCoccus": (14, 101, 235, 178)
    },
    checkpoint_path=os.path.join(MODELS_DIR, "combined.pth"),
    class_labels={
        "background": 0,
        "biofilmBacillus": 1,
        "biofilmCoccus": 2,
        "singleBacillus": 3,
        "intermediateBacillus": 4,
        "planktonicCoccus": 5
    },
    postprocess_params={
        "biofilmBacillus": {
            "area": (3000, 4587520)
        },
        "biofilmCoccus": {
            "area": (14000, 4587520)
        },
        "singleBacillus": {
            "area": (350, 5000),
            "ecc": (0.65, 1.0)
        },
        "intermediateBacillus": {
            "area": (1250, 458752)
        },
        "planktonicCoccus": {
            "area": (300, 14000)
        }
    },
    filtration_params={
        "biofilmBacillus_area": (100, 10000),
        "biofilmCoccus_area": (100, 100000),
        "singleBacillus_area": (100, 5000),
        "singleBacillus_ecc": (0.0, 1.0),
        "intermediateBacillus_area": (100, 10000),
        "planktonicCoccus_area": (100, 20000)
    },
    class_titles={
        "biofilmBacillus": ("Bacillus Biofilms", "#24b353"),
        "biofilmCoccus": ("Coccus Biofilms", "#12e1f9"),
        "singleBacillus": ("Single Bacillus", "#b83df5"),
        "intermediateBacillus": ("Intermediate Bacillus", "#ddff33"),
        "planktonicCoccus": ("Planktonic Coccus", "#0e65eb")
    },
    class_weights={
        "background": 0.1,
        "biofilmBacillus": 1.0,
        "biofilmCoccus": 1.0,
        "singleBacillus": 1.0,
        "intermediateBacillus": 1.0,
        "planktonicCoccus": 10.0
    },

    cvat_labels={
        "biofilmBacillus": "Biofilm_Bacillus",
        "biofilmCoccus": "Biofilm_Coccus",
        "singleBacillus": "Single_Bacillus",
        "intermediateBacillus": "Intermediate_Bacillus",
        "planktonicCoccus": "Planktonic_Coccus"
    },
    cvat_label_colors={
        "Biofilm_Bacillus": "#24b353",
        "Biofilm_Coccus": "#12e1f9",
        "Single_Bacillus": "#b83df5",
        "Intermediate_Bacillus": "#ddff33",
        "Planktonic_Coccus": "#0e65eb",
        "Defect": "#ff0000"
    }
)

MODEL_CONFIGS = {
    "Bacillus": BACILLUS_CONFIG,
    "Coccus": COCCUS_CONFIG
}