# === LIBRARIES GENERAL ===
import io
from pathlib import Path
from typing import List, Optional

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "extra" / "examples"

EXAMPLE_FILES = [
    ("example1.bmp", "An example of a developing biofilm of bacillus (Pseudomonas Veronii)."),
    ("example2.bmp", "An example of a coccus biofilm forming (Staphylococcus Hominis)."),
    ("example3.bmp", "An example of a mature coccus biofilm (Staphylococcus Hominis)."),
    ("example4.bmp", "An example of accumulation of bacillus biofilms (Pseudomonas Veronii)."),
]


class ExampleFile(io.BytesIO):

    def __init__(self, data: bytes, name: str, description: str = ""):
        super().__init__(data)
        self.name = name
        self.description = description

    @property
    def size(self) -> int:
        return len(self.getbuffer())

    def getvalue(self) -> bytes:  # как у UploadedFile
        return self.getbuffer().tobytes()


def load_examples() -> List[ExampleFile]:
    examples: List[ExampleFile] = []
    for filename, description in EXAMPLE_FILES:
        path = EXAMPLES_DIR / filename
        if not path.exists():
            continue
        examples.append(ExampleFile(path.read_bytes(), filename, description))
    return examples


def get_example_by_name(name: str) -> Optional[ExampleFile]:
    for ex in load_examples():
        if ex.name == name:
            return ex
    return None