"""
Centralized ML model registry.

Models are loaded exactly once, at application startup (see app/main.py's
lifespan handler), and reused for every request. This mirrors the original
Flask app's module-level model loading, but avoids doing heavy work at
*import* time so FastAPI's startup lifecycle stays in control of it (useful
for testing, workers, etc).
"""
import torch
import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# correct.py lives at the project root (same level as the old app.py) and is
# not modified — plain import works as-is, no sys.path hack needed.
from correct import correct, DICTIONARY


class ModelRegistry:
    """Holds singleton references to the loaded OCR/TrOCR models."""

    reader = None
    processor = None
    model = None
    known_medicines = None
    _loaded = False

    @classmethod
    def load(cls):
        if cls._loaded:
            return

        print("Loading models...")

        cls.reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
        cls.processor = TrOCRProcessor.from_pretrained("./trocr_best")
        cls.model = VisionEncoderDecoderModel.from_pretrained("./trocr_best")
        cls.model.eval()
        if torch.cuda.is_available():
            cls.model = cls.model.cuda()

        cls.known_medicines = sorted(set(DICTIONARY))
        cls._loaded = True

        print("Models loaded.")

    @classmethod
    def ensure_loaded(cls):
        if not cls._loaded:
            cls.load()


registry = ModelRegistry()

# Re-exported so services can do `from app.models.registry import correct`
# without duplicating the crnn-pytorch sys.path hack in multiple files.
__all__ = ["registry", "ModelRegistry", "correct", "DICTIONARY"]
