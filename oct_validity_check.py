"""Zero-shot check for whether an uploaded image is actually a retinal OCT
scan, using a pretrained CLIP model — no OCT-specific training data needed.
Guards the vision model from being asked to classify AD/NORMAL on an image
that was never an OCT scan to begin with (e.g. a random photo).
"""
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

_MODEL_NAME = "openai/clip-vit-base-patch32"

_CANDIDATE_LABELS = [
    "a retinal optical coherence tomography (OCT) scan, a grayscale cross-sectional medical image of the retina showing layered tissue bands",
    "a photograph of a person, object, or scene",
    "a document, screenshot, or page of text",
    "a different kind of medical scan such as an X-ray, MRI, or CT scan",
]
_OCT_LABEL_INDEX = 0

_model = None
_processor = None


def _get_clip():
    global _model, _processor
    if _model is None:
        _model = CLIPModel.from_pretrained(_MODEL_NAME)
        _processor = CLIPProcessor.from_pretrained(_MODEL_NAME)
        _model.eval()
    return _model, _processor


def is_valid_oct_image(image_path: str, threshold: float = 0.5) -> dict:
    """Returns {"is_valid_oct": bool, "oct_probability": float, "label_probabilities": dict}."""
    model, processor = _get_clip()
    image = Image.open(image_path).convert("RGB")

    inputs = processor(text=_CANDIDATE_LABELS, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

    oct_probability = float(probs[_OCT_LABEL_INDEX].item())
    label_probabilities = {label: float(p) for label, p in zip(_CANDIDATE_LABELS, probs)}

    return {
        "is_valid_oct": oct_probability >= threshold,
        "oct_probability": oct_probability,
        "label_probabilities": label_probabilities,
    }
