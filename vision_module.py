"""Vision subsystem: loads the trained ResNet50 AD/NORMAL classifier and runs
single-image inference, returning the exact shape rag_pipeline.RAGState needs:
{cnn_prediction, cnn_confidence, visual_features}.

To update the model: overwrite models/vision_model.pth with a new checkpoint
(same architecture — ResNet50 backbone, binary head) — nothing else changes.
"""
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import models, transforms

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_CONFIG_PATH, "r") as _f:
    _CONFIG = yaml.safe_load(_f)

_VISION_CFG = _CONFIG["vision"]
_CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), _VISION_CFG["checkpoint_path"])
_IMAGE_SIZE = _VISION_CFG["image_size"]
_CLASS_NAMES = _VISION_CFG["class_names"]  # index 0 = AD, index 1 = NORMAL
_AD_INDEX = _CLASS_NAMES.index("AD")
_DEVICE = torch.device(
    _VISION_CFG["device"] if torch.cuda.is_available() or _VISION_CFG["device"] == "cpu" else "cpu"
)

_EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((_IMAGE_SIZE, _IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _build_model() -> nn.Module:
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, len(_CLASS_NAMES)))
    return model


class _GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        _, _, h, w = x.size()
        logits = self.model(x)
        self.model.zero_grad()
        logits[0, class_idx].backward(retain_graph=True)

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w_i in enumerate(weights):
            cam += w_i * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (w, h))
        cam -= np.min(cam)
        cam /= (np.max(cam) + 1e-8)
        return cam


_model = None
_cam_extractor = None


def _get_model() -> tuple[nn.Module, "_GradCAM"]:
    global _model, _cam_extractor
    if _model is None:
        if not os.path.exists(_CHECKPOINT_PATH):
            raise FileNotFoundError(
                f"No vision checkpoint at {_CHECKPOINT_PATH}. "
                f"Download the trained .pth from Colab and place it there "
                f"(see models/README.md)."
            )
        model = _build_model()
        state_dict = torch.load(_CHECKPOINT_PATH, map_location=_DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(_DEVICE)
        model.eval()

        # Defensive check: catches exactly the kind of bug where a stale/mismatched
        # checkpoint silently produces the wrong number of output classes.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, _IMAGE_SIZE, _IMAGE_SIZE, device=_DEVICE)
            n_out = model(dummy).shape[1]
        assert n_out == len(_CLASS_NAMES), (
            f"Model outputs {n_out} classes but config expects {len(_CLASS_NAMES)} "
            f"({_CLASS_NAMES}). Checkpoint/architecture mismatch — do not trust "
            f"predictions until this is fixed."
        )

        _model = model
        _cam_extractor = _GradCAM(model, model.layer4[-1].conv3)
    return _model, _cam_extractor


def _describe_attention(cam: np.ndarray, img_gray_resized: np.ndarray) -> str:
    h, w = cam.shape
    y_peak, x_peak = np.unravel_index(cam.argmax(), cam.shape)

    vert = "upper" if y_peak < h / 3 else ("lower" if y_peak > 2 * h / 3 else "central")
    horiz = "left" if x_peak < w / 3 else ("right" if x_peak > 2 * w / 3 else "central")

    peak_intensity = img_gray_resized[y_peak, x_peak]
    mean_intensity = np.mean(img_gray_resized)
    tissue_type = "retinal band" if peak_intensity > mean_intensity * 1.1 else "background"

    region = "central" if vert == horiz == "central" else f"{vert} {horiz}"
    return f"Peak model attention localized to the {region} region, primarily overlapping the {tissue_type}."


def run_inference(image_path: str) -> dict:
    """Returns {cnn_prediction: str, cnn_confidence: float, visual_features: str}."""
    model, cam_extractor = _get_model()

    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = _EVAL_TRANSFORMS(img_pil).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        confidence = probs[0, pred_idx].item()

    cam = cam_extractor(img_tensor.clone().requires_grad_(True), class_idx=pred_idx)

    img_gray = np.array(img_pil.convert("L"))
    img_gray_resized = cv2.resize(img_gray, cam.shape[::-1])
    visual_features = _describe_attention(cam, img_gray_resized)

    prediction = (
        "Alzheimer's Risk Indicator Positive"
        if pred_idx == _AD_INDEX
        else "Alzheimer's Risk Indicator Negative"
    )

    return {
        "cnn_prediction": prediction,
        "cnn_confidence": float(confidence),
        "visual_features": visual_features,
    }
