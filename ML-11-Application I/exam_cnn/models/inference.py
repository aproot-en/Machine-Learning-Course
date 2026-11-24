"""Load CNN15 once and classify character images with a confidence threshold."""

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from exam_cnn.data.preprocessing import character_normalize, normalize_pixels
from exam_cnn.models.cnn15 import CNN15

DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[2] / "exam_cnn/models_export/cnn15_best.pt"
)
CONFIDENCE_THRESHOLD = 0.90  ## matches config.yaml confidence.accept_threshold
TOP_K = 3
LETTERS = [chr(65 + i) for i in range(26)]  ## index -> 'A'..'Z'
UNKNOWN_LABEL = "UNKNOWN"


class LetterInferenceWrapper:
    def __init__(
        self,
        checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT_PATH,
        device: str | None = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.model = self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path) -> CNN15:
        """load_model(path): called once at construction, never per-request."""
        model = CNN15().to(self.device)
        ## The checkpoint is a plain state-dict bundle. Restrict unpickling to
        ## tensor weights so a replaced model file cannot execute arbitrary
        ## Python objects during application startup.
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Use the same character preprocessing as the training data."""
        normalized = character_normalize(image)
        return normalize_pixels(normalized)

    @torch.no_grad()
    def predict_batch(self, images: list[np.ndarray]) -> torch.Tensor:
        """Preprocess and predict a batch of character images."""
        if len(images) == 0:
            return torch.empty((0, len(LETTERS)))
        batch = np.stack([self.preprocess(img) for img in images])[
            :, None, :, :
        ].astype(np.float32)
        x = torch.from_numpy(batch).to(self.device)
        logits = self.model(x)
        return F.softmax(logits, dim=1).cpu()

    def predict_with_confidence(self, images: list[np.ndarray]):
        """Returns a list of (label, probability, top3) per image, where
        top3 is [(label, prob), ...] sorted by probability descending."""
        probs = self.predict_batch(images)
        results = []
        topk_probs, topk_idx = probs.topk(min(TOP_K, probs.shape[1]), dim=1)
        for i in range(len(images)):
            label = LETTERS[int(topk_idx[i, 0])]
            confidence = float(topk_probs[i, 0])
            top3 = [
                (LETTERS[int(topk_idx[i, k])], float(topk_probs[i, k]))
                for k in range(topk_idx.shape[1])
            ]
            results.append((label, confidence, top3))
        return results

    def reject_if_low_confidence(
        self, label: str, confidence: float, threshold: float | None = None
    ) -> str:
        """Returns UNKNOWN_LABEL if confidence is below threshold, else label unchanged."""
        if threshold is None:
            t = self.confidence_threshold
        else:
            t = threshold
        if confidence >= t:
            return label
        else:
            return UNKNOWN_LABEL

    def classify(self, images: list[np.ndarray], threshold: float | None = None):
        """Convenience: predict_with_confidence + reject_if_low_confidence combined."""
        out = []
        for label, confidence, top3 in self.predict_with_confidence(images):
            final = self.reject_if_low_confidence(label, confidence, threshold)
            out.append(
                {
                    "label": final,
                    "raw_label": label,
                    "confidence": confidence,
                    "top3": top3,
                }
            )
        return out
