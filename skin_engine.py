"""
BioAesthetic — Skin Analysis Engine
Phase 1: Image-based heuristic scoring via OpenCV.
Phase 2: Plug in CNN (ResNet-18) by setting USE_ML_INFERENCE=True.

All scores are proxy aesthetic estimates — not medical diagnosis.
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime, timezone
import uuid

from config import settings

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RawSkinMetrics:
    acne_score: float
    pigmentation_score: float
    pore_score: float
    inflammation_score: float
    under_eye_score: float
    skin_age_estimate: int
    composite_skin_score: float
    analysis_id: str
    analyzed_at: datetime
    recommendations: list[str]


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE PREPROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

class ImagePreprocessor:
    """
    Handles face detection and normalization before analysis.
    Uses Haar cascade for Phase 1; swap for MTCNN in Phase 2.
    """

    TARGET_SIZE = (224, 224)
    CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)

    def load_image(self, image_bytes: bytes) -> np.ndarray:
        """Decode bytes → BGR ndarray."""
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image. Ensure valid JPEG/PNG.")
        return img

    def detect_and_crop_face(self, img: np.ndarray) -> np.ndarray:
        """
        Detect the largest frontal face and return a tight crop.
        Falls back to full image if no face is detected.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        if len(faces) == 0:
            log.warning("No face detected — analysing full image.")
            return img

        # Pick largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        # Add 15% padding around the face
        pad_x = int(w * 0.15)
        pad_y = int(h * 0.15)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img.shape[1], x + w + pad_x)
        y2 = min(img.shape[0], y + h + pad_y)
        return img[y1:y2, x1:x2]

    def normalize(self, img: np.ndarray) -> np.ndarray:
        """Resize to 224×224 and normalize pixel values to [0, 1]."""
        resized = cv2.resize(img, self.TARGET_SIZE, interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    def preprocess(self, image_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
        """Full pipeline: bytes → (normalized_face, raw_face_crop)."""
        raw = self.load_image(image_bytes)
        face_crop = self.detect_and_crop_face(raw)
        normalized = self.normalize(face_crop)
        return normalized, face_crop


# ══════════════════════════════════════════════════════════════════════════════
# RULE-BASED SCORING FUNCTIONS (Phase 1)
# ══════════════════════════════════════════════════════════════════════════════

class HeuristicSkinScorer:
    """
    Computes individual skin metric scores using classical CV.
    All outputs are [0, 1] unless noted.
    """

    # ── Acne Detection ─────────────────────────────────────────────────────────
    def score_acne(self, face_crop: np.ndarray) -> float:
        """
        Detect potential acne lesions as small circular high-contrast regions.
        Returns probability 0–1 (higher = more likely acne present).
        """
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        # Morphological top-hat: isolates small bright spots on dark skin
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        # Blob detection for small circular anomalies
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 10
        params.maxArea = 800
        params.filterByCircularity = True
        params.minCircularity = 0.5
        params.filterByConvexity = True
        params.minConvexity = 0.7
        detector = cv2.SimpleBlobDetector_create(params)
        _, thresh = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)
        keypoints = detector.detect(thresh)

        # Normalize: assume >15 blobs = high acne probability
        raw = len(keypoints) / 15.0
        return float(np.clip(raw, 0.0, 1.0))

    # ── Pigmentation ───────────────────────────────────────────────────────────
    def score_pigmentation(self, face_crop: np.ndarray) -> float:
        """
        Estimate hyperpigmentation density via luminance variance in L*a*b* space.
        High local luminance variance in skin-like hues → higher score.
        """
        lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2Lab)
        l_channel = lab[:, :, 0].astype(np.float32)

        # Local standard deviation via sliding window
        blur = cv2.GaussianBlur(l_channel, (21, 21), 0)
        diff = np.abs(l_channel - blur)
        mean_diff = float(np.mean(diff))

        # Normalize: empirically 0–25 is typical range
        return float(np.clip(mean_diff / 25.0, 0.0, 1.0))

    # ── Pore Density ───────────────────────────────────────────────────────────
    def score_pores(self, face_crop: np.ndarray) -> float:
        """
        Estimate pore density using Laplacian edge detection.
        More fine-grained edge detail → higher pore score.
        """
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        # Normalize brightness first
        equalized = cv2.equalizeHist(gray)
        laplacian = cv2.Laplacian(equalized, cv2.CV_64F)
        variance = float(np.var(laplacian))

        # Empirical normalization; high variance ≈ high pore density
        return float(np.clip(variance / 5000.0, 0.0, 1.0))

    # ── Inflammation ───────────────────────────────────────────────────────────
    def score_inflammation(self, face_crop: np.ndarray) -> float:
        """
        Detect redness/inflammation via red-channel dominance in skin regions.
        """
        # Convert to float and extract channels
        img_f = face_crop.astype(np.float32) / 255.0
        r, g, b = img_f[:, :, 2], img_f[:, :, 1], img_f[:, :, 0]

        # Redness index: normalized red dominance over green
        with np.errstate(divide="ignore", invalid="ignore"):
            redness = np.where(g > 0, (r - g) / (r + g + 1e-6), 0.0)

        mean_redness = float(np.mean(np.clip(redness, 0, 1)))
        return float(np.clip(mean_redness * 3.5, 0.0, 1.0))

    # ── Under-Eye Darkness ────────────────────────────────────────────────────
    def score_under_eye(self, face_crop: np.ndarray) -> float:
        """
        Sample the lower-third of the cropped face region as a proxy for
        the under-eye area and measure mean darkness.
        """
        h, w = face_crop.shape[:2]
        under_eye_region = face_crop[int(h * 0.45):int(h * 0.65), int(w * 0.2):int(w * 0.8)]
        if under_eye_region.size == 0:
            return 0.3  # fallback

        lab = cv2.cvtColor(under_eye_region, cv2.COLOR_BGR2Lab)
        mean_l = float(np.mean(lab[:, :, 0]))  # L* channel; lower = darker

        # Normalize: L*=60 is typical light, L*=30 is dark
        darkness = 1.0 - (mean_l / 100.0)
        return float(np.clip(darkness, 0.0, 1.0))

    # ── Skin Age Estimate ─────────────────────────────────────────────────────
    def estimate_skin_age(
        self,
        acne: float,
        pigmentation: float,
        pores: float,
        inflammation: float,
        user_age: Optional[int] = None,
    ) -> int:
        """
        Heuristic skin age estimation based on combined metric weights.
        If user's chronological age is provided, anchors the estimate.
        """
        base = user_age if user_age else 30
        penalty = (
            pigmentation * 8.0
            + pores * 5.0
            + acne * 3.0
            + inflammation * 4.0
        )
        skin_age = int(round(base + penalty - 2.0))  # -2 optimism bias
        return max(13, min(90, skin_age))


# ══════════════════════════════════════════════════════════════════════════════
# CNN INFERENCE STUB (Phase 2)
# ══════════════════════════════════════════════════════════════════════════════

class CNNSkinInferencer:
    """
    Phase 2: ResNet-18 fine-tuned on dermatological data.
    Replace heuristic outputs with model predictions.
    """

    def __init__(self, model_path: str):
        self._model = None
        self._model_path = model_path
        self._load_model()

    def _load_model(self):
        try:
            import torch
            import torchvision.models as models

            model = models.resnet18(weights=None)
            # Replace classifier for multi-label output
            model.fc = torch.nn.Linear(model.fc.in_features, 6)
            model.load_state_dict(torch.load(self._model_path, map_location="cpu"))
            model.eval()
            self._model = model
            log.info("CNN skin model loaded successfully.")
        except Exception as e:
            log.warning(f"Could not load CNN model: {e}. Falling back to heuristics.")
            self._model = None

    def predict(self, normalized_face: np.ndarray) -> Optional[dict]:
        """Returns dict of raw model outputs [0,1] or None if model unavailable."""
        if self._model is None:
            return None
        try:
            import torch
            tensor = torch.tensor(normalized_face).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                output = torch.sigmoid(self._model(tensor)).squeeze().numpy()
            keys = ["acne_score", "pigmentation_score", "pore_score",
                    "inflammation_score", "under_eye_score", "skin_age_raw"]
            return dict(zip(keys, output.tolist()))
        except Exception as e:
            log.error(f"CNN inference failed: {e}")
            return None


# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATION GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_skin_recommendations(
    acne: float,
    pigmentation: float,
    pores: float,
    inflammation: float,
    under_eye: float,
) -> list[str]:
    recs = []
    if acne > 0.5:
        recs.append("Consider a BHA (salicylic acid) cleanser to address congestion in pores.")
    if acne > 0.7:
        recs.append("Reduce dietary glycemic load; high-GI foods are linked to acne exacerbation.")
    if pigmentation > 0.5:
        recs.append("Apply broad-spectrum SPF 50+ daily to prevent further UV-induced pigmentation.")
    if pigmentation > 0.65:
        recs.append("A topical niacinamide serum (5–10%) may help reduce hyperpigmentation over 8–12 weeks.")
    if pores > 0.55:
        recs.append("Retinol (0.025–0.1%) applied nightly can reduce pore appearance over time.")
    if inflammation > 0.4:
        recs.append("Switch to fragrance-free, non-comedogenic products to reduce irritant-driven redness.")
    if under_eye > 0.55:
        recs.append("Prioritize 7–9 hours of sleep and maintain adequate iron and vitamin K intake.")
    if under_eye > 0.7:
        recs.append("An eye cream containing caffeine and vitamin C may temporarily reduce under-eye darkness.")
    if not recs:
        recs.append("Skin metrics are within healthy ranges. Maintain your current routine.")
    return recs


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE SKIN SCORE
# ══════════════════════════════════════════════════════════════════════════════

def compute_composite_skin_score(
    acne: float,
    pigmentation: float,
    pores: float,
    inflammation: float,
    under_eye: float,
) -> float:
    """
    Weighted composite score 0–100.
    All inputs in [0, 1]; higher raw value = more of that issue = lower score.
    """
    penalty = (
        0.25 * acne
        + 0.20 * pigmentation
        + 0.20 * pores
        + 0.20 * inflammation
        + 0.15 * under_eye
    )
    return round((1.0 - penalty) * 100.0, 2)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class SkinEngine:
    """
    Facade that routes through heuristic (Phase 1) or CNN (Phase 2) pipeline.
    """

    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.heuristic = HeuristicSkinScorer()
        self.cnn: Optional[CNNSkinInferencer] = None

        if settings.USE_ML_INFERENCE:
            self.cnn = CNNSkinInferencer(settings.SKIN_MODEL_PATH)

    def analyse(self, image_bytes: bytes, user_age: Optional[int] = None) -> RawSkinMetrics:
        """
        End-to-end skin analysis pipeline.

        Args:
            image_bytes: Raw uploaded image bytes.
            user_age: Chronological age to anchor skin age estimate.

        Returns:
            RawSkinMetrics dataclass with all scores.
        """
        normalized, face_crop = self.preprocessor.preprocess(image_bytes)

        # Phase 2 CNN path
        cnn_output = None
        if self.cnn:
            cnn_output = self.cnn.predict(normalized)

        if cnn_output:
            acne        = float(np.clip(cnn_output["acne_score"], 0, 1))
            pigmentation = float(np.clip(cnn_output["pigmentation_score"], 0, 1))
            pores       = float(np.clip(cnn_output["pore_score"], 0, 1))
            inflammation = float(np.clip(cnn_output["inflammation_score"], 0, 1))
            under_eye   = float(np.clip(cnn_output["under_eye_score"], 0, 1))
            skin_age    = int(round(cnn_output["skin_age_raw"] * 80 + 13))
        else:
            # Phase 1 heuristic path
            face_uint8 = (face_crop * 255).astype(np.uint8) if face_crop.dtype != np.uint8 else face_crop
            acne        = self.heuristic.score_acne(face_uint8)
            pigmentation = self.heuristic.score_pigmentation(face_uint8)
            pores       = self.heuristic.score_pores(face_uint8)
            inflammation = self.heuristic.score_inflammation(face_uint8)
            under_eye   = self.heuristic.score_under_eye(face_uint8)
            skin_age    = self.heuristic.estimate_skin_age(acne, pigmentation, pores, inflammation, user_age)

        composite = compute_composite_skin_score(acne, pigmentation, pores, inflammation, under_eye)
        recs = generate_skin_recommendations(acne, pigmentation, pores, inflammation, under_eye)

        return RawSkinMetrics(
            acne_score=round(acne, 4),
            pigmentation_score=round(pigmentation, 4),
            pore_score=round(pores, 4),
            inflammation_score=round(inflammation, 4),
            under_eye_score=round(under_eye, 4),
            skin_age_estimate=skin_age,
            composite_skin_score=composite,
            analysis_id=str(uuid.uuid4()),
            analyzed_at=datetime.now(timezone.utc),
            recommendations=recs,
        )


# ── Module-level singleton ────────────────────────────────────────────────────
skin_engine = SkinEngine()
