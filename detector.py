import io
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Optional
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)

ELEPHANT_CLASS_ID = 20  # COCO class 20 = elephant
MODEL_NAME = "yolov8n.pt"  # nano — fast; swap to yolov8s.pt for better accuracy

_model: Optional[YOLO] = None

def get_model() -> YOLO:
    global _model
    if _model is None:
        logger.info(f"Loading YOLO model: {MODEL_NAME}")
        _model = YOLO(MODEL_NAME)
    return _model


def detect_elephants(image_bytes: bytes, confidence: float = 0.5) -> dict:
    """
    Run elephant detection on raw image bytes.
    Returns dict with detections, annotated image bytes, and summary.
    """
    model = get_model()

    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image")

    # Run inference — filter to elephant class only
    results = model(img_bgr, classes=[ELEPHANT_CLASS_ID], conf=confidence, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            detections.append({
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
            })

    # Draw annotations on image
    annotated_bgr = results[0].plot() if results else img_bgr
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(annotated_rgb)

    # Add overlay banner if elephants found
    if detections:
        draw = ImageDraw.Draw(pil_img)
        banner = f"🐘 ELEPHANT DETECTED — {len(detections)} found"
        draw.rectangle([0, 0, pil_img.width, 36], fill=(220, 50, 50, 200))
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except Exception:
            font = ImageFont.load_default()
        draw.text((10, 8), banner, fill="white", font=font)

    # Encode back to JPEG
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    annotated_bytes = buf.getvalue()

    return {
        "elephant_count": len(detections),
        "detected": len(detections) > 0,
        "detections": detections,
        "annotated_image": annotated_bytes,
    }


def capture_rtsp_frame(rtsp_url: str) -> Optional[bytes]:
    """Grab one frame from an RTSP stream and return it as JPEG bytes."""
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame)
    return bytes(buf) if ok else None
