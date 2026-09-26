"""
BeltGuard — Conveyor Belt Damage Detection Pipeline

Features:
1. Belt ROI masking
2. YOLOv8 defect detection
3. Confidence filtering
4. Optional TTA
5. Visual severity estimation
6. Maintenance priority estimation
7. Annotated images
8. ROI autoencoder anomaly detection
9. Health score
10. Optional Grad-CAM explainability

IMPORTANT:
- Grad-CAM is DISABLED by default to reduce memory usage.
- Enable it only when sufficient RAM is available.
- Severity is a visual severity PROXY based on relative
  detected bounding-box area.
"""

# ============================================================
# MEMORY SETTINGS
# ============================================================

import os

# Reduce CPU thread memory usage.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import gc
import json
import argparse

import numpy as np
import cv2
import torch

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from ultralytics import YOLO
from belt_roi_autoencoder import (
    BeltAutoencoder,
    extract_belt_roi
)


# ============================================================
# CONFIGURATION
# ============================================================

CLASS_NAMES = {
    0: "scratch",
    1: "edge_damage"
}

CLASS_COLORS = {
    0: (0, 255, 0),
    1: (0, 0, 255)
}

# Based on validation.
DEFAULT_CONFIDENCE = 0.24

# ROI Autoencoder
ANOMALY_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "belt_roi_autoencoder.pt"
)

ANOMALY_THRESHOLD = 0.001486


# ============================================================
# GRADCAM CONFIGURATION
# ============================================================

# IMPORTANT:
# False = Render-safe mode
# True  = Generate Grad-CAM when pipeline runs
#
# Keep this FALSE on the Render 512 MB instance.
ENABLE_GRADCAM = False

# Environment variable can override the setting.
#
# Example:
# BELTGUARD_GRADCAM=true
#
if os.getenv(
    "BELTGUARD_GRADCAM",
    "false"
).lower() == "true":

    ENABLE_GRADCAM = True


# ============================================================
# DEVICE
# ============================================================

# Render free instance should use CPU.
DEVICE = "cpu"


# ============================================================
# ROI FUNCTIONS
# ============================================================

def load_belt_roi(
    label_path,
    img_w,
    img_h
):
    """Load belt ROI polygon from YOLO label file."""

    if not os.path.exists(label_path):
        return None

    with open(
        label_path,
        "r"
    ) as f:

        line = f.readline().strip()

    if not line:
        return None

    parts = line.split()

    if len(parts) < 7:
        return None

    coords = [
        float(x)
        for x in parts[1:]
    ]

    pts = np.array(
        [
            (
                int(coords[i] * img_w),
                int(coords[i + 1] * img_h)
            )
            for i in range(
                0,
                len(coords),
                2
            )
        ],
        dtype=np.int32
    )

    return pts


def create_belt_mask(
    img_w,
    img_h,
    polygon_pts
):
    """Create binary mask of the belt region."""

    mask = np.zeros(
        (img_h, img_w),
        dtype=np.uint8
    )

    cv2.fillPoly(
        mask,
        [polygon_pts],
        255
    )

    return mask


def mask_image(
    img,
    belt_mask
):
    """Apply belt mask to image."""

    masked = img.copy()

    masked[
        belt_mask == 0
    ] = [128, 128, 128]

    return masked


def filter_detections_in_belt(
    detections,
    belt_mask
):
    """Keep detections whose center is inside the belt ROI."""

    h, w = belt_mask.shape

    filtered = []

    for det in detections:

        bbox = det["bbox"]

        cx = int(
            (bbox[0] + bbox[2]) / 2
        )

        cy = int(
            (bbox[1] + bbox[3]) / 2
        )

        cx = max(
            0,
            min(w - 1, cx)
        )

        cy = max(
            0,
            min(h - 1, cy)
        )

        if belt_mask[cy, cx] > 0:

            filtered.append(det)

    return filtered


# ============================================================
# ROI AUTOENCODER
# ============================================================

def load_anomaly_model():

    if not ANOMALY_MODEL_PATH.exists():

        print(
            "WARNING: ROI Autoencoder not found:"
        )

        print(
            ANOMALY_MODEL_PATH
        )

        return None, torch.device("cpu")

    device = torch.device("cpu")

    try:

        model = BeltAutoencoder().to(device)

        state_dict = torch.load(
            ANOMALY_MODEL_PATH,
            map_location=device,
            weights_only=True
        )

        model.load_state_dict(
            state_dict
        )

        del state_dict

        model.eval()

        print(
            f"ROI Autoencoder loaded: "
            f"{ANOMALY_MODEL_PATH}"
        )

        return model, device

    except Exception as e:

        print(
            "WARNING: Could not load ROI Autoencoder:"
        )

        print(e)

        gc.collect()

        return None, device


def calculate_roi_anomaly_score(
    model,
    device,
    image_path,
    label_path
):

    if model is None:

        return None, "UNAVAILABLE"

    try:

        roi = extract_belt_roi(
            image_path,
            label_path
        )

        if roi is None or roi.size == 0:

            return None, "ROI_UNAVAILABLE"

        roi = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2RGB
        )

        roi = cv2.resize(
            roi,
            (128, 128),
            interpolation=cv2.INTER_AREA
        )

        roi = (
            roi.astype(
                np.float32
            ) / 255.0
        )

        roi = np.transpose(
            roi,
            (2, 0, 1)
        )

        tensor = torch.from_numpy(
            roi
        ).unsqueeze(0)

        tensor = tensor.to(device)

        with torch.no_grad():

            reconstructed = model(
                tensor
            )

            error = torch.mean(
                (
                    tensor
                    - reconstructed
                ) ** 2
            ).item()

        status = (
            "ANOMALOUS"
            if error >= ANOMALY_THRESHOLD
            else "NORMAL"
        )

        del roi
        del tensor
        del reconstructed

        gc.collect()

        return (
            round(error, 8),
            status
        )

    except Exception as e:

        print(
            "WARNING: Anomaly scoring failed "
            f"for {os.path.basename(image_path)}: {e}"
        )

        gc.collect()

        return None, "ERROR"


# ============================================================
# SEVERITY
# ============================================================

def calculate_severity(
    bbox,
    img_w,
    img_h
):
    """
    Estimate visual severity using relative
    bounding-box area.
    """

    x1, y1, x2, y2 = bbox

    box_width = max(
        0,
        x2 - x1
    )

    box_height = max(
        0,
        y2 - y1
    )

    box_area = (
        box_width
        * box_height
    )

    image_area = (
        img_w
        * img_h
    )

    if image_area == 0:

        return {
            "severity_score": 0.0,
            "area_ratio": 0.0,
            "severity": "Low"
        }

    area_ratio = (
        box_area
        / image_area
    )

    score = min(
        area_ratio * 5000,
        100
    )

    if score < 25:

        severity = "Low"

    elif score < 60:

        severity = "Medium"

    else:

        severity = "High"

    return {
        "severity_score": round(
            score,
            2
        ),
        "area_ratio": round(
            area_ratio,
            6
        ),
        "severity": severity
    }


# ============================================================
# HEALTH SCORE
# ============================================================

def calculate_health_score(
    detections,
    anomaly_score,
    anomaly_threshold
):
    """
    Calculate Visual Condition Score from 0-100.
    """

    # --------------------------------------------------------
    # DEFECT CONDITION
    # --------------------------------------------------------

    defect_count = len(
        detections
    )

    defect_condition = (
        100.0
        /
        (
            1.0
            + defect_count / 5.0
        )
    )

    # --------------------------------------------------------
    # SEVERITY CONDITION
    # --------------------------------------------------------

    severity_values = {
        "High": 1.0,
        "Medium": 0.5,
        "Low": 0.15
    }

    severity_burden = sum(
        severity_values.get(
            det.get(
                "severity",
                "Low"
            ),
            0.0
        )
        for det in detections
    )

    severity_condition = (
        100.0
        /
        (
            1.0
            + severity_burden / 3.0
        )
    )

    # --------------------------------------------------------
    # ANOMALY CONDITION
    # --------------------------------------------------------

    if (
        anomaly_score is None
        or anomaly_threshold <= 0
    ):

        anomaly_condition = 100.0

    else:

        anomaly_ratio = (
            anomaly_score
            / anomaly_threshold
        )

        anomaly_condition = (
            100.0
            /
            (
                1.0
                + anomaly_ratio
            )
        )

    # --------------------------------------------------------
    # WEIGHTED SCORE
    # --------------------------------------------------------

    health_score = (
        0.25 * defect_condition
        + 0.45 * severity_condition
        + 0.30 * anomaly_condition
    )

    health_score = max(
        0.0,
        min(
            100.0,
            health_score
        )
    )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    if health_score >= 80:

        status = "HEALTHY"

    elif health_score >= 60:

        status = "WARNING"

    elif health_score >= 40:

        status = "DEGRADED"

    else:

        status = "CRITICAL"

    return {
        "score": round(
            health_score,
            2
        ),
        "status": status,
        "defect_condition": round(
            defect_condition,
            2
        ),
        "severity_condition": round(
            severity_condition,
            2
        ),
        "anomaly_condition": round(
            anomaly_condition,
            2
        ),
        "defect_weight": 0.25,
        "severity_weight": 0.45,
        "anomaly_weight": 0.30,
        "method": (
            "Normalized weighted visual "
            "condition score"
        )
    }


# ============================================================
# MAINTENANCE PRIORITY
# ============================================================

def calculate_maintenance_priority(
    severity,
    confidence
):

    if severity == "High":

        return "IMMEDIATE INSPECTION"

    if severity == "Medium":

        return "SCHEDULE INSPECTION"

    if confidence >= 0.70:

        return "MONITOR"

    return "MONITOR / VERIFY"


# ============================================================
# NMS
# ============================================================

def nms_detections(
    detections,
    iou_thresh=0.5
):

    if not detections:

        return []

    dets = sorted(
        detections,
        key=lambda d: d["confidence"],
        reverse=True
    )

    keep = []

    for d in dets:

        box = d["bbox"]

        overlap = False

        for k in keep:

            kb = k["bbox"]

            ix1 = max(
                box[0],
                kb[0]
            )

            iy1 = max(
                box[1],
                kb[1]
            )

            ix2 = min(
                box[2],
                kb[2]
            )

            iy2 = min(
                box[3],
                kb[3]
            )

            inter = (
                max(
                    0,
                    ix2 - ix1
                )
                *
                max(
                    0,
                    iy2 - iy1
                )
            )

            a1 = (
                max(
                    0,
                    box[2] - box[0]
                )
                *
                max(
                    0,
                    box[3] - box[1]
                )
            )

            a2 = (
                max(
                    0,
                    kb[2] - kb[0]
                )
                *
                max(
                    0,
                    kb[3] - kb[1]
                )
            )

            union = (
                a1
                + a2
                - inter
            )

            if union > 0:

                iou = (
                    inter
                    / union
                )

                if iou > iou_thresh:

                    overlap = True
                    break

        if not overlap:

            keep.append(d)

    return keep


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(
    img_pil,
    detections,
    font_size=None
):

    draw = ImageDraw.Draw(
        img_pil
    )

    if font_size is None:

        font_size = max(
            24,
            min(
                42,
                int(
                    min(
                        img_pil.width,
                        img_pil.height
                    )
                    * 0.014
                )
            )
        )

    try:

        font = ImageFont.truetype(
            "C:/Windows/Fonts/arialbd.ttf",
            font_size
        )

    except (
        IOError,
        OSError
    ):

        try:

            font = ImageFont.truetype(
                "C:/Windows/Fonts/arial.ttf",
                font_size
            )

        except (
            IOError,
            OSError
        ):

            font = ImageFont.load_default()

    for det in detections:

        x1, y1, x2, y2 = det["bbox"]

        cls = det.get(
            "class",
            0
        )

        conf = det.get(
            "confidence",
            0.0
        )

        severity = det.get(
            "severity",
            "Unknown"
        )

        class_name = CLASS_NAMES.get(
            cls,
            f"cls{cls}"
        )

        color = CLASS_COLORS.get(
            cls,
            (255, 255, 0)
        )

        # ----------------------------------------------------
        # BOUNDING BOX
        # ----------------------------------------------------

        x1 = int(
            max(
                0,
                min(
                    img_pil.width - 1,
                    x1
                )
            )
        )

        y1 = int(
            max(
                0,
                min(
                    img_pil.height - 1,
                    y1
                )
            )
        )

        x2 = int(
            max(
                0,
                min(
                    img_pil.width - 1,
                    x2
                )
            )
        )

        y2 = int(
            max(
                0,
                min(
                    img_pil.height - 1,
                    y2
                )
            )
        )

        box_width = max(
            3,
            int(
                min(
                    img_pil.width,
                    img_pil.height
                )
                * 0.0025
            )
        )

        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2
            ],
            outline=color,
            width=box_width
        )

        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        label = (
            f"{class_name.upper()} | "
            f"{conf:.0%} | "
            f"{severity.upper()}"
        )

        text_box = draw.textbbox(
            (0, 0),
            label,
            font=font
        )

        text_width = (
            text_box[2]
            - text_box[0]
        )

        text_height = (
            text_box[3]
            - text_box[1]
        )

        padding_x = 10
        padding_y = 6

        label_width = (
            text_width
            + padding_x * 2
        )

        label_height = (
            text_height
            + padding_y * 2
        )

        label_x = x1

        label_y = (
            y1
            - label_height
            - 5
        )

        if label_y < 0:

            label_y = y2 + 5

        if (
            label_x
            + label_width
            > img_pil.width
        ):

            label_x = (
                img_pil.width
                - label_width
                - 5
            )

        if (
            label_y
            + label_height
            > img_pil.height
        ):

            label_y = (
                img_pil.height
                - label_height
                - 5
            )

        draw.rectangle(
            [
                label_x,
                label_y,
                label_x + label_width,
                label_y + label_height
            ],
            fill=(20, 20, 20)
        )

        draw.rectangle(
            [
                label_x,
                label_y,
                label_x + label_width,
                label_y + 4
            ],
            fill=color
        )

        draw.text(
            (
                label_x + padding_x,
                label_y + padding_y
            ),
            label,
            fill=(255, 255, 255),
            font=font
        )

    return img_pil


# ============================================================
# GRAD-CAM
# ============================================================

def run_gradcam_if_enabled(
    model,
    image_path,
    output_path
):
    """
    Generate Grad-CAM only when explicitly enabled.

    This function imports gradcam_utils lazily so that
    Render does not load Grad-CAM during normal startup.
    """

    if not ENABLE_GRADCAM:

        print(
            "Grad-CAM disabled for memory-safe deployment."
        )

        return (
            "DISABLED",
            None,
            None
        )

    try:

        # IMPORTANT:
        # Lazy import.
        from gradcam_utils import (
            generate_gradcam
        )

        print(
            f"Generating Grad-CAM for: "
            f"{image_path}"
        )

        generate_gradcam(
            model=model,
            image_path=image_path,
            output_path=output_path
        )

        if os.path.isfile(
            output_path
        ):

            return (
                "GENERATED",
                output_path,
                None
            )

        return (
            "FAILED",
            None,
            "Grad-CAM output was not created."
        )

    except Exception as e:

        error = (
            f"{type(e).__name__}: {e}"
        )

        print(
            "Grad-CAM failed:"
        )

        print(error)

        return (
            "FAILED",
            None,
            error
        )

    finally:

        gc.collect()


# ============================================================
# PIPELINE
# ============================================================

def run_pipeline(
    image_dir,
    output_dir,
    model_path=None,
    conf_threshold=DEFAULT_CONFIDENCE,
    use_roi=False,
    tta=False
):

    """
    Run complete BeltGuard inference pipeline.

    Grad-CAM is disabled by default to keep memory usage
    low enough for Render's 512 MB instance.
    """

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    print(
        "=========================================="
    )

    print(
        "BeltGuard Pipeline Starting"
    )

    print(
        "=========================================="
    )

    print(
        f"Grad-CAM enabled: {ENABLE_GRADCAM}"
    )

    print(
        f"Device: {DEVICE}"
    )

    # ========================================================
    # MODEL PATH
    # ========================================================

    if model_path is None:

        script_dir = (
            Path(__file__).resolve().parent
        )

        candidates = [

            script_dir
            / "runs"
            / "train"
            / "belt_damage_improved"
            / "weights"
            / "best.pt",

            script_dir
            / "model_weights"
            / "best.pt",

            script_dir
            / "runs"
            / "train"
            / "belt_damage_v7"
            / "weights"
            / "best.pt",

            script_dir
            / "runs"
            / "train"
            / "belt_damage_v6"
            / "weights"
            / "best.pt",

            script_dir
            / "runs"
            / "train"
            / "belt_damage_v5"
            / "weights"
            / "best.pt"
        ]

        for candidate in candidates:

            if candidate.exists():

                model_path = str(
                    candidate
                )

                break

    if model_path is None:

        print(
            "ERROR: No model weights found."
        )

        return None

    # ========================================================
    # LOAD YOLO
    # ========================================================

    print(
        f"Loading model: {model_path}"
    )

    model = YOLO(
        model_path
    )

    # Force CPU.
    try:

        model.to(DEVICE)

    except Exception:

        pass

    # ========================================================
    # ANOMALY MODEL
    # ========================================================

    anomaly_model = None
    anomaly_device = torch.device(
        "cpu"
    )

    # Only load anomaly model when ROI/anomaly processing
    # is actually needed.
    if use_roi:

        anomaly_model, anomaly_device = (
            load_anomaly_model()
        )

    # ========================================================
    # IMAGE LIST
    # ========================================================

    image_files = sorted(
        [
            f
            for f in os.listdir(
                image_dir
            )
            if f.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )
            )
        ]
    )

    print(
        f"Processing {len(image_files)} "
        f"images from {image_dir}"
    )

    labels_dir_default = os.path.join(
        Path(__file__).resolve().parent,
        "training_data",
        "labels"
    )

    total_detections = 0

    images_with_detections = 0

    anomalous_images = 0

    # ========================================================
    # IMAGE LOOP
    # ========================================================

    for img_file in image_files:

        print(
            f"\nProcessing: {img_file}"
        )

        img_path = os.path.join(
            image_dir,
            img_file
        )

        base_name = os.path.splitext(
            img_file
        )[0]

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        img_cv = cv2.imread(
            img_path
        )

        if img_cv is None:

            print(
                f"WARNING: Could not read "
                f"{img_file}"
            )

            continue

        img_h, img_w = (
            img_cv.shape[:2]
        )

        # ----------------------------------------------------
        # GRAD-CAM
        # ----------------------------------------------------

        gradcam_output_path = os.path.join(
            output_dir,
            base_name
            + "_gradcam.jpg"
        )

        (
            gradcam_status,
            gradcam_file,
            gradcam_error
        ) = run_gradcam_if_enabled(
            model=model,
            image_path=img_path,
            output_path=gradcam_output_path
        )

        # ----------------------------------------------------
        # ROI LABEL
        # ----------------------------------------------------

        anomaly_label_path = os.path.join(
            labels_dir_default,
            base_name + ".txt"
        )

        # ----------------------------------------------------
        # ROI ANOMALY
        # ----------------------------------------------------

        roi_anomaly_score = None

        anomaly_status = "UNAVAILABLE"

        if (
            use_roi
            and os.path.exists(
                anomaly_label_path
            )
        ):

            (
                roi_anomaly_score,
                anomaly_status
            ) = calculate_roi_anomaly_score(
                anomaly_model,
                anomaly_device,
                img_path,
                anomaly_label_path
            )

        # ----------------------------------------------------
        # BELT ROI MASK
        # ----------------------------------------------------

        belt_mask = None

        if use_roi:

            label_file = (
                base_name
                + ".txt"
            )

            search_dirs = [

                os.path.join(
                    image_dir,
                    "..",
                    "labels"
                ),

                labels_dir_default,

                os.path.dirname(
                    img_path
                )
            ]

            for search_dir in search_dirs:

                label_path = os.path.join(
                    search_dir,
                    label_file
                )

                if os.path.exists(
                    label_path
                ):

                    polygon = load_belt_roi(
                        label_path,
                        img_w,
                        img_h
                    )

                    if polygon is not None:

                        belt_mask = (
                            create_belt_mask(
                                img_w,
                                img_h,
                                polygon
                            )
                        )

                    break

        # ----------------------------------------------------
        # MASK
        # ----------------------------------------------------

        if belt_mask is not None:

            masked_cv = mask_image(
                img_cv,
                belt_mask
            )

        else:

            masked_cv = img_cv

        # ====================================================
        # YOLO INFERENCE
        # ====================================================

        all_detections = []

        try:

            results = model.predict(
                source=masked_cv,
                conf=conf_threshold,
                imgsz=640,
                device=DEVICE,
                verbose=False,
                stream=True
            )

            for r in results:

                if r.boxes is None:
                    continue

                for box in r.boxes:

                    x1, y1, x2, y2 = (
                        box.xyxy[0].tolist()
                    )

                    cls = int(
                        box.cls[0].item()
                    )

                    conf = float(
                        box.conf[0].item()
                    )

                    bbox = [
                        round(x1),
                        round(y1),
                        round(x2),
                        round(y2)
                    ]

                    severity_info = (
                        calculate_severity(
                            bbox,
                            img_w,
                            img_h
                        )
                    )

                    maintenance = (
                        calculate_maintenance_priority(
                            severity_info[
                                "severity"
                            ],
                            conf
                        )
                    )

                    all_detections.append(
                        {
                            "class": cls,

                            "class_name":
                                CLASS_NAMES.get(
                                    cls,
                                    f"cls{cls}"
                                ),

                            "confidence":
                                round(
                                    conf,
                                    4
                                ),

                            "bbox":
                                bbox,

                            "severity_score":
                                severity_info[
                                    "severity_score"
                                ],

                            "area_ratio":
                                severity_info[
                                    "area_ratio"
                                ],

                            "severity":
                                severity_info[
                                    "severity"
                                ],

                            "maintenance_priority":
                                maintenance
                        }
                    )

            del results

        except Exception as e:

            print(
                "WARNING: YOLO inference failed:"
            )

            print(e)

        # ====================================================
        # TTA
        # ====================================================

        if tta:

            # ------------------------------------------------
            # HORIZONTAL FLIP
            # ------------------------------------------------

            flipped = cv2.flip(
                masked_cv,
                1
            )

            try:

                results_flip = model.predict(
                    source=flipped,
                    conf=conf_threshold,
                    imgsz=640,
                    device=DEVICE,
                    verbose=False,
                    stream=True
                )

                for r in results_flip:

                    if r.boxes is None:
                        continue

                    for box in r.boxes:

                        x1, y1, x2, y2 = (
                            box.xyxy[0].tolist()
                        )

                        x1_new = (
                            img_w - x2
                        )

                        x2_new = (
                            img_w - x1
                        )

                        cls = int(
                            box.cls[0].item()
                        )

                        conf = float(
                            box.conf[0].item()
                        )

                        bbox = [
                            round(x1_new),
                            round(y1),
                            round(x2_new),
                            round(y2)
                        ]

                        severity_info = (
                            calculate_severity(
                                bbox,
                                img_w,
                                img_h
                            )
                        )

                        maintenance = (
                            calculate_maintenance_priority(
                                severity_info[
                                    "severity"
                                ],
                                conf
                            )
                        )

                        all_detections.append(
                            {
                                "class": cls,

                                "class_name":
                                    CLASS_NAMES.get(
                                        cls,
                                        f"cls{cls}"
                                    ),

                                "confidence":
                                    round(
                                        conf,
                                        4
                                    ),

                                "bbox":
                                    bbox,

                                "severity_score":
                                    severity_info[
                                        "severity_score"
                                    ],

                                "area_ratio":
                                    severity_info[
                                        "area_ratio"
                                    ],

                                "severity":
                                    severity_info[
                                        "severity"
                                    ],

                                "maintenance_priority":
                                    maintenance
                            }
                        )

                del results_flip

            except Exception as e:

                print(
                    "WARNING: Horizontal TTA failed:"
                )

                print(e)

            del flipped

            # ------------------------------------------------
            # VERTICAL + HORIZONTAL FLIP
            # ------------------------------------------------

            flipped_v = cv2.flip(
                masked_cv,
                -1
            )

            try:

                results_flip_v = model.predict(
                    source=flipped_v,
                    conf=conf_threshold,
                    imgsz=640,
                    device=DEVICE,
                    verbose=False,
                    stream=True
                )

                for r in results_flip_v:

                    if r.boxes is None:
                        continue

                    for box in r.boxes:

                        x1, y1, x2, y2 = (
                            box.xyxy[0].tolist()
                        )

                        x1_new = (
                            img_w - x2
                        )

                        x2_new = (
                            img_w - x1
                        )

                        y1_new = (
                            img_h - y2
                        )

                        y2_new = (
                            img_h - y1
                        )

                        cls = int(
                            box.cls[0].item()
                        )

                        conf = float(
                            box.conf[0].item()
                        )

                        bbox = [
                            round(x1_new),
                            round(y1_new),
                            round(x2_new),
                            round(y2_new)
                        ]

                        severity_info = (
                            calculate_severity(
                                bbox,
                                img_w,
                                img_h
                            )
                        )

                        maintenance = (
                            calculate_maintenance_priority(
                                severity_info[
                                    "severity"
                                ],
                                conf
                            )
                        )

                        all_detections.append(
                            {
                                "class": cls,

                                "class_name":
                                    CLASS_NAMES.get(
                                        cls,
                                        f"cls{cls}"
                                    ),

                                "confidence":
                                    round(
                                        conf,
                                        4
                                    ),

                                "bbox":
                                    bbox,

                                "severity_score":
                                    severity_info[
                                        "severity_score"
                                    ],

                                "area_ratio":
                                    severity_info[
                                        "area_ratio"
                                    ],

                                "severity":
                                    severity_info[
                                        "severity"
                                    ],

                                "maintenance_priority":
                                    maintenance
                            }
                        )

                del results_flip_v

            except Exception as e:

                print(
                    "WARNING: Vertical TTA failed:"
                )

                print(e)

            del flipped_v

            # ------------------------------------------------
            # NMS
            # ------------------------------------------------

            all_detections = (
                nms_detections(
                    all_detections,
                    iou_thresh=0.5
                )
            )

        # ====================================================
        # ROI FILTER
        # ====================================================

        if belt_mask is not None:

            all_detections = (
                filter_detections_in_belt(
                    all_detections,
                    belt_mask
                )
            )

        # ====================================================
        # ANOMALY COUNTER
        # ====================================================

        if (
            anomaly_status
            == "ANOMALOUS"
        ):

            anomalous_images += 1

        # ====================================================
        # HEALTH SCORE
        # ====================================================

        health_info = (
            calculate_health_score(
                all_detections,
                roi_anomaly_score,
                ANOMALY_THRESHOLD
            )
        )

        # ====================================================
        # SORT
        # ====================================================

        all_detections.sort(
            key=lambda d:
                d["confidence"],
            reverse=True
        )

        # ====================================================
        # ANNOTATED IMAGE
        # ====================================================

        # Convert only when needed.
        img_pil = Image.fromarray(
            cv2.cvtColor(
                img_cv,
                cv2.COLOR_BGR2RGB
            )
        )

        annotated_img = (
            draw_detections(
                img_pil,
                all_detections
            )
        )

        out_img_path = os.path.join(
            output_dir,
            base_name + ".jpg"
        )

        if annotated_img.mode != "RGB":

            annotated_img = (
                annotated_img.convert(
                    "RGB"
                )
            )

        annotated_img.save(
            out_img_path,
            quality=90,
            optimize=True
        )

        # ====================================================
        # JSON
        # ====================================================

        det_json = {

            "image":
                img_file,

            "image_width":
                img_w,

            "image_height":
                img_h,

            "model":
                model_path,

            "confidence_threshold":
                conf_threshold,

            "anomaly_detection": {

                "model":
                    str(
                        ANOMALY_MODEL_PATH
                    ),

                "threshold":
                    ANOMALY_THRESHOLD,

                "anomaly_score":
                    roi_anomaly_score,

                "status":
                    anomaly_status,

                "method":
                    "ROI convolutional autoencoder"
            },

            "health": {

                "score":
                    health_info[
                        "score"
                    ],

                "status":
                    health_info[
                        "status"
                    ],

                "defect_condition":
                    health_info[
                        "defect_condition"
                    ],

                "severity_condition":
                    health_info[
                        "severity_condition"
                    ],

                "anomaly_condition":
                    health_info[
                        "anomaly_condition"
                    ],

                "defect_weight":
                    health_info[
                        "defect_weight"
                    ],

                "severity_weight":
                    health_info[
                        "severity_weight"
                    ],

                "anomaly_weight":
                    health_info[
                        "anomaly_weight"
                    ],

                "method":
                    health_info[
                        "method"
                    ]
            },

            "explainability": {

                "method":
                    "Grad-CAM",

                "status":
                    gradcam_status,

                "output":
                    (
                        os.path.basename(
                            gradcam_file
                        )
                        if gradcam_file
                        else None
                    ),

                "error":
                    gradcam_error
            },

            "detections":
                all_detections
        }

        out_json_path = os.path.join(
            output_dir,
            base_name + ".json"
        )

        with open(
            out_json_path,
            "w"
        ) as f:

            json.dump(
                det_json,
                f,
                indent=2
            )

        # ====================================================
        # COUNTERS
        # ====================================================

        total_detections += (
            len(all_detections)
        )

        if all_detections:

            images_with_detections += 1

        # ====================================================
        # MEMORY CLEANUP
        # ====================================================

        del img_cv
        del masked_cv
        del img_pil
        del annotated_img

        if belt_mask is not None:

            del belt_mask

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    # ========================================================
    # FINAL CLEANUP
    # ========================================================

    if anomaly_model is not None:

        del anomaly_model

    del model

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "BeltGuard Inference Complete"
    )

    print(
        "=========================================="
    )

    print(
        f"Images processed: "
        f"{len(image_files)}"
    )

    print(
        f"Total detections: "
        f"{total_detections}"
    )

    print(
        f"Images with detections: "
        f"{images_with_detections}"
    )

    print(
        f"Anomalous belt images: "
        f"{anomalous_images}"
    )

    print(
        f"ROI anomaly threshold: "
        f"{ANOMALY_THRESHOLD}"
    )

    print(
        f"Confidence threshold: "
        f"{conf_threshold}"
    )

    print(
        f"Grad-CAM enabled: "
        f"{ENABLE_GRADCAM}"
    )

    print(
        f"Output saved to: "
        f"{output_dir}"
    )

    return {
        "images_processed":
            len(image_files),

        "total_detections":
            total_detections,

        "images_with_detections":
            images_with_detections,

        "anomalous_images":
            anomalous_images,

        "gradcam_enabled":
            ENABLE_GRADCAM,

        "output_dir":
            output_dir
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=
        "BeltGuard conveyor belt "
        "monitoring pipeline"
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Path to image folder"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output folder"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to YOLO model"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="Confidence threshold"
    )

    parser.add_argument(
        "--roi",
        action="store_true",
        help="Enable belt ROI masking and anomaly detection"
    )

    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable test-time augmentation"
    )

    args = parser.parse_args()

    # ========================================================
    # VALIDATE IMAGE DIRECTORY
    # ========================================================

    if not os.path.isdir(
        args.image_dir
    ):

        print(
            f"ERROR: Image directory not found: "
            f"{args.image_dir}"
        )

        return

    # ========================================================
    # RUN
    # ========================================================

    run_pipeline(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        model_path=args.model,
        conf_threshold=args.conf,
        use_roi=args.roi,
        tta=args.tta
    )


if __name__ == "__main__":

    main()