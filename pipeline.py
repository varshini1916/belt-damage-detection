"""
BeltGuard — Memory Optimized Conveyor Belt Damage Detection Pipeline

Features:
1. Belt ROI masking
2. YOLO defect detection
3. Confidence filtering
4. Optional TTA
5. Visual severity estimation
6. Maintenance priority estimation
7. Annotated images
8. ROI autoencoder anomaly detection
9. Health score
10. Optional Grad-CAM explainability

IMPORTANT FOR RENDER:
- Designed for low-memory CPU deployment.
- Grad-CAM is disabled by default.
- YOLO inference uses a smaller input size.
- Large images are resized before inference.
- Only one image is processed at a time.
- Explicit memory cleanup is performed after every image.
"""

# ============================================================
# MEMORY SETTINGS
# ============================================================

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import gc
import json
import argparse

import cv2
import numpy as np
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


# Confidence threshold
DEFAULT_CONFIDENCE = 0.24


# ============================================================
# MEMORY OPTIMIZATION
# ============================================================

# Maximum dimension used for YOLO inference.
#
# This is intentionally smaller than your previous 640 setting.
# 416 significantly reduces inference memory.
DETECTION_SIZE = 416


# Maximum image dimension kept during processing.
#
# Very large 3840x2160 images consume a lot of RAM.
MAX_IMAGE_DIMENSION = 1280


# Render free instance = CPU
DEVICE = "cpu"


# ============================================================
# ROI AUTOENCODER
# ============================================================

ANOMALY_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "belt_roi_autoencoder.pt"
)

ANOMALY_THRESHOLD = 0.001486


# ============================================================
# GRAD-CAM
# ============================================================

# NEVER enable this on Render 512 MB.
ENABLE_GRADCAM = False


# Environment override.
#
# Keep this FALSE on Render.
#
# BELTGUARD_GRADCAM=true
#
# should only be used locally on a machine
# with sufficient RAM.

if os.getenv(
    "BELTGUARD_GRADCAM",
    "false"
).lower() == "true":

    ENABLE_GRADCAM = True


# ============================================================
# GENERAL MEMORY CLEANUP
# ============================================================

def cleanup_memory():

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# IMAGE RESIZING
# ============================================================

def resize_for_detection(
    image,
    max_dimension=MAX_IMAGE_DIMENSION
):
    """
    Resize large images before YOLO inference.

    Returns:
        resized_image
        scale_x
        scale_y
    """

    original_h, original_w = image.shape[:2]

    largest_dimension = max(
        original_w,
        original_h
    )

    if largest_dimension <= max_dimension:

        return (
            image,
            1.0,
            1.0
        )

    scale = (
        max_dimension
        / float(largest_dimension)
    )

    new_w = max(
        1,
        int(original_w * scale)
    )

    new_h = max(
        1,
        int(original_h * scale)
    )

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    scale_x = (
        original_w
        / float(new_w)
    )

    scale_y = (
        original_h
        / float(new_h)
    )

    return (
        resized,
        scale_x,
        scale_y
    )


# ============================================================
# ROI FUNCTIONS
# ============================================================

def load_belt_roi(
    label_path,
    img_w,
    img_h
):

    if not os.path.exists(label_path):

        return None

    try:

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

        pts = []

        for i in range(
            0,
            len(coords),
            2
        ):

            if i + 1 >= len(coords):

                break

            pts.append(
                (
                    int(
                        coords[i]
                        * img_w
                    ),
                    int(
                        coords[i + 1]
                        * img_h
                    )
                )
            )

        if len(pts) < 3:

            return None

        return np.array(
            pts,
            dtype=np.int32
        )

    except Exception as e:

        print(
            f"WARNING: ROI loading failed: {e}"
        )

        return None


# ============================================================

def create_belt_mask(
    img_w,
    img_h,
    polygon_pts
):

    mask = np.zeros(
        (
            img_h,
            img_w
        ),
        dtype=np.uint8
    )

    cv2.fillPoly(
        mask,
        [polygon_pts],
        255
    )

    return mask


# ============================================================

def mask_image(
    img,
    belt_mask
):

    masked = img.copy()

    masked[
        belt_mask == 0
    ] = (
        128,
        128,
        128
    )

    return masked


# ============================================================

def filter_detections_in_belt(
    detections,
    belt_mask
):

    if belt_mask is None:

        return detections

    h, w = belt_mask.shape

    filtered = []

    for det in detections:

        bbox = det["bbox"]

        cx = int(
            (
                bbox[0]
                + bbox[2]
            ) / 2
        )

        cy = int(
            (
                bbox[1]
                + bbox[3]
            ) / 2
        )

        cx = max(
            0,
            min(
                w - 1,
                cx
            )
        )

        cy = max(
            0,
            min(
                h - 1,
                cy
            )
        )

        if belt_mask[
            cy,
            cx
        ] > 0:

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

        return (
            None,
            torch.device("cpu")
        )

    device = torch.device("cpu")

    try:

        model = BeltAutoencoder().to(
            device
        )

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
            "ROI Autoencoder loaded."
        )

        return (
            model,
            device
        )

    except Exception as e:

        print(
            "WARNING: Could not load ROI Autoencoder:"
        )

        print(e)

        cleanup_memory()

        return (
            None,
            device
        )


# ============================================================

def calculate_roi_anomaly_score(
    model,
    device,
    image_path,
    label_path
):

    if model is None:

        return (
            None,
            "UNAVAILABLE"
        )

    tensor = None
    reconstructed = None
    roi = None

    try:

        roi = extract_belt_roi(
            image_path,
            label_path
        )

        if roi is None:

            return (
                None,
                "ROI_UNAVAILABLE"
            )

        if roi.size == 0:

            return (
                None,
                "ROI_UNAVAILABLE"
            )

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
            )
            / 255.0
        )

        roi = np.transpose(
            roi,
            (2, 0, 1)
        )

        tensor = torch.from_numpy(
            roi
        ).unsqueeze(0)

        tensor = tensor.to(
            device
        )

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

        if error >= ANOMALY_THRESHOLD:

            status = "ANOMALOUS"

        else:

            status = "NORMAL"

        return (
            round(
                error,
                8
            ),
            status
        )

    except Exception as e:

        print(
            "WARNING: Anomaly scoring failed:"
        )

        print(e)

        return (
            None,
            "ERROR"
        )

    finally:

        del roi

        if tensor is not None:

            del tensor

        if reconstructed is not None:

            del reconstructed

        cleanup_memory()


# ============================================================
# SEVERITY
# ============================================================

def calculate_severity(
    bbox,
    img_w,
    img_h
):

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

    if image_area <= 0:

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

        "severity_score":
            round(
                score,
                2
            ),

        "area_ratio":
            round(
                area_ratio,
                6
            ),

        "severity":
            severity
    }


# ============================================================
# HEALTH SCORE
# ============================================================

def calculate_health_score(
    detections,
    anomaly_score,
    anomaly_threshold
):

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

    health_score = (

        0.25
        * defect_condition

        +

        0.45
        * severity_condition

        +

        0.30
        * anomaly_condition
    )

    health_score = max(
        0.0,
        min(
            100.0,
            health_score
        )
    )

    if health_score >= 80:

        status = "HEALTHY"

    elif health_score >= 60:

        status = "WARNING"

    elif health_score >= 40:

        status = "DEGRADED"

    else:

        status = "CRITICAL"

    return {

        "score":
            round(
                health_score,
                2
            ),

        "status":
            status,

        "defect_condition":
            round(
                defect_condition,
                2
            ),

        "severity_condition":
            round(
                severity_condition,
                2
            ),

        "anomaly_condition":
            round(
                anomaly_condition,
                2
            ),

        "defect_weight":
            0.25,

        "severity_weight":
            0.45,

        "anomaly_weight":
            0.30,

        "method":
            "Normalized weighted visual condition score"
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

    detections = sorted(
        detections,
        key=lambda x:
            x["confidence"],
        reverse=True
    )

    keep = []

    for det in detections:

        box = det["bbox"]

        overlap = False

        for kept in keep:

            kb = kept["bbox"]

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

            area1 = (

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

            area2 = (

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
                area1
                + area2
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

            keep.append(
                det
            )

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
            20,
            min(
                36,
                int(
                    min(
                        img_pil.width,
                        img_pil.height
                    )
                    * 0.014
                )
            )
        )

    font = None

    # Linux / Render
    font_paths = [

        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",

        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",

        "C:/Windows/Fonts/arialbd.ttf"
    ]

    for font_path in font_paths:

        try:

            font = ImageFont.truetype(
                font_path,
                font_size
            )

            break

        except (
            IOError,
            OSError
        ):

            continue

    if font is None:

        font = ImageFont.load_default()

    for det in detections:

        x1, y1, x2, y2 = (
            det["bbox"]
        )

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
            2,
            int(
                min(
                    img_pil.width,
                    img_pil.height
                )
                * 0.002
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

        padding_x = 8
        padding_y = 5

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
                label_y + 3
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

    if not ENABLE_GRADCAM:

        return (
            "DISABLED",
            None,
            None
        )

    try:

        # IMPORTANT:
        # Import ONLY when Grad-CAM is actually used.

        from gradcam_utils import (
            generate_gradcam
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

        cleanup_memory()


# ============================================================
# YOLO DETECTION
# ============================================================

def run_yolo_detection(
    model,
    image,
    conf_threshold,
    scale_x,
    scale_y,
    original_width,
    original_height
):

    detections = []

    results = None

    try:

        results = model.predict(

            source=image,

            conf=conf_threshold,

            imgsz=DETECTION_SIZE,

            device=DEVICE,

            verbose=False,

            stream=False,

            batch=1,

            max_det=20
        )

        if not results:

            return []

        result = results[0]

        if (
            result.boxes is None
            or len(result.boxes) == 0
        ):

            return []

        boxes = result.boxes

        for index in range(
            len(boxes)
        ):

            box = boxes[index]

            xyxy = box.xyxy[
                0
            ].detach().cpu().numpy()

            cls = int(
                box.cls[
                    0
                ].detach().cpu().item()
            )

            confidence = float(
                box.conf[
                    0
                ].detach().cpu().item()
            )

            # Map coordinates back
            # to original image size.

            x1 = int(
                round(
                    xyxy[0]
                    * scale_x
                )
            )

            y1 = int(
                round(
                    xyxy[1]
                    * scale_y
                )
            )

            x2 = int(
                round(
                    xyxy[2]
                    * scale_x
                )
            )

            y2 = int(
                round(
                    xyxy[3]
                    * scale_y
                )
            )

            x1 = max(
                0,
                min(
                    original_width - 1,
                    x1
                )
            )

            y1 = max(
                0,
                min(
                    original_height - 1,
                    y1
                )
            )

            x2 = max(
                0,
                min(
                    original_width - 1,
                    x2
                )
            )

            y2 = max(
                0,
                min(
                    original_height - 1,
                    y2
                )
            )

            bbox = [
                x1,
                y1,
                x2,
                y2
            ]

            severity_info = (
                calculate_severity(
                    bbox,
                    original_width,
                    original_height
                )
            )

            maintenance = (
                calculate_maintenance_priority(
                    severity_info[
                        "severity"
                    ],
                    confidence
                )
            )

            detections.append({

                "class":
                    cls,

                "class_name":
                    CLASS_NAMES.get(
                        cls,
                        f"cls{cls}"
                    ),

                "confidence":
                    round(
                        confidence,
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
            })

        return detections

    finally:

        if results is not None:

            del results

        cleanup_memory()


# ============================================================
# TTA
# ============================================================

def run_tta(
    model,
    image,
    conf_threshold,
    scale_x,
    scale_y,
    original_width,
    original_height
):

    all_tta_detections = []

    # --------------------------------------------------------
    # Horizontal flip
    # --------------------------------------------------------

    flipped = cv2.flip(
        image,
        1
    )

    try:

        detections = run_yolo_detection(
            model,
            flipped,
            conf_threshold,
            scale_x,
            scale_y,
            original_width,
            original_height
        )

        for det in detections:

            x1, y1, x2, y2 = (
                det["bbox"]
            )

            det["bbox"] = [

                original_width - x2,

                y1,

                original_width - x1,

                y2
            ]

        all_tta_detections.extend(
            detections
        )

    except Exception as e:

        print(
            f"WARNING: Horizontal TTA failed: {e}"
        )

    del flipped

    cleanup_memory()

    # --------------------------------------------------------
    # Do NOT perform vertical flip on Render.
    #
    # It doubles inference memory again.
    # --------------------------------------------------------

    return all_tta_detections


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
        f"Device: {DEVICE}"
    )

    print(
        f"YOLO image size: {DETECTION_SIZE}"
    )

    print(
        f"Maximum image dimension: "
        f"{MAX_IMAGE_DIMENSION}"
    )

    print(
        f"Grad-CAM enabled: "
        f"{ENABLE_GRADCAM}"
    )

    print(
        f"TTA enabled: "
        f"{tta}"
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

    try:

        model.to(
            DEVICE
        )

    except Exception:

        pass

    # ========================================================
    # ANOMALY MODEL
    # ========================================================

    anomaly_model = None

    anomaly_device = torch.device(
        "cpu"
    )

    if use_roi:

        (
            anomaly_model,
            anomaly_device
        ) = load_anomaly_model()

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
        f"Processing "
        f"{len(image_files)} images."
    )

    labels_dir_default = (
        Path(__file__).resolve().parent
        / "training_data"
        / "labels"
    )

    labels_dir_default = str(
        labels_dir_default
    )

    total_detections = 0

    images_with_detections = 0

    anomalous_images = 0

    # ========================================================
    # IMAGE LOOP
    # ========================================================

    for image_index, img_file in enumerate(
        image_files,
        start=1
    ):

        print(
            "\n------------------------------------------"
        )

        print(
            f"Image {image_index}/"
            f"{len(image_files)}: "
            f"{img_file}"
        )

        print(
            "------------------------------------------"
        )

        img_path = os.path.join(
            image_dir,
            img_file
        )

        base_name = os.path.splitext(
            img_file
        )[0]

        img_cv = None
        resized_cv = None
        belt_mask = None
        masked_cv = None
        img_pil = None
        annotated_img = None

        try:

            # =================================================
            # READ IMAGE
            # =================================================

            img_cv = cv2.imread(
                img_path,
                cv2.IMREAD_COLOR
            )

            if img_cv is None:

                print(
                    f"WARNING: Could not read "
                    f"{img_file}"
                )

                continue

            original_height, original_width = (
                img_cv.shape[:2]
            )

            print(
                f"Original size: "
                f"{original_width}x"
                f"{original_height}"
            )

            # =================================================
            # RESIZE FOR DETECTION
            # =================================================

            (
                resized_cv,
                scale_x,
                scale_y
            ) = resize_for_detection(
                img_cv
            )

            resized_h, resized_w = (
                resized_cv.shape[:2]
            )

            print(
                f"Detection size: "
                f"{resized_w}x"
                f"{resized_h}"
            )

            # =================================================
            # ROI
            # =================================================

            anomaly_label_path = os.path.join(

                labels_dir_default,

                base_name
                + ".txt"
            )

            roi_anomaly_score = None

            anomaly_status = (
                "UNAVAILABLE"
            )

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

            # =================================================
            # BELT MASK
            # =================================================

            if use_roi:

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

                label_file = (
                    base_name
                    + ".txt"
                )

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

                            original_width,

                            original_height
                        )

                        if polygon is not None:

                            belt_mask = (
                                create_belt_mask(

                                    original_width,

                                    original_height,

                                    polygon
                                )
                            )

                        break

            # =================================================
            # CREATE RESIZED MASK
            # =================================================

            if belt_mask is not None:

                resized_mask = cv2.resize(

                    belt_mask,

                    (
                        resized_w,
                        resized_h
                    ),

                    interpolation=cv2.INTER_NEAREST
                )

                masked_cv = mask_image(

                    resized_cv,

                    resized_mask
                )

                del resized_mask

            else:

                masked_cv = resized_cv.copy()

            # =================================================
            # YOLO
            # =================================================

            print(
                "Running YOLO inference..."
            )

            all_detections = (
                run_yolo_detection(

                    model,

                    masked_cv,

                    conf_threshold,

                    scale_x,

                    scale_y,

                    original_width,

                    original_height
                )
            )

            # =================================================
            # OPTIONAL TTA
            # =================================================

            if tta:

                print(
                    "Running horizontal TTA..."
                )

                tta_detections = run_tta(

                    model,

                    masked_cv,

                    conf_threshold,

                    scale_x,

                    scale_y,

                    original_width,

                    original_height
                )

                all_detections.extend(
                    tta_detections
                )

                all_detections = (
                    nms_detections(
                        all_detections,
                        iou_thresh=0.5
                    )
                )

            # =================================================
            # ROI FILTER
            # =================================================

            if belt_mask is not None:

                all_detections = (
                    filter_detections_in_belt(

                        all_detections,

                        belt_mask
                    )
                )

            # =================================================
            # ANOMALY
            # =================================================

            if (
                anomaly_status
                == "ANOMALOUS"
            ):

                anomalous_images += 1

            # =================================================
            # HEALTH
            # =================================================

            health_info = (
                calculate_health_score(

                    all_detections,

                    roi_anomaly_score,

                    ANOMALY_THRESHOLD
                )
            )

            # =================================================
            # SORT
            # =================================================

            all_detections.sort(

                key=lambda d:
                    d["confidence"],

                reverse=True
            )

            # =================================================
            # ANNOTATED IMAGE
            # =================================================

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

                base_name
                + ".jpg"
            )

            if annotated_img.mode != "RGB":

                annotated_img = (
                    annotated_img.convert(
                        "RGB"
                    )
                )

            annotated_img.save(

                out_img_path,

                format="JPEG",

                quality=85,

                optimize=True
            )

            # =================================================
            # GRAD-CAM
            # =================================================

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

                model,

                img_path,

                gradcam_output_path
            )

            # =================================================
            # JSON
            # =================================================

            det_json = {

                "image":
                    img_file,

                "image_width":
                    original_width,

                "image_height":
                    original_height,

                "model":
                    model_path,

                "confidence_threshold":
                    conf_threshold,

                "inference_size":
                    DETECTION_SIZE,

                "health":
                    health_info,

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

                base_name
                + ".json"
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

            # =================================================
            # COUNTERS
            # =================================================

            total_detections += (
                len(
                    all_detections
                )
            )

            if all_detections:

                images_with_detections += 1

            # =================================================
            # LOG RESULT
            # =================================================

            print(
                f"Detections: "
                f"{len(all_detections)}"
            )

            print(
                f"Health score: "
                f"{health_info['score']}"
            )

            print(
                f"Health status: "
                f"{health_info['status']}"
            )

            print(
                f"Anomaly: "
                f"{anomaly_status}"
            )

            print(
                f"Output: "
                f"{out_img_path}"
            )

        except Exception as e:

            print(
                "\nWARNING: Image processing failed:"
            )

            print(
                f"{type(e).__name__}: {e}"
            )

        finally:

            # =================================================
            # IMPORTANT MEMORY CLEANUP
            # =================================================

            if img_cv is not None:

                del img_cv

            if resized_cv is not None:

                del resized_cv

            if masked_cv is not None:

                del masked_cv

            if belt_mask is not None:

                del belt_mask

            if img_pil is not None:

                del img_pil

            if annotated_img is not None:

                del annotated_img

            cleanup_memory()

            print(
                "Memory cleanup completed."
            )

    # ========================================================
    # FINAL CLEANUP
    # ========================================================

    if anomaly_model is not None:

        del anomaly_model

    del model

    cleanup_memory()

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
        f"YOLO inference size: "
        f"{DETECTION_SIZE}"
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

        help=
        "Path to image folder"
    )

    parser.add_argument(

        "--output_dir",

        type=str,

        required=True,

        help=
        "Output folder"
    )

    parser.add_argument(

        "--model",

        type=str,

        default=None,

        help=
        "Path to YOLO model"
    )

    parser.add_argument(

        "--conf",

        type=float,

        default=DEFAULT_CONFIDENCE,

        help=
        "Confidence threshold"
    )

    parser.add_argument(

        "--roi",

        action="store_true",

        help=
        "Enable belt ROI "
        "masking and anomaly detection"
    )

    parser.add_argument(

        "--tta",

        action="store_true",

        help=
        "Enable horizontal "
        "test-time augmentation"
    )

    args = parser.parse_args()

    # ========================================================
    # VALIDATE
    # ========================================================

    if not os.path.isdir(
        args.image_dir
    ):

        print(
            f"ERROR: Image directory "
            f"not found: "
            f"{args.image_dir}"
        )

        return

    # ========================================================
    # RUN
    # ========================================================

    run_pipeline(

        image_dir=
            args.image_dir,

        output_dir=
            args.output_dir,

        model_path=
            args.model,

        conf_threshold=
            args.conf,

        use_roi=
            args.roi,

        tta=
            args.tta
    )


# ============================================================

if __name__ == "__main__":

    main()