import os
import cv2
import time

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    "runs",
    "train",
    "belt_damage_improved",
    "weights",
    "best.pt"
)

DEFAULT_CONFIDENCE = 0.24
DEFAULT_IMGSZ = 640

# A defect must remain visible for this many
# processed frames before being considered persistent.
PERSISTENCE_FRAMES = 5

# IoU threshold used to associate detections
# between consecutive frames.
IOU_THRESHOLD = 0.30

# ------------------------------------------------------------
# BELT ROI
# ------------------------------------------------------------
# Normalized polygon covering the conveyor belt.
#
# These coordinates are fractions of:
# width  -> 0.0 to 1.0
# height -> 0.0 to 1.0
#
# The upper part of the frame is excluded so objects,
# machinery and background areas are not treated as
# conveyor-belt defects.
#
# Shape:
#
#       top-left ----------------
#          \                      \
#           \                      \
#            ---------------------- top-right
#            |                     |
#            |     CONVEYOR        |
#            |       BELT          |
#            |                     |
#            |_____________________|
#
BELT_ROI_POINTS = [
    (0.00, 0.28),
    (0.42, 0.27),
    (1.00, 0.36),
    (1.00, 1.00),
    (0.00, 1.00)
]


# ============================================================
# MODEL
# ============================================================

def load_model(model_path=MODEL_PATH):

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"YOLO model not found: {model_path}"
        )

    return YOLO(model_path)


# ============================================================
# BELT ROI FUNCTIONS
# ============================================================

def get_belt_roi_points(image_width, image_height):
    """
    Convert normalized ROI coordinates into pixel coordinates.
    """

    points = []

    for x_norm, y_norm in BELT_ROI_POINTS:

        x = int(x_norm * image_width)
        y = int(y_norm * image_height)

        points.append([x, y])

    return points


def is_point_inside_belt(x, y, roi_points):
    """
    Check whether a point lies inside the conveyor-belt ROI.
    """

    contour = cv2.convexHull(
        cv2.UMat(
            __import__("numpy").array(roi_points, dtype="int32")
        ).get()
    )

    result = cv2.pointPolygonTest(
        contour,
        (float(x), float(y)),
        False
    )

    return result >= 0


def detection_inside_belt(bbox, roi_points):
    """
    Determine whether the center of a detection lies inside
    the conveyor-belt ROI.
    """

    x1, y1, x2, y2 = bbox

    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    return is_point_inside_belt(
        center_x,
        center_y,
        roi_points
    )


# ============================================================
# IOU
# ============================================================

def calculate_iou(box_a, box_b):
    """
    Calculate Intersection over Union between
    two bounding boxes.

    Box format:
    [x1, y1, x2, y2]
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)

    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0,
        intersection_x2 - intersection_x1
    )

    intersection_height = max(
        0,
        intersection_y2 - intersection_y1
    )

    intersection_area = (
        intersection_width *
        intersection_height
    )

    area_a = (
        max(0, ax2 - ax1) *
        max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1) *
        max(0, by2 - by1)
    )

    union_area = (
        area_a +
        area_b -
        intersection_area
    )

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


# ============================================================
# SEVERITY
# ============================================================

def calculate_severity(
    bbox,
    image_width,
    image_height
):
    """
    Visual severity proxy based on bounding-box area.

    This is NOT a learned severity classifier.

    The scale is deliberately less aggressive than the
    original version so small detections are not automatically
    classified as HIGH severity.
    """

    x1, y1, x2, y2 = bbox

    box_width = max(0, x2 - x1)
    box_height = max(0, y2 - y1)

    box_area = box_width * box_height

    image_area = image_width * image_height

    if image_area <= 0:
        return "Low", 0.0

    area_ratio = box_area / image_area

    # Convert area ratio into a 0-100 visual severity score.
    #
    # Example:
    # 0.5% image area -> approximately 10
    # 1.25%          -> approximately 25
    # 3.0%           -> approximately 60
    # 5.0%+          -> 100
    #
    severity_score = min(
        area_ratio * 2000,
        100
    )

    if severity_score < 25:
        severity = "Low"

    elif severity_score < 60:
        severity = "Medium"

    else:
        severity = "High"

    return severity, severity_score


# ============================================================
# TRACK MANAGEMENT
# ============================================================

class DefectTracker:

    def __init__(
        self,
        iou_threshold=IOU_THRESHOLD,
        persistence_frames=PERSISTENCE_FRAMES
    ):

        self.iou_threshold = iou_threshold
        self.persistence_frames = persistence_frames

        self.tracks = {}
        self.next_track_id = 1

    def update(self, detections):

        matched_track_ids = set()

        for detection in detections:

            best_track_id = None
            best_iou = 0.0

            for track_id, track in self.tracks.items():

                if track_id in matched_track_ids:
                    continue

                if (
                    track["class_name"]
                    != detection["class_name"]
                ):
                    continue

                iou = calculate_iou(
                    track["bbox"],
                    detection["bbox"]
                )

                if iou > best_iou:

                    best_iou = iou
                    best_track_id = track_id

            # ------------------------------------------------
            # Existing track
            # ------------------------------------------------

            if (
                best_track_id is not None
                and best_iou >= self.iou_threshold
            ):

                track = self.tracks[best_track_id]

                track["bbox"] = detection["bbox"]
                track["confidence"] = detection["confidence"]
                track["severity"] = detection["severity"]
                track["severity_score"] = detection[
                    "severity_score"
                ]

                track["frames_seen"] += 1
                track["missed_frames"] = 0

                matched_track_ids.add(best_track_id)

                detection["track_id"] = best_track_id

            # ------------------------------------------------
            # New track
            # ------------------------------------------------

            else:

                track_id = self.next_track_id

                self.next_track_id += 1

                self.tracks[track_id] = {

                    "track_id": track_id,

                    "class_name":
                        detection["class_name"],

                    "bbox":
                        detection["bbox"],

                    "confidence":
                        detection["confidence"],

                    "severity":
                        detection["severity"],

                    "severity_score":
                        detection["severity_score"],

                    "frames_seen": 1,

                    "missed_frames": 0
                }

                matched_track_ids.add(track_id)

                detection["track_id"] = track_id

        # ----------------------------------------------------
        # Increase missed-frame count
        # ----------------------------------------------------

        for track_id, track in list(
            self.tracks.items()
        ):

            if track_id not in matched_track_ids:

                track["missed_frames"] += 1

                if track["missed_frames"] > 10:

                    del self.tracks[track_id]

        # ----------------------------------------------------
        # Add persistence status
        # ----------------------------------------------------

        for detection in detections:

            track_id = detection["track_id"]

            track = self.tracks[track_id]

            detection["frames_seen"] = (
                track["frames_seen"]
            )

            detection["persistent"] = (
                track["frames_seen"]
                >= self.persistence_frames
            )

        return detections


# ============================================================
# DRAW DETECTION
# ============================================================

def draw_detection(
    frame,
    detection
):

    x1, y1, x2, y2 = detection["bbox"]

    class_name = detection["class_name"]
    confidence = detection["confidence"]
    severity = detection["severity"]

    track_id = detection["track_id"]
    persistent = detection["persistent"]
    frames_seen = detection["frames_seen"]

    # --------------------------------------------------------
    # Severity-based visual style
    # --------------------------------------------------------

    if severity == "High":

        color = (0, 0, 255)

    elif severity == "Medium":

        color = (0, 165, 255)

    else:

        color = (0, 200, 0)

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        3
    )

    # --------------------------------------------------------
    # Persistence indicator
    # --------------------------------------------------------

    if persistent:

        persistence_text = (
            f"PERSISTENT | {frames_seen} frames"
        )

    else:

        persistence_text = (
            f"TRACK {track_id}"
        )

    label = (
        f"{class_name} | "
        f"{confidence:.0%} | "
        f"{severity}"
    )

    label2 = persistence_text

    # --------------------------------------------------------
    # Label background
    # --------------------------------------------------------

    label_width = max(
        300,
        len(label) * 12
    )

    cv2.rectangle(
        frame,
        (x1, max(0, y1 - 60)),
        (
            x1 + label_width,
            y1
        ),
        color,
        -1
    )

    # --------------------------------------------------------
    # Draw labels
    # --------------------------------------------------------

    cv2.putText(
        frame,
        label,
        (x1 + 5, max(20, y1 - 35)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        label2,
        (x1 + 5, max(42, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


# ============================================================
# DRAW BELT ROI
# ============================================================

def draw_belt_roi(
    frame,
    roi_points
):
    """
    Draw the calibrated conveyor-belt ROI.
    """

    import numpy as np

    points = np.array(
        roi_points,
        dtype=np.int32
    )

    # Thin outline so it does not hide the belt.
    cv2.polylines(
        frame,
        [points],
        True,
        (255, 255, 0),
        2
    )

    # Label
    x = 20
    y = max(95, int(frame.shape[0] * 0.10))

    cv2.putText(
        frame,
        "BELT ROI",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        2,
        cv2.LINE_AA
    )


# ============================================================
# PROCESS VIDEO
# ============================================================

def process_video(
    input_video,
    output_video,
    confidence=DEFAULT_CONFIDENCE,
    imgsz=DEFAULT_IMGSZ
):

    model = load_model()

    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():

        raise RuntimeError(
            "Unable to open the uploaded video."
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25.0

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = (
        total_frames / fps
        if fps > 0
        else 0
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    output_dir = os.path.dirname(
        output_video
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Output writer
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_video,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            "Unable to create output video."
        )

    # --------------------------------------------------------
    # Belt ROI
    # --------------------------------------------------------

    roi_points = get_belt_roi_points(
        width,
        height
    )

    tracker = DefectTracker()

    frame_number = 0

    total_detections = 0
    high_severity_detections = 0
    medium_severity_detections = 0
    low_severity_detections = 0

    persistent_events = 0

    seen_persistent_tracks = set()

    start_time = time.time()

    # ========================================================
    # VIDEO LOOP
    # ========================================================

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # ----------------------------------------------------
        # YOLO inference
        # ----------------------------------------------------

        results = model.predict(
            source=frame,
            conf=confidence,
            imgsz=imgsz,
            verbose=False
        )

        result = results[0]

        detections = []

        if result.boxes is not None:

            for box in result.boxes:

                xyxy = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )

                x1, y1, x2, y2 = map(
                    int,
                    xyxy
                )

                confidence_value = float(
                    box.conf[0]
                    .cpu()
                    .item()
                )

                class_id = int(
                    box.cls[0]
                    .cpu()
                    .item()
                )

                class_name = model.names[
                    class_id
                ]

                bbox = [
                    x1,
                    y1,
                    x2,
                    y2
                ]

                # ------------------------------------------------
                # BELT ROI FILTER
                # ------------------------------------------------
                #
                # Only accept detections whose center lies
                # inside the conveyor-belt region.
                #
                if not detection_inside_belt(
                    bbox,
                    roi_points
                ):
                    continue

                severity, severity_score = (
                    calculate_severity(
                        bbox,
                        width,
                        height
                    )
                )

                detections.append({

                    "bbox": bbox,

                    "class_name":
                        class_name,

                    "confidence":
                        confidence_value,

                    "severity":
                        severity,

                    "severity_score":
                        severity_score
                })

        # ----------------------------------------------------
        # Temporal tracking
        # ----------------------------------------------------

        detections = tracker.update(
            detections
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_detections += len(
            detections
        )

        high_count = sum(
            1
            for d in detections
            if d["severity"] == "High"
        )

        medium_count = sum(
            1
            for d in detections
            if d["severity"] == "Medium"
        )

        low_count = sum(
            1
            for d in detections
            if d["severity"] == "Low"
        )

        high_severity_detections += (
            high_count
        )

        medium_severity_detections += (
            medium_count
        )

        low_severity_detections += (
            low_count
        )

        # ----------------------------------------------------
        # Persistent defect events
        # ----------------------------------------------------

        for detection in detections:

            if (
                detection["persistent"]
                and detection["severity"]
                == "High"
            ):

                track_id = detection[
                    "track_id"
                ]

                if (
                    track_id
                    not in seen_persistent_tracks
                ):

                    persistent_events += 1

                    seen_persistent_tracks.add(
                        track_id
                    )

        # ----------------------------------------------------
        # Draw detections
        # ----------------------------------------------------

        for detection in detections:

            draw_detection(
                frame,
                detection
            )

        # ----------------------------------------------------
        # Draw belt ROI
        # ----------------------------------------------------

        draw_belt_roi(
            frame,
            roi_points
        )

        # ----------------------------------------------------
        # Live frame statistics
        # ----------------------------------------------------

        elapsed = (
            time.time() - start_time
        )

        processing_fps = (
            frame_number / elapsed
            if elapsed > 0
            else 0
        )

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (0, 0),
            (width, 75),
            (15, 23, 42),
            -1
        )

        cv2.putText(
            frame,
            "BELTGUARD | CONTINUOUS INSPECTION",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        status_text = (
            f"Frame: {frame_number} | "
            f"Defects: {len(detections)} | "
            f"FPS: {processing_fps:.1f}"
        )

        cv2.putText(
            frame,
            status_text,
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 230, 240),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Alert
        # ----------------------------------------------------

        persistent_high = any(
            d["persistent"]
            and d["severity"] == "High"
            for d in detections
        )

        if persistent_high:

            alert_text = (
                "ALERT: PERSISTENT HIGH-SEVERITY DEFECT"
            )

            cv2.rectangle(
                frame,
                (0, height - 55),
                (width, height),
                (0, 0, 180),
                -1
            )

            cv2.putText(
                frame,
                alert_text,
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        # ----------------------------------------------------
        # Write output
        # ----------------------------------------------------

        writer.write(frame)

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    writer.release()

    processing_time = (
        time.time() - start_time
    )

    return {

        "total_frames":
            frame_number,

        "video_fps":
            fps,

        "width":
            width,

        "height":
            height,

        "duration_seconds":
            duration,

        "processing_time_seconds":
            processing_time,

        "total_detections":
            total_detections,

        "high_severity_detections":
            high_severity_detections,

        "medium_severity_detections":
            medium_severity_detections,

        "low_severity_detections":
            low_severity_detections,

        "persistent_events":
            persistent_events,

        "output_video":
            output_video
    }