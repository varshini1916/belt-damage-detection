import cv2
import numpy as np
import torch

from ultralytics import YOLO

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = (
    "runs/train/"
    "belt_damage_improved/"
    "weights/"
    "best.pt"
)

IMAGE_PATH = "test.jpg"

OUTPUT_PATH = "gradcam_result.jpg"

CAM_SIZE = 640


# =========================================================
# YOLO GRAD-CAM TARGET
# =========================================================

class YOLOGradCAMTarget:

    def __call__(self, model_output):

        if isinstance(
            model_output,
            (tuple, list)
        ):

            model_output = model_output[0]

        if not torch.is_tensor(
            model_output
        ):

            raise TypeError(
                f"Unexpected model output type: "
                f"{type(model_output)}"
            )

        return model_output.reshape(
            -1
        ).sum()


# =========================================================
# LOAD YOLO
# =========================================================

print(
    "Loading YOLO model..."
)

model = YOLO(
    MODEL_PATH
)

print(
    "Model loaded successfully."
)


# =========================================================
# LOAD IMAGE
# =========================================================

image = cv2.imread(
    IMAGE_PATH
)

if image is None:

    raise FileNotFoundError(
        f"Could not find image: "
        f"{IMAGE_PATH}"
    )

print(
    f"Image loaded: {IMAGE_PATH}"
)


# =========================================================
# RUN YOLO PREDICTION
# =========================================================

print(
    "Running YOLO prediction..."
)

results = model.predict(
    source=IMAGE_PATH,
    conf=0.24,
    verbose=False
)

detections = results[0].boxes

print(
    f"Detections found: "
    f"{len(detections)}"
)


# =========================================================
# CONVERT IMAGE TO RGB
# =========================================================

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

original_height, original_width = (
    image.shape[:2]
)

print(
    f"Original image size: "
    f"{original_width}x{original_height}"
)


# =========================================================
# RESIZE FOR GRAD-CAM
# =========================================================

resized_rgb = cv2.resize(
    image_rgb,
    (CAM_SIZE, CAM_SIZE),
    interpolation=cv2.INTER_AREA
)

rgb_float = (
    resized_rgb.astype(
        np.float32
    ) / 255.0
)


# =========================================================
# CREATE INPUT TENSOR
# =========================================================

input_tensor = torch.from_numpy(
    rgb_float.transpose(2, 0, 1)
).unsqueeze(0).float()


# =========================================================
# MODEL DEVICE
# =========================================================

device = next(
    model.model.parameters()
).device

input_tensor = input_tensor.to(
    device
)


# =========================================================
# ENABLE GRADIENTS
# =========================================================

model.model.train()

for parameter in model.model.parameters():

    parameter.requires_grad_(True)


# =========================================================
# FIND CONVOLUTIONAL LAYERS
# =========================================================

conv_layers = [

    layer

    for layer in model.model.modules()

    if isinstance(
        layer,
        torch.nn.Conv2d
    )
]

print(
    f"Found {len(conv_layers)} Conv2d layers."
)


if not conv_layers:

    raise RuntimeError(
        "No Conv2d layers found."
    )


# =========================================================
# SELECT LAST 3x3 CONVOLUTION
# =========================================================

target_layer = None

for layer in reversed(
    conv_layers
):

    if layer.kernel_size == (3, 3):

        target_layer = layer

        break


if target_layer is None:

    target_layer = conv_layers[-1]


print(
    "Selected Grad-CAM++ layer:"
)

print(
    target_layer
)


# =========================================================
# INITIALIZE GRAD-CAM++
# =========================================================

cam = GradCAMPlusPlus(

    model=model.model,

    target_layers=[
        target_layer
    ]
)


targets = [
    YOLOGradCAMTarget()
]


# =========================================================
# GENERATE GRAD-CAM++
# =========================================================

print(
    "Generating Grad-CAM++ heatmap..."
)

with torch.enable_grad():

    grayscale_cam = cam(

        input_tensor=input_tensor,

        targets=targets
    )[0]


# =========================================================
# CLEAN HEATMAP
# =========================================================

grayscale_cam = np.asarray(
    grayscale_cam,
    dtype=np.float32
)

grayscale_cam = np.nan_to_num(
    grayscale_cam,
    nan=0.0,
    posinf=1.0,
    neginf=0.0
)

grayscale_cam = np.clip(
    grayscale_cam,
    0.0,
    1.0
)


# =========================================================
# RESIZE CAM TO 640x640
# =========================================================

grayscale_cam = cv2.resize(

    grayscale_cam,

    (CAM_SIZE, CAM_SIZE),

    interpolation=cv2.INTER_LINEAR
)


# =========================================================
# MEMORY-SAFE VISUALIZATION
#
# IMPORTANT:
# DO NOT use the original 3840x2160 image here.
# =========================================================

visualization_small = show_cam_on_image(

    rgb_float,

    grayscale_cam,

    use_rgb=True
)


# =========================================================
# RESIZE FINAL RESULT
# =========================================================

visualization = cv2.resize(

    visualization_small,

    (
        original_width,
        original_height
    ),

    interpolation=cv2.INTER_LINEAR
)


# =========================================================
# RGB -> BGR
# =========================================================

visualization = cv2.cvtColor(

    visualization,

    cv2.COLOR_RGB2BGR
)


# =========================================================
# SAVE RESULT
# =========================================================

success = cv2.imwrite(

    OUTPUT_PATH,

    visualization
)


if not success:

    raise RuntimeError(
        f"Could not save output: "
        f"{OUTPUT_PATH}"
    )


# =========================================================
# FINISHED
# =========================================================

print()
print(
    "========================================"
)

print(
    "Grad-CAM++ completed successfully!"
)

print(
    "========================================"
)

print(
    f"Output saved to: {OUTPUT_PATH}"
)