import cv2
import numpy as np
import torch
from pathlib import Path

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class YOLOGradCAMTarget:
    """
    Converts YOLO model output into a scalar target
    that Grad-CAM can use for gradient computation.
    """

    def __call__(self, model_output):

        # Handle tuple/list outputs
        if isinstance(model_output, (tuple, list)):
            model_output = model_output[0]

        # Make sure output is a tensor
        if not torch.is_tensor(model_output):
            raise TypeError(
                f"Unexpected YOLO output type: "
                f"{type(model_output)}"
            )

        # Convert YOLO output into a scalar
        return model_output.reshape(-1).sum()


def generate_gradcam(
    model,
    image_path,
    output_path
):
    """
    Memory-efficient Grad-CAM visualization
    for a YOLO detection model.

    Designed for low-memory deployment environments
    such as Render Free (512 MB).

    Processing resolution:
        320 x 320

    Final image:
        Original image dimensions
    """

    print(
        f"Generating Grad-CAM for: {image_path}"
    )

    # =====================================================
    # LOAD IMAGE
    # =====================================================

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    original_height, original_width = image.shape[:2]

    print(
        f"Original image size: "
        f"{original_width}x{original_height}"
    )

    # =====================================================
    # CONVERT BGR -> RGB
    # =====================================================

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # =====================================================
    # MEMORY-SAFE CAM SIZE
    # =====================================================

    # 320 instead of 640.
    #
    # This significantly reduces:
    # - activation memory
    # - gradient memory
    # - CAM memory
    #
    CAM_SIZE = 320

    resized_rgb = cv2.resize(
        image_rgb,
        (CAM_SIZE, CAM_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # =====================================================
    # FLOAT IMAGE
    # =====================================================

    rgb_float = (
        resized_rgb.astype(
            np.float32
        ) / 255.0
    )

    # =====================================================
    # CREATE TORCH INPUT
    # =====================================================

    tensor = torch.from_numpy(
        rgb_float.transpose(2, 0, 1)
    ).unsqueeze(0)

    tensor = tensor.float()

    # =====================================================
    # FIND MODEL DEVICE
    # =====================================================

    try:

        device = next(
            model.model.parameters()
        ).device

    except StopIteration:

        device = torch.device("cpu")

    print(
        f"Grad-CAM device: {device}"
    )

    tensor = tensor.to(device)

    # =====================================================
    # SAVE ORIGINAL TRAINING STATE
    # =====================================================

    previous_training_state = (
        model.model.training
    )

    # =====================================================
    # ENABLE GRADIENTS
    # =====================================================

    model.model.train()

    # Only parameters need gradients for Grad-CAM.
    for parameter in model.model.parameters():

        parameter.requires_grad_(True)

    # =====================================================
    # FIND CONVOLUTIONAL LAYERS
    # =====================================================

    conv_layers = []

    for module in model.model.modules():

        if isinstance(
            module,
            torch.nn.Conv2d
        ):

            conv_layers.append(module)

    if not conv_layers:

        raise RuntimeError(
            "No Conv2d layer found in YOLO model."
        )

    print(
        f"Found {len(conv_layers)} Conv2d layers."
    )

    # =====================================================
    # SELECT TARGET LAYER
    # =====================================================

    target_layer = None

    # Prefer the last 3x3 convolution.
    for layer in reversed(conv_layers):

        if layer.kernel_size == (3, 3):

            target_layer = layer

            break

    # Fallback to last convolution.
    if target_layer is None:

        target_layer = conv_layers[-1]

    print(
        "Selected Grad-CAM layer:"
    )

    print(
        target_layer
    )

    # =====================================================
    # CREATE GRAD-CAM
    # =====================================================

    cam = None

    try:

        cam = GradCAM(
            model=model.model,
            target_layers=[
                target_layer
            ]
        )

        targets = [
            YOLOGradCAMTarget()
        ]

        # =================================================
        # GENERATE CAM
        # =================================================

        print(
            "Generating Grad-CAM heatmap..."
        )

        with torch.enable_grad():

            grayscale_cam = cam(
                input_tensor=tensor,
                targets=targets
            )[0]

    finally:

        # =================================================
        # RELEASE CAM RESOURCES
        # =================================================

        if cam is not None:

            try:
                cam.activations_and_grads.release()
            except Exception:
                pass

            del cam

        # =================================================
        # RESTORE MODEL STATE
        # =================================================

        if previous_training_state:

            model.model.train()

        else:

            model.model.eval()

        # =================================================
        # RELEASE INPUT TENSOR
        # =================================================

        del tensor

        # =================================================
        # CLEAR GPU CACHE IF AVAILABLE
        # =================================================

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    # =====================================================
    # CLEAN CAM
    # =====================================================

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

    # =====================================================
    # RESIZE CAM TO 320x320
    # =====================================================

    grayscale_cam = cv2.resize(
        grayscale_cam,
        (CAM_SIZE, CAM_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    # =====================================================
    # CREATE VISUALIZATION
    # =====================================================

    visualization_small = show_cam_on_image(
        rgb_float,
        grayscale_cam,
        use_rgb=True
    )

    # =====================================================
    # RESIZE TO ORIGINAL IMAGE
    # =====================================================

    if (
        original_width != CAM_SIZE
        or original_height != CAM_SIZE
    ):

        visualization = cv2.resize(
            visualization_small,
            (
                original_width,
                original_height
            ),
            interpolation=cv2.INTER_LINEAR
        )

    else:

        visualization = visualization_small

    # =====================================================
    # RGB -> BGR
    # =====================================================

    visualization = cv2.cvtColor(
        visualization,
        cv2.COLOR_RGB2BGR
    )

    # =====================================================
    # OUTPUT DIRECTORY
    # =====================================================

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # =====================================================
    # SAVE RESULT
    # =====================================================

    success = cv2.imwrite(
        str(output_path),
        visualization
    )

    if not success:

        raise RuntimeError(
            f"Could not save Grad-CAM "
            f"result to: {output_path}"
        )

    print(
        "Grad-CAM completed successfully!"
    )

    print(
        f"Output saved to: {output_path}"
    )

    return str(output_path)