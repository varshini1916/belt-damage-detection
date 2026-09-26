import cv2
import numpy as np
import torch
from pathlib import Path

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image


class YOLOGradCAMTarget:
    """
    Converts YOLO model output into a scalar target
    that Grad-CAM++ can use for gradient computation.
    """

    def __call__(self, model_output):

        if isinstance(model_output, (tuple, list)):
            model_output = model_output[0]

        if not torch.is_tensor(model_output):
            raise TypeError(
                f"Unexpected YOLO output type: "
                f"{type(model_output)}"
            )

        return model_output.reshape(-1).sum()


def generate_gradcam(
    model,
    image_path,
    output_path
):
    """
    Generate a memory-efficient Grad-CAM++ visualization
    for a YOLO detection model.

    Grad-CAM++ is calculated at 640x640.
    The final visualization is then resized to the
    original image dimensions.
    """

    print(
        f"Generating Grad-CAM++ for: {image_path}"
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

    CAM_SIZE = 640

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

    # =====================================================
    # CREATE TORCH INPUT
    # =====================================================

    tensor = torch.from_numpy(
        rgb_float.transpose(2, 0, 1)
    ).unsqueeze(0).float()

    # =====================================================
    # MOVE TO MODEL DEVICE
    # =====================================================

    device = next(
        model.model.parameters()
    ).device

    tensor = tensor.to(device)

    # =====================================================
    # ENABLE GRADIENTS
    # =====================================================

    previous_training_state = (
        model.model.training
    )

    model.model.train()

    for parameter in model.model.parameters():
        parameter.requires_grad_(True)

    # =====================================================
    # FIND CONVOLUTIONAL LAYERS
    # =====================================================

    conv_layers = [

        module

        for module in model.model.modules()

        if isinstance(
            module,
            torch.nn.Conv2d
        )
    ]

    if not conv_layers:
        raise RuntimeError(
            "No Conv2d layer found in YOLO model."
        )

    print(
        f"Found {len(conv_layers)} Conv2d layers."
    )

    # =====================================================
    # SELECT LAST 3x3 CONVOLUTION
    # =====================================================

    target_layer = None

    for layer in reversed(conv_layers):

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

    # =====================================================
    # INITIALIZE GRAD-CAM++
    # =====================================================

    cam = GradCAMPlusPlus(
        model=model.model,
        target_layers=[
            target_layer
        ]
    )

    targets = [
        YOLOGradCAMTarget()
    ]

    # =====================================================
    # GENERATE CAM
    # =====================================================

    try:

        print(
            "Generating Grad-CAM++ heatmap..."
        )

        with torch.enable_grad():

            grayscale_cam = cam(
                input_tensor=tensor,
                targets=targets
            )[0]

    finally:

        # Restore original model state
        if previous_training_state:
            model.model.train()
        else:
            model.model.eval()

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
    # ENSURE CAM IS 640x640
    # =====================================================

    grayscale_cam = cv2.resize(
        grayscale_cam,
        (CAM_SIZE, CAM_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    # =====================================================
    # GENERATE HEATMAP AT 640x640
    #
    # IMPORTANT:
    # Never pass the original 3840x2160 image here.
    # =====================================================

    visualization_small = show_cam_on_image(
        rgb_float,
        grayscale_cam,
        use_rgb=True
    )

    # =====================================================
    # RESIZE FINAL VISUALIZATION
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
    # SAVE
    # =====================================================

    success = cv2.imwrite(
        str(output_path),
        visualization
    )

    if not success:

        raise RuntimeError(
            f"Could not save Grad-CAM++ "
            f"result to: {output_path}"
        )

    print(
        "Grad-CAM++ completed successfully!"
    )

    print(
        f"Output saved to: {output_path}"
    )

    return str(output_path)