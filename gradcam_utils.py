import cv2
import numpy as np
import torch
from pathlib import Path

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image


class YOLOGradCAMTarget:
    """
    Gradient target for Ultralytics YOLO detection output.

    The target selects the detection-related values from the
    model output and converts them into a scalar for Grad-CAM.
    """

    def __call__(self, model_output):

        # YOLO output can sometimes be wrapped in a tuple/list.
        if isinstance(model_output, (tuple, list)):
            model_output = model_output[0]

        if not torch.is_tensor(model_output):
            raise TypeError(
                f"Unexpected YOLO output type: "
                f"{type(model_output)}"
            )

        # Make sure we have a tensor connected to the graph.
        if not model_output.requires_grad:
            raise RuntimeError(
                "YOLO model output does not require gradients. "
                "Grad-CAM cannot be generated from this output."
            )

        # Flatten the output and create a scalar target.
        return model_output.reshape(-1).sum()


def _find_target_layer(model):
    """
    Find a suitable convolutional layer in the YOLO model.

    Preference:
        1. Last 3x3 Conv2d layer
        2. Otherwise last Conv2d layer
    """

    conv_layers = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.Conv2d)
    ]

    if not conv_layers:
        raise RuntimeError(
            "No Conv2d layer found in YOLO model."
        )

    print(
        f"Found {len(conv_layers)} Conv2d layers."
    )

    # Prefer the last 3x3 convolution.
    for layer in reversed(conv_layers):

        if layer.kernel_size == (3, 3):

            print(
                "Selected Grad-CAM++ layer:"
            )

            print(layer)

            return layer

    # Fallback.
    target_layer = conv_layers[-1]

    print(
        "Selected final Conv2d layer:"
    )

    print(target_layer)

    return target_layer


def generate_gradcam(
    model,
    image_path,
    output_path
):
    """
    Generate a memory-efficient Grad-CAM++ visualization
    for an Ultralytics YOLO detection model.

    Processing:
        Original image
            ↓
        Resize to 640x640
            ↓
        Grad-CAM++
            ↓
        Generate visualization
            ↓
        Resize back to original dimensions
            ↓
        Save output
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

    # Float image for visualization.
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
    # GET YOLO PYTORCH MODEL
    # =====================================================

    if hasattr(model, "model"):
        torch_model = model.model
    else:
        torch_model = model

    # =====================================================
    # DETERMINE DEVICE
    # =====================================================

    try:

        device = next(
            torch_model.parameters()
        ).device

    except StopIteration:

        device = torch.device(
            "cpu"
        )

    print(
        f"Grad-CAM device: {device}"
    )

    tensor = tensor.to(device)

    # =====================================================
    # SAVE ORIGINAL MODEL STATE
    # =====================================================

    previous_training_state = (
        torch_model.training
    )

    # =====================================================
    # ENABLE GRADIENTS
    # =====================================================

    torch_model.train()

    for parameter in torch_model.parameters():

        parameter.requires_grad_(True)

    # =====================================================
    # DISABLE INFERENCE-ONLY BEHAVIOR
    # =====================================================

    # IMPORTANT:
    # Do NOT use torch.no_grad() here.
    # Grad-CAM requires a computational graph.

    # =====================================================
    # FIND TARGET LAYER
    # =====================================================

    target_layer = _find_target_layer(
        torch_model
    )

    # =====================================================
    # INITIALIZE GRAD-CAM++
    # =====================================================

    cam = None

    try:

        cam = GradCAMPlusPlus(
            model=torch_model,
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
            "Generating Grad-CAM++ heatmap..."
        )

        with torch.enable_grad():

            grayscale_cam = cam(
                input_tensor=tensor,
                targets=targets
            )[0]

    finally:

        # =================================================
        # CLEAN UP CAM
        # =================================================

        if cam is not None:

            try:
                cam.activations_and_grads.release()
            except Exception:
                pass

        # =================================================
        # RESTORE MODEL STATE
        # =================================================

        if previous_training_state:

            torch_model.train()

        else:

            torch_model.eval()

        # Restore parameter gradient settings.
        for parameter in torch_model.parameters():

            parameter.requires_grad_(False)

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
    # RESIZE CAM TO 640x640
    # =====================================================

    grayscale_cam = cv2.resize(
        grayscale_cam,
        (CAM_SIZE, CAM_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    # =====================================================
    # GENERATE HEATMAP
    # =====================================================

    visualization_small = show_cam_on_image(
        rgb_float,
        grayscale_cam,
        use_rgb=True
    )

    # =====================================================
    # RESIZE TO ORIGINAL IMAGE SIZE
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