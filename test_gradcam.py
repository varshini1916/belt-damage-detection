from ultralytics import YOLO
import torch

from pytorch_grad_cam import GradCAMPlusPlus


MODEL_PATH = "runs/train/belt_damage_improved/weights/best.pt"


model = YOLO(MODEL_PATH)

# Get Conv2d layers
conv_layers = [
    layer
    for layer in model.model.modules()
    if isinstance(layer, torch.nn.Conv2d)
]

print("Total Conv2d layers:", len(conv_layers))

# Layer 221 from the complete module list
target_layer = list(model.model.modules())[221]

print("Target layer:", target_layer)

cam = GradCAMPlusPlus(
    model=model.model,
    target_layers=[target_layer]
)

print("Grad-CAM++ initialized successfully!")