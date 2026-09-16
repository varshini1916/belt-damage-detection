import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# SETTINGS
# ============================================================

IMAGE_DIR = "training_data/images"
LABEL_DIR = "training_data/labels"

IMAGE_SIZE = 128
EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 0.001

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "belt_roi_autoencoder.pt"
)


# ============================================================
# ROI EXTRACTION
# ============================================================

def extract_belt_roi(image_path, label_path):
    """
    Reads a YOLO polygon annotation and extracts
    the conveyor-belt region.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    height, width = image.shape[:2]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    if not os.path.exists(label_path):
        return image

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:

        values = line.strip().split()

        if len(values) < 7:
            continue

        # First value = class
        coordinates = list(
            map(float, values[1:])
        )

        # x,y pairs
        points = []

        for i in range(
            0,
            len(coordinates) - 1,
            2
        ):

            x = int(
                coordinates[i] * width
            )

            y = int(
                coordinates[i + 1] * height
            )

            points.append([x, y])

        if len(points) >= 3:

            polygon = np.array(
                points,
                dtype=np.int32
            )

            cv2.fillPoly(
                mask,
                [polygon],
                255
            )

    # If ROI couldn't be created,
    # return original image
    if np.sum(mask) == 0:
        return image

    # Keep only belt region
    roi = cv2.bitwise_and(
        image,
        image,
        mask=mask
    )

    # Crop to bounding rectangle
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return image

    x1 = max(0, np.min(xs))
    x2 = min(width, np.max(xs) + 1)

    y1 = max(0, np.min(ys))
    y2 = min(height, np.max(ys) + 1)

    roi = roi[y1:y2, x1:x2]

    return roi


# ============================================================
# DATASET
# ============================================================

class BeltROIDataset(Dataset):

    def __init__(
        self,
        image_dir,
        label_dir,
        image_size=128
    ):

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size

        valid_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp"
        )

        self.images = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(
                valid_extensions
            )
        ]

        self.images.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        filename = self.images[index]

        image_path = os.path.join(
            self.image_dir,
            filename
        )

        label_name = os.path.splitext(
            filename
        )[0] + ".txt"

        label_path = os.path.join(
            self.label_dir,
            label_name
        )

        roi = extract_belt_roi(
            image_path,
            label_path
        )

        roi = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2RGB
        )

        roi = cv2.resize(
            roi,
            (
                self.image_size,
                self.image_size
            )
        )

        roi = roi.astype(
            np.float32
        ) / 255.0

        # HWC → CHW
        roi = np.transpose(
            roi,
            (2, 0, 1)
        )

        return torch.tensor(
            roi,
            dtype=torch.float32
        )


# ============================================================
# CONVOLUTIONAL AUTOENCODER
# ============================================================

class BeltAutoencoder(nn.Module):

    def __init__(self):

        super().__init__()

        # -------------------------------
        # Encoder
        # -------------------------------

        self.encoder = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU()
        )

        # -------------------------------
        # Decoder
        # -------------------------------

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32,
                3,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        encoded = self.encoder(x)

        reconstructed = self.decoder(encoded)

        return reconstructed


# ============================================================
# TRAINING
# ============================================================

def train():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    dataset = BeltROIDataset(
        IMAGE_DIR,
        LABEL_DIR,
        IMAGE_SIZE
    )

    print(
        "Images:",
        len(dataset)
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    model = BeltAutoencoder().to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    print("\nStarting belt ROI Autoencoder training...\n")

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for images in dataloader:

            images = images.to(device)

            optimizer.zero_grad()

            reconstructed = model(images)

            loss = criterion(
                reconstructed,
                images
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss /
            len(dataloader)
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {average_loss:.6f}"
        )

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print("\nTraining complete!")

    print(
        "Model saved to:",
        MODEL_PATH
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train()
    