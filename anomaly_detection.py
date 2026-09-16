import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. DATASET
# ============================================================

class BeltImageDataset(Dataset):

    def __init__(self, image_dir, image_size=128):
        self.image_dir = image_dir
        self.image_size = image_size

        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

        self.images = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(valid_extensions)
        ]

        self.images.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_path = self.images[idx]

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(
            image,
            (self.image_size, self.image_size)
        )

        image = image.astype(np.float32) / 255.0

        # HWC → CHW
        image = np.transpose(image, (2, 0, 1))

        image = torch.tensor(image, dtype=torch.float32)

        return image


# ============================================================
# 2. CONVOLUTIONAL AUTOENCODER
# ============================================================

class ConvAutoencoder(nn.Module):

    def __init__(self):

        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                256, 128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32, 3,
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
# 3. TRAINING
# ============================================================

def train_autoencoder(
    image_dir,
    epochs=30,
    batch_size=8,
    learning_rate=0.001
):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    dataset = BeltImageDataset(image_dir)

    print("Images:", len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    model = ConvAutoencoder().to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    os.makedirs("models", exist_ok=True)

    print("\nStarting Autoencoder training...\n")

    for epoch in range(epochs):

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

        average_loss = total_loss / len(dataloader)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {average_loss:.6f}"
        )

    model_path = "models/belt_autoencoder.pt"

    torch.save(
        model.state_dict(),
        model_path
    )

    print("\nTraining complete!")
    print("Model saved to:", model_path)

    return model


# ============================================================
# 4. ANOMALY SCORE
# ============================================================

def calculate_anomaly_score(
    model,
    image_path,
    device
):

    model.eval()

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (128, 128)
    )

    image = image.astype(
        np.float32
    ) / 255.0

    image = np.transpose(
        image,
        (2, 0, 1)
    )

    tensor = torch.tensor(
        image,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        reconstructed = model(tensor)

        error = torch.mean(
            (tensor - reconstructed) ** 2
        )

    return error.item()


# ============================================================
# 5. TEST ANOMALY DETECTION
# ============================================================

if __name__ == "__main__":

    IMAGE_DIR = "training_data/images"

    model = train_autoencoder(
        IMAGE_DIR,
        epochs=30,
        batch_size=8,
        learning_rate=0.001
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nTesting anomaly detection...\n")

    test_images = [
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    scores = []

    for image_path in test_images[:20]:

        score = calculate_anomaly_score(
            model,
            image_path,
            device
        )

        scores.append(
            (os.path.basename(image_path), score)
        )

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print("Top anomalous images:\n")

    for filename, score in scores:

        print(
            f"{filename:40s} "
            f"Anomaly Score: {score:.6f}"
        )