import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from belt_roi_autoencoder import (
    BeltAutoencoder,
    extract_belt_roi
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_DIR = "training_data/images"
LABEL_DIR = "training_data/labels"

MODEL_PATH = "models/belt_roi_autoencoder.pt"

IMAGE_SIZE = 128


# ============================================================
# PREPROCESS ROI
# ============================================================

def preprocess_roi(image_path):

    filename = os.path.basename(image_path)

    label_name = os.path.splitext(filename)[0] + ".txt"

    label_path = os.path.join(
        LABEL_DIR,
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
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    roi = roi.astype(
        np.float32
    ) / 255.0

    roi = np.transpose(
        roi,
        (2, 0, 1)
    )

    tensor = torch.tensor(
        roi,
        dtype=torch.float32
    ).unsqueeze(0)

    return tensor


# ============================================================
# ANOMALY SCORE
# ============================================================

def calculate_anomaly_score(
    model,
    image_path,
    device
):

    tensor = preprocess_roi(
        image_path
    )

    tensor = tensor.to(device)

    with torch.no_grad():

        reconstructed = model(
            tensor
        )

        error = torch.mean(
            (tensor - reconstructed) ** 2
        )

    return error.item()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = BeltAutoencoder().to(
        device
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.eval()

    print(
        "ROI Autoencoder loaded successfully."
    )

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    image_files = [
        os.path.join(
            IMAGE_DIR,
            f
        )
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    image_files.sort()

    print(
        "Images found:",
        len(image_files)
    )

    # --------------------------------------------------------
    # Calculate scores
    # --------------------------------------------------------

    results = []

    print(
        "\nCalculating ROI anomaly scores...\n"
    )

    for i, image_path in enumerate(
        image_files
    ):

        try:

            score = calculate_anomaly_score(
                model,
                image_path,
                device
            )

            results.append(
                (
                    os.path.basename(
                        image_path
                    ),
                    score
                )
            )

        except Exception as e:

            print(
                "Error:",
                os.path.basename(
                    image_path
                ),
                e
            )

        if (i + 1) % 50 == 0:

            print(
                f"Processed "
                f"{i + 1}/{len(image_files)}"
            )

    # --------------------------------------------------------
    # Convert scores
    # --------------------------------------------------------

    scores = np.array(
        [score for _, score in results]
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    minimum = np.min(scores)
    maximum = np.max(scores)

    mean = np.mean(scores)
    median = np.median(scores)

    std = np.std(scores)

    p90 = np.percentile(
        scores,
        90
    )

    p95 = np.percentile(
        scores,
        95
    )

    p99 = np.percentile(
        scores,
        99
    )

    print("\n" + "=" * 60)
    print(
        "ROI ANOMALY SCORE STATISTICS"
    )
    print("=" * 60)

    print(
        f"Minimum        : {minimum:.6f}"
    )

    print(
        f"Maximum        : {maximum:.6f}"
    )

    print(
        f"Mean           : {mean:.6f}"
    )

    print(
        f"Median         : {median:.6f}"
    )

    print(
        f"Std deviation  : {std:.6f}"
    )

    print(
        f"90th percentile: {p90:.6f}"
    )

    print(
        f"95th percentile: {p95:.6f}"
    )

    print(
        f"99th percentile: {p99:.6f}"
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    threshold = p95

    print("\n" + "=" * 60)
    print(
        "ROI ANOMALY THRESHOLD"
    )
    print("=" * 60)

    print(
        f"Threshold: {threshold:.6f}"
    )

    print(
        "\n95th percentile is being used "
        "as the initial unsupervised threshold."
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print("\n" + "=" * 60)
    print(
        "TOP 20 ROI ANOMALOUS IMAGES"
    )
    print("=" * 60)

    for filename, score in results[:20]:

        status = (
            "ANOMALY"
            if score >= threshold
            else "NORMAL"
        )

        print(
            f"{filename} | "
            f"{score:.6f} | "
            f"{status}"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_dir = "outputs/roi_anomaly"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    csv_path = os.path.join(
        output_dir,
        "roi_anomaly_scores.csv"
    )

    with open(
        csv_path,
        "w"
    ) as f:

        f.write(
            "image,anomaly_score,status\n"
        )

        for filename, score in results:

            status = (
                "anomaly"
                if score >= threshold
                else "normal"
            )

            f.write(
                f"{filename},"
                f"{score:.8f},"
                f"{status}\n"
            )

    threshold_path = os.path.join(
        output_dir,
        "threshold.txt"
    )

    with open(
        threshold_path,
        "w"
    ) as f:

        f.write(
            str(threshold)
        )

    # --------------------------------------------------------
    # Histogram
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        scores,
        bins=30
    )

    plt.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold:.6f}"
    )

    plt.xlabel(
        "ROI Reconstruction Error"
    )

    plt.ylabel(
        "Number of Images"
    )

    plt.title(
        "Conveyor Belt ROI Anomaly Distribution"
    )

    plt.legend()

    plt.tight_layout()

    histogram_path = os.path.join(
        output_dir,
        "roi_anomaly_distribution.png"
    )

    plt.savefig(
        histogram_path,
        dpi=150
    )

    plt.close()

    print("\nResults saved:")
    print(
        f"  {csv_path}"
    )

    print(
        f"  {threshold_path}"
    )

    print(
        f"  {histogram_path}"
    )