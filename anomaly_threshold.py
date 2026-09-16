import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from anomaly_detection import ConvAutoencoder


# ============================================================
# SETTINGS
# ============================================================

IMAGE_DIR = "training_data/images"
MODEL_PATH = "models/belt_autoencoder.pt"

IMAGE_SIZE = 128


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return None

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
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
    ).unsqueeze(0)

    return tensor


# ============================================================
# CALCULATE ANOMALY SCORE
# ============================================================

def calculate_score(model, image_path, device):

    tensor = preprocess_image(image_path)

    if tensor is None:
        return None

    tensor = tensor.to(device)

    with torch.no_grad():

        reconstructed = model(tensor)

        error = torch.mean(
            (tensor - reconstructed) ** 2
        )

    return error.item()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # Load model
    model = ConvAutoencoder().to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.eval()

    print("Autoencoder loaded successfully.")

    # Get images
    image_files = [
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    image_files.sort()

    print("Images found:", len(image_files))

    # Calculate scores
    results = []

    print("\nCalculating anomaly scores...\n")

    for i, image_path in enumerate(image_files):

        score = calculate_score(
            model,
            image_path,
            device
        )

        if score is not None:

            results.append(
                (
                    os.path.basename(image_path),
                    score
                )
            )

        if (i + 1) % 50 == 0:

            print(
                f"Processed {i + 1}/{len(image_files)}"
            )

    # Extract scores
    scores = np.array(
        [score for _, score in results]
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    mean_score = np.mean(scores)
    median_score = np.median(scores)

    std_score = np.std(scores)

    percentile_90 = np.percentile(
        scores,
        90
    )

    percentile_95 = np.percentile(
        scores,
        95
    )

    percentile_99 = np.percentile(
        scores,
        99
    )

    print("\n" + "=" * 60)
    print("ANOMALY SCORE STATISTICS")
    print("=" * 60)

    print(f"Minimum       : {np.min(scores):.6f}")
    print(f"Maximum       : {np.max(scores):.6f}")
    print(f"Mean          : {mean_score:.6f}")
    print(f"Median        : {median_score:.6f}")
    print(f"Std deviation : {std_score:.6f}")
    print(f"90th percentile: {percentile_90:.6f}")
    print(f"95th percentile: {percentile_95:.6f}")
    print(f"99th percentile: {percentile_99:.6f}")

    # ========================================================
    # THRESHOLD
    # ========================================================

    # We use the 95th percentile as an initial
    # unsupervised anomaly threshold.

    threshold = percentile_95

    print("\n" + "=" * 60)
    print("INITIAL ANOMALY THRESHOLD")
    print("=" * 60)

    print(
        f"Threshold: {threshold:.6f}"
    )

    print(
        "\nImages above this threshold "
        "will initially be classified as anomalous."
    )

    # ========================================================
    # SORT RESULTS
    # ========================================================

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print("\n" + "=" * 60)
    print("TOP 20 ANOMALOUS IMAGES")
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

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    os.makedirs(
        "outputs/anomaly",
        exist_ok=True
    )

    results_file = (
        "outputs/anomaly/anomaly_scores.csv"
    )

    with open(
        results_file,
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
                f"{filename},{score:.8f},{status}\n"
            )

    # Save threshold
    with open(
        "outputs/anomaly/threshold.txt",
        "w"
    ) as f:

        f.write(
            str(threshold)
        )

    # ========================================================
    # HISTOGRAM
    # ========================================================

    plt.figure(figsize=(10, 6))

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
        "Reconstruction Error"
    )

    plt.ylabel(
        "Number of Images"
    )

    plt.title(
        "Autoencoder Anomaly Score Distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "outputs/anomaly/anomaly_distribution.png",
        dpi=150
    )

    plt.close()

    print("\nResults saved:")
    print(
        f"  {results_file}"
    )
    print(
        "  outputs/anomaly/threshold.txt"
    )
    print(
        "  outputs/anomaly/anomaly_distribution.png"
    )