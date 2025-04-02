import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np


def read_image(path):
    image = mpimg.imread(path)
    print(type(image))
    print(image.shape)
    image = image.reshape((3, 3956, 5280))
    print(image.shape)
    print(len(image))
    channel1 = image[0]
    print(channel1.shape)

def calculate_ndwi(image_path):
    rgb_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Extract Green and Blue channels
    G = rgb_image[:, :, 1].astype(np.float32)  # Green channel
    B = rgb_image[:, :, 2].astype(np.float32)  # Blue channel

    # Compute NDWI using (G - B) / (G + B)
    ndwi = (G - B) / (G + B + 1e-5)  # Adding small value to avoid division by zero

    # Normalize NDWI to range [0, 1] for visualization
    ndwi_normalized = (ndwi - np.min(ndwi)) / (np.max(ndwi) - np.min(ndwi))

    # Apply a threshold to highlight water regions
    threshold = 0.05  # Adjust this based on your images
    water_mask = (ndwi > threshold).astype(np.uint8) * 255  # Convert to binary mask

    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(ndwi_normalized, cmap="jet")
    axes[1].set_title("NDWI Map")
    axes[1].axis("off")

    axes[2].imshow(water_mask, cmap="gray")
    axes[2].set_title("Water Mask (Thresholded)")
    axes[2].axis("off")

    plt.show()


if __name__ == "__main__":
    sample1_image = "../../data/samples/sample1.JPG"
    sample2_image = "../../data/samples/sample2.JPG"
    sample3_image = "../../data/samples/sample3.JPG"
    calculate_ndwi(sample3_image)

