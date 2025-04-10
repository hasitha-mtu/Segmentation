import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
from glob import glob
import random
from skimage.feature import local_binary_pattern
from keras.utils import img_to_array
import tensorflow as tf

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

def calculate_ndwi_with_edge_detection(image_path):
    rgb_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    # Extract Green and Blue channels
    G = rgb_image[:, :, 1].astype(np.float32)  # Green channel
    B = rgb_image[:, :, 2].astype(np.float32)  # Blue channel

    # Compute NDWI
    ndwi = (G - B) / (G + B + 1e-5)

    # Normalize NDWI
    ndwi_normalized = (ndwi - np.min(ndwi)) / (np.max(ndwi) - np.min(ndwi))

    # Apply thresholding to get a water mask
    threshold = 0.05  # Adjust based on your image
    water_mask = (ndwi > threshold).astype(np.uint8) * 255

    # Apply Edge Detection (Sobel)
    sobelx = cv2.Sobel(water_mask, cv2.CV_64F, 1, 0, ksize=5)  # Horizontal edges
    sobely = cv2.Sobel(water_mask, cv2.CV_64F, 0, 1, ksize=5)  # Vertical edges
    sobel_edges = cv2.magnitude(sobelx, sobely)  # Compute gradient magnitude
    sobel_edges = (sobel_edges / np.max(sobel_edges) * 255).astype(np.uint8)  # Normalize

    # Apply Edge Detection (Canny)
    canny_edges = cv2.Canny(water_mask, 50, 150)  # Adjust thresholds as needed

    # Display Results
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(ndwi_normalized, cmap="jet")
    axes[1].set_title("NDWI Map")
    axes[1].axis("off")

    axes[2].imshow(sobel_edges, cmap="gray")
    axes[2].set_title("Sobel Edge Detection")
    axes[2].axis("off")

    axes[3].imshow(canny_edges, cmap="gray")
    axes[3].set_title("Canny Edge Detection")
    axes[3].axis("off")

    plt.show()

def load_ndwi_edge_map(image_path, edge_map_type = 'canny'):
    rgb_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    # Extract Green and Blue channels
    G = rgb_image[:, :, 1].astype(np.float32)  # Green channel
    B = rgb_image[:, :, 2].astype(np.float32)  # Blue channel

    # Compute NDWI
    ndwi = (G - B) / (G + B + 1e-5)

    # Normalize NDWI
    ndwi_normalized = (ndwi - np.min(ndwi)) / (np.max(ndwi) - np.min(ndwi))

    # Apply thresholding to get a water mask
    threshold = 0.05  # Adjust based on your image
    water_mask = (ndwi > threshold).astype(np.uint8) * 255

    if edge_map_type == 'canny':
        # Apply Edge Detection (Canny)
        edge_map = cv2.Canny(water_mask, 50, 150)  # Adjust thresholds as needed
        return {'rgb':rgb_image, 'ndwi': ndwi, 'edge_map': edge_map}
    else:
        # Apply Edge Detection (Sobel)
        sobelx = cv2.Sobel(water_mask, cv2.CV_64F, 1, 0, ksize=5)  # Horizontal edges
        sobely = cv2.Sobel(water_mask, cv2.CV_64F, 0, 1, ksize=5)  # Vertical edges
        sobel_edges = cv2.magnitude(sobelx, sobely)  # Compute gradient magnitude
        edge_map = (sobel_edges / np.max(sobel_edges) * 255).astype(np.uint8)  # Normalize
        return {'rgb':rgb_image, 'ndwi': ndwi, 'edge_map': edge_map}


def compute_ndwi(rgb_image):
    green = rgb_image[:, :, 1].astype(float)
    nir_fake = rgb_image[:, :, 0].astype(float)  # Substitute (since no NIR)
    ndwi = (green - nir_fake) / (green + nir_fake + 1e-6)
    ndwi = np.clip(ndwi, -1, 1)
    return ndwi

def compute_edges(rgb_image, edge_type='canny'):
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    if edge_type == 'canny':
        edges = cv2.Canny(gray, 100, 200)
    else:
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)  # Horizontal edges
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)  # Vertical edges
        edges = cv2.magnitude(sobelx, sobely)  # Compute gradient magnitude
    return edges

def stack_rgb_ndwi_edges(image_path):
    rgb_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    ndwi = compute_ndwi(rgb_image)
    edges = compute_edges(rgb_image)
    stacked = np.dstack((rgb_image, ndwi, edges))
    return stacked.astype(np.float32)

def compute_lbp(rgb_image):
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    return lbp

def compute_hsv(rgb_image):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    return saturation, value

def compute_shadow_mask(rgb_image, threshold = 0.05):
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    shadow_mask = (gray < threshold).astype(np.uint8)
    return shadow_mask

def compute_morphological_edge(rgb_image):
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    return gradient_mag

# Channel	Description
# 1-3	RGB
# 4	NDWI (from RGB)
# 5	Canny edges
# 6	LBP (texture)
# 7	HSV Saturation
# 8	Gradient magnitude (Sobel)

def stack_input_channels(size, image_path):
    rgb_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    ndwi = compute_ndwi(rgb_image)
    canny = compute_edges(rgb_image)
    lbp = compute_lbp(rgb_image)
    hsv_saturation, hsv_value = compute_hsv(rgb_image)
    gradient_mag = compute_morphological_edge(rgb_image)
    shadow_mask = compute_shadow_mask(rgb_image)
    stacked = np.dstack((format_image(size, rgb_image),
                         format_image(size, ndwi),
                         format_image(size, canny),
                         format_image(size, lbp),
                         format_image(size, hsv_saturation),
                         format_image(size, hsv_value),
                         format_image(size, gradient_mag),
                         format_image(size, shadow_mask)
                         ))
    return stacked.astype(np.float32)

def format_image(size, img):
    img_array = img_to_array(img)
    normalized_img_array = img_array/255.
    resized_img = tf.image.resize(normalized_img_array, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return resized_img

def plot_colors(image_path):
    # Load image in RGB format
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Split into R, G, B channels
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Plot
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(R, cmap='Reds')
    plt.title('Red Channel')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(G, cmap='Greens')
    plt.title('Green Channel')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(B, cmap='Blues')
    plt.title('Blue Channel')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_details(image_path):
    rgb = cv2.imread(image_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    ndwi = compute_ndwi(rgb)
    canny_edge = compute_edges(rgb)
    sobel_edge = compute_edges(rgb, edge_type='sobel')

    red = rgb[:, :, 0].astype(float)
    green = rgb[:, :, 1].astype(float)
    blue = rgb[:, :, 2].astype(float)

    plt.figure(figsize=(20, 18))

    plt.subplot(1, 4, 1)
    plt.imshow(rgb)
    plt.title('RGB')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(red, cmap='Reds')
    plt.title('Red')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(green, cmap='Greens')
    plt.title('Green')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(blue, cmap='Blues')
    plt.title('Blue')
    plt.axis('off')

    plt.subplot(2, 4, 1)
    plt.imshow(rgb)
    plt.title('RGB')
    plt.axis('off')

    plt.subplot(2, 4, 2)
    plt.imshow(ndwi)
    plt.title('NDWI')
    plt.axis('off')

    plt.subplot(2, 4, 3)
    plt.imshow(canny_edge)
    plt.title('Canny')
    plt.axis('off')

    plt.subplot(2, 4, 4)
    plt.imshow(sobel_edge)
    plt.title('Sobel')
    plt.axis('off')

    plt.show()

# if __name__ == "__main__":
#     path = "../../input/samples/crookstown/images"
#     image_paths = sorted(glob(path + "/*.jpg"))
#     random.Random(1337).shuffle(image_paths)
#     for i in range(10):
#         plot_details(image_paths[i])

if __name__ == "__main__":
    sample1_image = "../../input/samples/sample1.jpg"
    stacked = stack_input_channels((512, 512), sample1_image)
    print(f"Shape stacked image : {stacked.shape}")

# if __name__ == "__main__":
#     sample1_image = "../../input/samples/sample1.JPG"
#     sample1_mask = "../../input/samples/sample1_mask.png"
#     image = cv2.imread(sample1_image)
#     image_mask = cv2.imread(sample1_mask)
#     print(f'Image shape : {image.shape}')
#     print(f'Image mask shape : {image_mask.shape}')