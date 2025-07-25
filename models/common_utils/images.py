import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern
import tensorflow as tf
from keras.utils import load_img, img_to_array


def load_image(path: str, size=(512,512), color_mode = "rgb"):
    img = load_img(path, color_mode=color_mode)
    img_array = img_to_array(img)
    normalized_img_array = img_array/255.
    resized_img = tf.image.resize(normalized_img_array, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return resized_img

def save_image(dir_path, image, filename):
    file_name = f"{dir_path}/{filename}.png"
    os.makedirs(dir_path, exist_ok=True)
    plt.imsave(file_name, image)

def show_image(dir_path, image, index, title=None, save=False):
    if save:
        if title:
            file_name = f"{dir_path}/{title}_{index}.png"
        else:
            file_name = f"{dir_path}/predicted_mask_{index}.png"
        os.makedirs(dir_path, exist_ok=True)
        plt.imsave(file_name, image)
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')

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

def compute_lab(rgb_image):
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    return lab

def compute_xyz(rgb_image):
    xyz = cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2XYZ)
    return xyz

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

# channels = ['RED', 'GREEN', 'BLUE', 'NDWI', 'Canny', 'LBP', 'HSV Saturation', 'HSV Value', 'GradMag', 'Shadow Mask']
def selected_channels(channels, size, image_path):
    rgb_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    channel_stack = [rgb_image]
    if 'NDWI' in channels:
        ndwi = compute_ndwi(rgb_image)
        channel_stack.append(ndwi)
    if 'Canny' in channels:
        canny = compute_edges(rgb_image)
        channel_stack.append(canny)
    if 'LBP' in channels:
        lbp = compute_lbp(rgb_image)
        channel_stack.append(lbp)
    if 'HSV Saturation' in channels:
        hsv_saturation, _hsv_value = compute_hsv(rgb_image)
        channel_stack.append(hsv_saturation)
    if 'HSV Value' in channels:
        _hsv_saturation, hsv_value = compute_hsv(rgb_image)
        channel_stack.append(hsv_value)
    if 'GradMag' in channels:
        gradient_mag = compute_morphological_edge(rgb_image)
        channel_stack.append(gradient_mag)
    if 'Shadow Mask' in channels:
        shadow_mask = compute_shadow_mask(rgb_image)
        channel_stack.append(shadow_mask)
    # LAB Color Space
    if 'Lightness' in channels:
        lab = compute_lab(rgb_image)
        lightness = lab[:, :, 0]
        channel_stack.append(lightness)
    if 'GreenRed' in channels:
        lab = compute_lab(rgb_image)
        green_red = lab[:, :, 1]
        channel_stack.append(green_red)
    if 'BlueYellow' in channels:
        lab = compute_lab(rgb_image)
        blue_yellow = lab[:, :, 2]
        channel_stack.append(blue_yellow)
    # XYZ Color Space
    if 'X' in channels:
        xyz = compute_xyz(rgb_image)
        x = xyz[:, :, 0]
        channel_stack.append(x)
    if 'Y' in channels:
        xyz = compute_xyz(rgb_image)
        y = xyz[:, :, 1]
        channel_stack.append(y)
    if 'Z' in channels:
        xyz = compute_xyz(rgb_image)
        z = xyz[:, :, 2]
        channel_stack.append(z)


    stacked = np.dstack(tuple([format_image(size, channel) for channel in channel_stack]))
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
    shadow_mask = compute_shadow_mask(rgb, threshold=0.8)
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
    plt.imshow(shadow_mask)
    plt.title('Shadow Mask')
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

def create_confidence_mask(annotation, threshold=0.0):
    """
    Convert a weak annotation into a binary mask:
    1.0 → confidently labeled pixel
    0.0 → ignore in loss (e.g., occluded or unlabeled)

    Parameters:
        annotation: np.ndarray or tf.Tensor of shape (H, W) or (H, W, 1)
        threshold: pixel values > threshold are considered labeled

    Returns:
        mask: same shape as input, with values 0.0 or 1.0
    """
    # Ensure annotation is a NumPy array
    if isinstance(annotation, tf.Tensor):
        annotation = annotation.numpy()

    # Remove channel dim if exists
    if annotation.ndim == 3 and annotation.shape[-1] == 1:
        annotation = annotation[..., 0]

    # Create mask
    mask = np.where(annotation > threshold, 1.0, 0.0).astype(np.float32)

    # Add channel dim back
    return mask[..., np.newaxis]

def resize_normalize_scale(data):
    data = cv2.resize(data, (512, 512))
    normalized = (data - data.min()) / (data.max() - data.min())
    scaled = (normalized * 255).astype(np.uint8)
    return scaled

if __name__ == "__main__":
    base_dir = "../../input/updated_samples/multi_channel_segnet_512"
    mask_dir = "../../input/updated_samples/segnet_512/masks"
    image_dir = "../../input/updated_samples/segnet_512/images"

    os.makedirs(base_dir, exist_ok=True)

    for filename in os.listdir(mask_dir):
        print(f'filename : {filename}')
        image_path = os.path.join(image_dir, filename)
        print(f'image_path : {image_path}')
        rgb = cv2.imread(image_path, cv2.COLOR_BGR2RGB)  # shape: (H, W)

        new_mask_dir = f"{base_dir}/mask"
        os.makedirs(new_mask_dir, exist_ok=True)
        mask_path = os.path.join(mask_dir, filename)
        new_mask_path = os.path.join(new_mask_dir, filename)
        mask_data = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # shape: (H, W)
        mask_data = cv2.resize(mask_data, (512, 512))
        mask_data = create_confidence_mask(mask_data)
        cv2.imwrite(new_mask_path, (mask_data * 255).astype(np.uint8))

        rgb_dir = f"{base_dir}/image"
        os.makedirs(rgb_dir, exist_ok=True)
        rgb_path = os.path.join(rgb_dir, filename)
        rgb_data = cv2.resize(rgb, (512, 512))
        cv2.imwrite(rgb_path, rgb)

        # Generate NDWI dataset
        ndwi_dir = f"{base_dir}/ndwi"
        os.makedirs(ndwi_dir, exist_ok=True)
        ndwi_path = os.path.join(ndwi_dir, filename)
        ndwi_data = compute_ndwi(rgb)
        cv2.imwrite(ndwi_path, resize_normalize_scale(ndwi_data))

        # Generate Canny Edge dataset
        canny_dir = f"{base_dir}/canny"
        os.makedirs(canny_dir, exist_ok=True)
        canny_path = os.path.join(canny_dir, filename)
        canny_data = compute_edges(rgb)
        cv2.imwrite(canny_path, resize_normalize_scale(canny_data))

        # Generate Canny LBP dataset
        lbp_dir = f"{base_dir}/lbp"
        os.makedirs(lbp_dir, exist_ok=True)
        lbp_path = os.path.join(lbp_dir, filename)
        lbp_data = compute_lbp(rgb)
        cv2.imwrite(lbp_path, resize_normalize_scale(lbp_data))

        hsv_saturation_dir = f"{base_dir}/hsv_saturation"
        os.makedirs(hsv_saturation_dir, exist_ok=True)
        hsv_saturation_path = os.path.join(hsv_saturation_dir, filename)
        hsv_saturation_data = compute_edges(rgb)
        hsv_saturation_data, _hsv_value_data = compute_hsv(rgb)
        cv2.imwrite(hsv_saturation_path, resize_normalize_scale(hsv_saturation_data))

        hsv_value_dir = f"{base_dir}/hsv_value"
        os.makedirs(hsv_value_dir, exist_ok=True)
        hsv_value_path = os.path.join(hsv_value_dir, filename)
        _hsv_saturation, hsv_value_data = compute_hsv(rgb)
        cv2.imwrite(hsv_value_path, resize_normalize_scale(hsv_value_data))

        grad_mag_dir = f"{base_dir}/grad_mag"
        os.makedirs(grad_mag_dir, exist_ok=True)
        grad_mag_path = os.path.join(grad_mag_dir, filename)
        grad_mag_data = compute_morphological_edge(rgb)
        cv2.imwrite(grad_mag_path, resize_normalize_scale(grad_mag_data))

        shadow_mask_dir = f"{base_dir}/shadow_mask"
        os.makedirs(shadow_mask_dir, exist_ok=True)
        shadow_mask_path = os.path.join(shadow_mask_dir, filename)
        shadow_mask_data = compute_shadow_mask(rgb)
        cv2.imwrite(shadow_mask_path, resize_normalize_scale(shadow_mask_data))

        lightness_dir = f"{base_dir}/lightness"
        os.makedirs(lightness_dir, exist_ok=True)
        lightness_path = os.path.join(lightness_dir, filename)
        lab = compute_lab(rgb)
        lightness_data = lab[:, :, 0]
        cv2.imwrite(lightness_path, resize_normalize_scale(lightness_data))

        green_red_dir = f"{base_dir}/green_red"
        os.makedirs(green_red_dir, exist_ok=True)
        green_red_path = os.path.join(green_red_dir, filename)
        lab = compute_lab(rgb)
        green_red_data = lab[:, :, 1]
        cv2.imwrite(green_red_path, resize_normalize_scale(green_red_data))

        blue_yellow_dir = f"{base_dir}/blue_yellow"
        os.makedirs(blue_yellow_dir, exist_ok=True)
        blue_yellow_path = os.path.join(blue_yellow_dir, filename)
        lab = compute_lab(rgb)
        blue_yellow_data = lab[:, :, 2]
        cv2.imwrite(blue_yellow_path, resize_normalize_scale(blue_yellow_data))

        x_dir = f"{base_dir}/x"
        os.makedirs(x_dir, exist_ok=True)
        x_path = os.path.join(x_dir, filename)
        xyz = compute_xyz(rgb)
        x_data = xyz[:, :, 0]
        cv2.imwrite(x_path, resize_normalize_scale(x_data))

        y_dir = f"{base_dir}/y"
        os.makedirs(y_dir, exist_ok=True)
        y_path = os.path.join(y_dir, filename)
        xyz = compute_xyz(rgb)
        y_data = xyz[:, :, 1]
        cv2.imwrite(y_path, resize_normalize_scale(y_data))

        z_dir = f"{base_dir}/z"
        os.makedirs(z_dir, exist_ok=True)
        z_path = os.path.join(z_dir, filename)
        xyz = compute_xyz(rgb)
        z_data = xyz[:, :, 2]
        cv2.imwrite(z_path, resize_normalize_scale(z_data))

# if __name__ == "__main__":
#     image_dir = "../../input/updated_samples/segnet_512/test/images"
#     for filename in os.listdir(image_dir):
#         path = os.path.join(image_dir, filename)
#         plot_details(path)

# if __name__ == "__main__":
#     annotation_dir = "../../input/updated_samples/segnet_512/masks"
#     formatted_annotation_dir = "../../input/updated_samples/segnet_512/test/masks"
#     formatted_image_dir = "../../input/updated_samples/segnet_512/test/images"
#
#     os.makedirs(formatted_annotation_dir, exist_ok=True)
#     os.makedirs(formatted_image_dir, exist_ok=True)
#
#     for filename in os.listdir(annotation_dir):
#         mask_path = os.path.join(annotation_dir, filename)
#         ann = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # shape: (H, W)
#         ann = cv2.resize(ann, (512, 512))
#
#         mask = create_confidence_mask(ann)
#
#         # Save mask as 8-bit single-channel image
#         # updated_mask_path = os.path.join(formatted_annotation_dir, f'mask_{filename}')
#         updated_mask_path = os.path.join(formatted_annotation_dir, filename)
#         print(f"Resized mask path: {updated_mask_path}")
#         cv2.imwrite(updated_mask_path, (mask * 255).astype(np.uint8))
#
#         image_path = mask_path.replace("masks", "images")
#         image_path = image_path.replace(".png", ".jpg")
#
#         image = cv2.imread(image_path)
#         resized_image = cv2.resize(image, (512, 512))
#
#         updated_image_path = os.path.join(formatted_image_dir, filename)
#         print(f"Resized image path: {updated_image_path}")
#         cv2.imwrite(updated_image_path, resized_image)
