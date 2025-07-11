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

if __name__ == "__main__":
    annotation_dir = "../../data/dataset/masks"
    formatted_annotation_dir = "../../input/updated_samples/segnet_512/test/masks"
    formatted_image_dir = "../../input/updated_samples/segnet_512/test/images"

    os.makedirs(formatted_annotation_dir, exist_ok=True)
    os.makedirs(formatted_image_dir, exist_ok=True)

    for filename in os.listdir(annotation_dir):
        mask_path = os.path.join(annotation_dir, filename)
        ann = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # shape: (H, W)
        ann = cv2.resize(ann, (512, 512))

        mask = create_confidence_mask(ann)

        # Save mask as 8-bit single-channel image
        # updated_mask_path = os.path.join(formatted_annotation_dir, f'mask_{filename}')
        updated_mask_path = os.path.join(formatted_annotation_dir, filename)
        print(f"Resized mask path: {updated_mask_path}")
        cv2.imwrite(updated_mask_path, (mask * 255).astype(np.uint8))

        image_path = mask_path.replace("masks", "images")
        image_path = image_path.replace(".png", ".jpg")

        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (512, 512))

        updated_image_path = os.path.join(formatted_image_dir, filename)
        print(f"Resized image path: {updated_image_path}")
        cv2.imwrite(updated_image_path, resized_image)



# if __name__ == "__main__":
#     image_dir = "../../input/updated_samples/samples/crookstown/images"
#     formatted_image_dir = "../../input/updated_samples/segnet_512/images"
#
#     os.makedirs(formatted_image_dir, exist_ok=True)
#
#     for filename in os.listdir(image_dir):
#         path = os.path.join(image_dir, filename)
#         print(f"Original image path: {path}")
#
#         image = cv2.imread(path)
#         resized_image = cv2.resize(image, (512, 512))
#
#         # updated_image_path = os.path.join(formatted_image_dir, f'image_{filename}')
#         updated_image_path = os.path.join(formatted_image_dir, filename)
#         cv2.imwrite(updated_image_path, resized_image)
#
# if __name__ == "__main__":
#     annotation_dir = "../../input/updated_samples/samples/crookstown/masks"
#     formatted_annotation_dir = "../../input/updated_samples/segnet_512/masks"
#
#     os.makedirs(formatted_annotation_dir, exist_ok=True)
#
#     for filename in os.listdir(annotation_dir):
#         path = os.path.join(annotation_dir, filename)
#         print(f"Original mask path: {path}")
#         ann = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # shape: (H, W)
#         ann = cv2.resize(ann, (512, 512))
#
#         mask = create_confidence_mask(ann)
#
#         # Save mask as 8-bit single-channel image
#         # updated_mask_path = os.path.join(formatted_annotation_dir, f'mask_{filename}')
#         updated_mask_path = os.path.join(formatted_annotation_dir, filename)
#         print(f"Updated mask path: {updated_mask_path}")
#         cv2.imwrite(updated_mask_path, (mask * 255).astype(np.uint8))

# if __name__ == "__main__":
#     image_dir = "../../input/samples/crookstown/images"
#     formatted_image_dir = "../../input/samples/segnet_256/images"
#
#     os.makedirs(formatted_image_dir, exist_ok=True)
#
#     for filename in os.listdir(image_dir):
#         path = os.path.join(image_dir, filename)
#         print(f"Original image path: {path}")
#
#         image = cv2.imread(path)
#         resized_image = cv2.resize(image, (256, 256))
#
#         updated_image_path = os.path.join(formatted_image_dir, filename)
#         print(f"Updated image path: {updated_image_path}")
#         cv2.imwrite(updated_image_path, resized_image)
#
# if __name__ == "__main__":
#     annotation_dir = "../../input/samples/crookstown/masks"
#     formatted_annotation_dir = "../../input/samples/segnet_256/masks"
#
#     os.makedirs(formatted_annotation_dir, exist_ok=True)
#
#     for filename in os.listdir(annotation_dir):
#         path = os.path.join(annotation_dir, filename)
#         print(f"Original mask path: {path}")
#         ann = cv2.imread(path)  # shape: (H, W)
#         resized_ann = cv2.resize(ann, (256, 256))
#
#         # Save mask as 8-bit single-channel image
#         updated_mask_path = os.path.join(formatted_annotation_dir, filename)
#         print(f"Updated mask path: {updated_mask_path}")
#         cv2.imwrite(updated_mask_path, resized_ann)

# if __name__ == "__main__":
#     path = "../../input/samples/crookstown/images"
#     image_paths = sorted(glob(path + "/*.jpg"))
#     random.Random(1337).shuffle(image_paths)
#     for i in range(10):
#         plot_details(image_paths[i])

# if __name__ == "__main__":
#     sample1_image = "../../input/samples/sample1.jpg"
#     channels = ['RED', 'GREEN', 'BLUE', 'NDWI', 'Canny', 'LBP', 'GradMag', 'Shadow Mask', 'Lightness', 'GreenRed', 'XYZ']
#     stacked = selected_channels(channels, (512, 512), sample1_image)
#     print(f"Shape stacked image : {stacked.shape}")

# if __name__ == "__main__":
#     sample1_image = "../../input/samples/sample1.JPG"
#     sample1_mask = "../../input/samples/sample1_mask.png"
#     image = cv2.imread(sample1_image)
#     image_mask = cv2.imread(sample1_mask)
#     print(f'Image shape : {image.shape}')
#     print(f'Image mask shape : {image_mask.shape}')