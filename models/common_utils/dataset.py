import tensorflow as tf
import numpy as np
import os
import random
from glob import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
from skimage.feature import local_binary_pattern
from models.common_utils.config import load_config, ModelConfig


def set_seed(seed_value, enable_op_determinism=True):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    print(f'OP Determinism enabled : {enable_op_determinism}')
    if enable_op_determinism:
        tf.config.experimental.enable_op_determinism()

# formatted_annotation_dir = "../../input/updated_samples/segnet_512/masks"
def get_image_mask_paths(image_dir, mask_dir):
    image_paths = sorted(glob(os.path.join(image_dir, '*.jpg')))  # Assuming .jpg for images
    mask_paths = sorted(glob(os.path.join(mask_dir, '*.png')))  # Assuming .png for masks

    # Ensure masks match images (by filename prefix)
    # This is a critical step for segmentation datasets
    paired_paths = []
    image_names = {os.path.basename(p).split('.')[0]: p for p in image_paths}
    for mask_path in mask_paths:
        mask_name_prefix = os.path.basename(mask_path).split('.')[0]
        if mask_name_prefix in image_names:
            paired_paths.append((image_names[mask_name_prefix], mask_path))

    if not paired_paths:
        raise ValueError(
            f"No matching image-mask pairs found. Check '{image_dir}' and '{mask_dir}' "
            f"directories and file naming conventions.")

    print(f"Found {len(paired_paths)} image-mask pairs.")
    return paired_paths

def load_image1(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=ModelConfig.MODEL_INPUT_CHANNELS) # Use decode_png if your images are PNG
    image = tf.image.convert_image_dtype(image, tf.float32) # Normalize to [0, 1]
    image = tf.image.resize(image, [ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH])
    return image

def load_image(image_path):
    print(f"load_image|image_path:{image_path}")
    # image = cv2.imread(image_path)
    image_tensor = tf.io.read_file(image_path)
    print(f"load_image|image_tensor type:{type(image_tensor)}")
    print(f"load_image|image_tensor shape:{image_tensor.shape}")
    image = image_tensor.numpy()
    print(f"load_image|image type:{type(image)}")
    print(f"load_image|image shape:{image.shape}")

    multi_channel_image = selected_channels(ModelConfig.MODEL_INPUT_CHANNELS, image)
    print(f"load_image|multi_channel_image type:{type(multi_channel_image)}")
    print(f"load_image|multi_channel_image shape:{multi_channel_image.shape}")
    multi_channel_image = tf.image.convert_image_dtype(multi_channel_image, tf.float32)  # Normalize to [0, 1]
    print(f"load_image|multi_channel_image1 type:{type(multi_channel_image)}")
    print(f"load_image|multi_channel_image1 shape:{multi_channel_image.shape}")
    multi_channel_image = tf.image.resize(multi_channel_image, [ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH])
    print(f"load_image|multi_channel_image2 type:{type(multi_channel_image)}")
    print(f"load_image|multi_channel_image2 shape:{multi_channel_image.shape}")
    return multi_channel_image

# channels = ['RED', 'GREEN', 'BLUE', 'NDWI', 'Canny', 'LBP', 'HSV Saturation', 'HSV Value', 'GradMag', 'Shadow Mask']
def selected_channels(channels, rgb_image):
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


    stacked = np.dstack(tuple(channel_stack))
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

def load_and_preprocess_image(rgb_path, ndwi_path, canny_path, mask_path):
    # Load RGB image
    rgb_image = tf.io.read_file(rgb_path)
    rgb_image = tf.image.decode_image(rgb_image, channels=3, expand_animations=False) # Use decode_image for broader support
    rgb_image = tf.image.resize(rgb_image, [IMG_HEIGHT, IMG_WIDTH])
    rgb_image = tf.cast(rgb_image, tf.float32) / 255.0 # Normalize to [0, 1]

    # Load NDWI image (assuming grayscale, 1 channel)
    ndwi_image = tf.io.read_file(ndwi_path)
    ndwi_image = tf.image.decode_image(ndwi_image, channels=1, expand_animations=False) # Specify 1 channel
    ndwi_image = tf.image.resize(ndwi_image, [IMG_HEIGHT, IMG_WIDTH])
    ndwi_image = tf.cast(ndwi_image, tf.float32) / 255.0

    # Load Canny image (assuming grayscale, 1 channel)
    canny_image = tf.io.read_file(canny_path)
    canny_image = tf.image.decode_image(canny_image, channels=1, expand_animations=False) # Specify 1 channel
    canny_image = tf.image.resize(canny_image, [IMG_HEIGHT, IMG_WIDTH])
    canny_image = tf.cast(canny_image, tf.float32) / 255.0

    # Load Mask image (assuming grayscale, 1 channel for segmentation masks)
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
    mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.cast(mask, tf.int32) # Masks are typically integers (class IDs)

    # Combine all input channels
    # The order here defines the channel order in your model input
    combined_input = tf.concat([rgb_image, ndwi_image, canny_image], axis=-1)
    # The shape will be (IMG_HEIGHT, IMG_WIDTH, 3 + 1 + 1 = 5)

    return combined_input, mask

def load_mask(mask_path):
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=ModelConfig.MODEL_OUTPUT_CHANNELS) # Masks are typically PNG
    # Convert mask to binary (0 or 1). Water > 0.0, rest 0.0
    mask = tf.image.convert_image_dtype(mask, tf.float32) # Convert to float
    mask = tf.image.resize(mask, [ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH],
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.where(mask > 0.0, 1.0, 0.0) # Ensure it's strictly binary (0 or 1)
    return mask

def load_image_mask(image_path, mask_path):
    image = load_image(image_path)
    mask = load_mask(mask_path)
    return image, mask

def augment_data(image, mask):
    # Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    # Random vertical flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    # Random rotation (by 90, 180, 270 degrees)
    # tf.image.rot90 is deterministic, random_uniform decides how many times
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)
    mask = tf.image.rot90(mask, k=k)

    # Image-specific augmentations (don't apply to mask)
    image = tf.image.random_brightness(image, max_delta=0.2) # Adjust brightness by up to 20%
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2) # Adjust contrast by +/- 20%
    # image = tf.image.random_saturation(image, lower=0.8, upper=1.2) # Only for RGB images
    # image = tf.image.random_hue(image, max_delta=0.1) # Only for RGB images

    # Clip values to ensure they stay within [0, 1] after augmentation
    image = tf.clip_by_value(image, 0.0, 1.0)
    mask = tf.clip_by_value(mask, 0.0, 1.0) # Mask should still be 0 or 1

    return image, mask

def create_tf_dataset(paired_paths, batch_size, buffer_size, augment=True, shuffle=True):
    # Create a dataset from slices of the paths
    path_ds = tf.data.Dataset.from_tensor_slices([path[0] for path in paired_paths])
    mask_path_ds = tf.data.Dataset.from_tensor_slices([path[1] for path in paired_paths])

    # Zip image and mask paths together
    dataset  = tf.data.Dataset.zip((path_ds, mask_path_ds))

    # Load images and masks from paths
    # Use num_parallel_calls for faster loading
    dataset = dataset.map(load_image_mask, num_parallel_calls=tf.data.AUTOTUNE)

    # Apply augmentation conditionally
    if augment:
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle the dataset
    if shuffle:
        dataset = dataset.shuffle(buffer_size, seed=ModelConfig.SEED)

    # Batch the dataset
    dataset = dataset.batch(batch_size)

    # Prefetch data for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def display_sample(image, mask):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("RGB Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    # For grayscale masks, use cmap='gray'.
    # Mask values are 0 or 1, so imshow will display black/white
    plt.imshow(tf.squeeze(mask), cmap='gray')
    plt.title("Water Mask")
    plt.axis('off')
    plt.show()

def load_datasets(config_file, config_loaded=False):
    if not config_loaded:
        load_config(config_file)
    image_dir = f'{ModelConfig.DATASET_PATH}/images'
    mask_dir = f'{ModelConfig.DATASET_PATH}/masks'
    all_paired_paths = get_image_mask_paths(image_dir, mask_dir)
    train_paths, val_paths = train_test_split(all_paired_paths,
                                              test_size=ModelConfig.TEST_SIZE,
                                              random_state=ModelConfig.SEED)
    train_dataset = create_tf_dataset(train_paths,
                                      ModelConfig.BATCH_SIZE,
                                      ModelConfig.BUFFER_SIZE,
                                      augment=True,
                                      shuffle=True)
    validation_dataset = create_tf_dataset(val_paths,
                                           ModelConfig.BATCH_SIZE,
                                           ModelConfig.BUFFER_SIZE,
                                           augment=False,
                                           shuffle=False)  # No augmentation/shuffle for validation
    return train_dataset, validation_dataset

if __name__ == '__main__':
    config_path = '../unet_wsl/config.yaml'

    train_dataset, validation_dataset = load_datasets(config_path)

    # Take a batch from the training dataset and display
    print("\nDisplaying a sample batch from the training dataset (with augmentation):")
    for image_batch, mask_batch in train_dataset.take(1):
        for i in range(min(3, 4)):  # Display first 3 samples from the batch
            display_sample(image_batch[i].numpy(), mask_batch[i].numpy())

    # Take a batch from the validation dataset and display
    print("\nDisplaying a sample batch from the validation dataset (without augmentation):")
    for image_batch, mask_batch in validation_dataset.take(1):
        for i in range(min(3, 4)):  # Display first 3 samples from the batch
            display_sample(image_batch[i].numpy(), mask_batch[i].numpy())


