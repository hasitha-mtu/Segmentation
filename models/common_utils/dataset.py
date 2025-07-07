import sklearn
import tensorflow as tf
import numpy as np
import os
import random
from glob import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from models.common_utils.config import load_config

# # --- Configuration ---
# IMG_HEIGHT = 512
# IMG_WIDTH = 512
# NUM_CHANNELS_RGB = 3
# NUM_CHANNELS_MASK = 1 # Masks are typically grayscale/single channel
# BATCH_SIZE = 4
# BUFFER_SIZE = 1000 # For shuffling
# SEED = 42 # For reproducibility

DATASET_ROOT_DIR = '../../input/updated_samples/segnet_512' # <--- IMPORTANT: Change this to your actual dataset path
IMAGE_DIR = os.path.join(DATASET_ROOT_DIR, 'images')
MASK_DIR = os.path.join(DATASET_ROOT_DIR, 'masks')

def set_seed(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    tf.config.experimental.enable_op_determinism()

# formatted_annotation_dir = "../../input/updated_samples/segnet_512/masks"
def get_image_mask_paths(config, image_dir, mask_dir):
    print(f'get_image_mask_paths|image_dir:{image_dir}')
    print(f'get_image_mask_paths|mask_dir:{mask_dir}')
    image_paths = sorted(glob(os.path.join(image_dir, '*.jpg')))  # Assuming .jpg for images
    mask_paths = sorted(glob(os.path.join(mask_dir, '*.png')))  # Assuming .png for masks

    # Ensure masks match images (by filename prefix)
    # This is a critical step for segmentation datasets
    paired_paths = []
    print(f'get_image_mask_paths|image_paths:{image_paths}')
    print(f'get_image_mask_paths|mask_paths:{mask_paths}')
    image_names = {os.path.basename(p).split('.')[0]: p for p in image_paths}
    for mask_path in mask_paths:
        print(f'get_image_mask_paths|mask_path:{mask_path}')
        mask_name_prefix = os.path.basename(mask_path).split('.')[0]
        print(f'get_image_mask_paths|mask_name_prefix:{mask_name_prefix}')
        if mask_name_prefix in image_names:
            paired_paths.append((config, image_names[mask_name_prefix], mask_path))

    if not paired_paths:
        raise ValueError(
            f"No matching image-mask pairs found. Check '{image_dir}' and '{mask_dir}' "
            f"directories and file naming conventions.")

    print(f"Found {len(paired_paths)} image-mask pairs.")
    return paired_paths

def load_image(config, image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=config['model']['input_channels']) # Use decode_png if your images are PNG
    image = tf.image.convert_image_dtype(image, tf.float32) # Normalize to [0, 1]
    image = tf.image.resize(image, [config['data']['image_size']['height'], config['data']['image_size']['width']])
    return image

def load_mask(config, mask_path):
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=config['model']['output_channels']) # Masks are typically PNG
    # Convert mask to binary (0 or 1). Water > 0.0, rest 0.0
    mask = tf.image.convert_image_dtype(mask, tf.float32) # Convert to float
    mask = tf.image.resize(mask, [config['data']['image_size']['height'], config['data']['image_size']['width']],
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.where(mask > 0.0, 1.0, 0.0) # Ensure it's strictly binary (0 or 1)
    return mask

def load_image_mask(config, image_path, mask_path):
    image = load_image(config, image_path)
    mask = load_mask(config, mask_path)
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

def create_tf_dataset(config, paired_paths, batch_size, buffer_size, augment=True, shuffle=True):
    # Create a dataset from slices of the paths
    path_ds = tf.data.Dataset.from_tensor_slices([path[1] for path in paired_paths])
    mask_path_ds = tf.data.Dataset.from_tensor_slices([path[2] for path in paired_paths])

    # Zip image and mask paths together
    dataset = tf.data.Dataset.zip((path_ds, mask_path_ds))

    # Load images and masks from paths
    # Use num_parallel_calls for faster loading
    dataset = dataset.map(load_image_mask, num_parallel_calls=tf.data.AUTOTUNE)

    # Apply augmentation conditionally
    if augment:
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle the dataset
    if shuffle:
        dataset = dataset.shuffle(buffer_size, seed=config['data']['seed'])

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

def load_datasets(config_path):
    config = load_config(config_path)
    dataset_path = config['data']['dataset_path']
    # image_dir = os.path.join(dataset_path, 'images')
    # mask_dir = os.path.join(dataset_path, 'masks')
    image_dir = f'{dataset_path}/images'
    mask_dir = f'{dataset_path}/masks'
    all_paired_paths = get_image_mask_paths(config, image_dir, mask_dir)
    train_paths, val_paths = train_test_split(all_paired_paths,
                                              test_size=config['data']['test_size'],
                                              random_state=config['data']['seed'])

    train_dataset = create_tf_dataset(config,
                                      train_paths,
                                      config['data']['batch_size'],
                                      config['data']['buffer_size'],
                                      augment=True,
                                      shuffle=True)
    validation_dataset = create_tf_dataset(config,
                                           val_paths,
                                           config['data']['batch_size'],
                                           config['data']['buffer_size'],
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


