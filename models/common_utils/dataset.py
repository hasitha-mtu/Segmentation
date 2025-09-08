import tensorflow as tf
import numpy as np
import os
import random
from glob import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
from random import shuffle
from tensorflow.keras.utils import save_img

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
    image_paths = sorted(glob(os.path.join(image_dir, '*.png')))  # Assuming .jpg for images
    mask_paths = sorted(glob(os.path.join(mask_dir, '*.png')))  # Assuming .png for masks
    print(f'image_paths count: {len(image_paths)}')
    print(f'mask_paths count: {len(mask_paths)}')

    # Ensure masks match images (by filename prefix)
    # This is a critical step for segmentation datasets
    paired_paths = []
    image_names = {os.path.basename(p).split('.')[0]: p for p in image_paths}
    for mask_path in mask_paths:
        mask_name_prefix = os.path.basename(mask_path).split('.')[0]
        if mask_name_prefix in image_names:
            paired_paths.append((image_names[mask_name_prefix], mask_path))
        else:
            print(f'Missing {mask_name_prefix} in images')

    if not paired_paths:
        raise ValueError(
            f"No matching image-mask pairs found. Check '{image_dir}' and '{mask_dir}' "
            f"directories and file naming conventions.")

    print(f"Found {len(paired_paths)} image-mask pairs.")
    return paired_paths

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=ModelConfig.MODEL_INPUT_CHANNELS) # Use decode_png if your images are PNG
    image = tf.image.convert_image_dtype(image, tf.float32) # Normalize to [0, 1]
    image = tf.image.resize(image, [ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH])
    return image

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
    mask = tf.where(mask > 0.0, 1.0, 0.0)

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
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2) # Only for RGB images
    image = tf.image.random_hue(image, max_delta=0.1) # Only for RGB images

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
    print(f'load_datasets|test set split: {ModelConfig.TEST_SIZE}')
    train_paths, val_paths = train_test_split(all_paired_paths,
                                              test_size=ModelConfig.TEST_SIZE,
                                              random_state=ModelConfig.SEED)

    print(f'load_datasets|train_paths count: {len(train_paths)}')
    print(f'load_datasets|val_paths count: {len(val_paths)}')

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


def load_datasets_all(config_file, config_loaded=False):
    if not config_loaded:
        load_config(config_file)
    image_dir = f'{ModelConfig.DATASET_PATH}/images'
    mask_dir = f'{ModelConfig.DATASET_PATH}/masks'
    all_paired_paths = get_image_mask_paths(image_dir, mask_dir)
    train_paths, remaining_paths = train_test_split(all_paired_paths,
                                              test_size=ModelConfig.TEST_SIZE,
                                              random_state=ModelConfig.SEED)

    validation_paths, test_paths = train_test_split(all_paired_paths,
                                                    test_size=0.5,
                                                    random_state=ModelConfig.SEED)

    train_dataset = create_tf_dataset(train_paths,
                                      ModelConfig.BATCH_SIZE,
                                      ModelConfig.BUFFER_SIZE,
                                      augment=True,
                                      shuffle=True)

    validation_dataset = create_tf_dataset(validation_paths,
                                           ModelConfig.BATCH_SIZE,
                                           ModelConfig.BUFFER_SIZE,
                                           augment=False,
                                           shuffle=False)  # No augmentation/shuffle for validation

    test_dataset = create_tf_dataset(test_paths,
                                           ModelConfig.BATCH_SIZE,
                                           ModelConfig.BUFFER_SIZE,
                                           augment=False,
                                           shuffle=False)  # No augmentation/shuffle for testing

    return train_dataset, validation_dataset, test_dataset


def check_class_imbalance(train_ds):
    all_labels = []
    for _, labels_batch in train_ds:
        # labels_batch is now expected to be a tensor of 0s and 1s (int32)
        # It's a batch of masks, so it's likely (batch_size, height, width, channels)

        # Flatten the entire batch of masks into a 1D array of pixel labels
        all_labels.extend(labels_batch.numpy().flatten().tolist())

    class_counts = Counter(all_labels)

    print("Class Counts (after binarization):")
    # You should now only see counts for 0 and 1
    # Adjusting for potential num_classes (binary = 2)
    class_names_map = {0: 'Non-Water', 1: 'Water'}

    for class_id, count in class_counts.items():
        print(f"  {class_names_map.get(class_id, class_id)}: {count} pixels")

    total_labels_counted = sum(class_counts.values())
    print(f"\nTotal pixels counted across all batches: {total_labels_counted}")

    print("\nPixel Class Distribution (Percentages):")
    for class_id, count in class_counts.items():
        percentage = (count / total_labels_counted) * 100
        print(f"  {class_names_map.get(class_id, class_id)}: {percentage:.2f}%")

    # Visualize (only two bars now)
    class_ids = list(class_counts.keys())
    counts = list(class_counts.values())
    class_labels_for_plot = [class_names_map.get(i, str(i)) for i in class_ids]

    # Sort if desired (e.g., 0 then 1)
    class_labels_for_plot, counts = zip(*sorted(zip(class_labels_for_plot, counts)))

    plt.figure(figsize=(6, 5))
    plt.bar(class_labels_for_plot, counts, color=['grey', 'blue'])  # Specific colors for non-water/water
    plt.title('Pixel Class Distribution (Water vs. Non-Water)')
    plt.xlabel('Class')
    plt.ylabel('Number of Pixels')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_augmented_images(config_file):
    load_config(config_file)
    base_path = '../../output/augmentation'
    # files = glob("../../input/updated_samples/segnet_512/images/*.png")
    files = glob("../../input/dataset/validation/images/*.png")
    shuffle(files)
    os.makedirs(base_path, exist_ok=True)
    for i in range(16):
        output_path = f'{base_path}/{i}'
        os.makedirs(output_path, exist_ok=True)

        image = load_image(files[i])

        save_img(f'{output_path}/image.jpg', image)

        flip_left_right = tf.image.flip_left_right(image)
        save_img(f'{output_path}/flip_left_right.jpg', flip_left_right)

        flip_up_down = tf.image.flip_up_down(image)
        save_img(f'{output_path}/flip_up_down.jpg', flip_up_down)

        # Random rotation (by 90, 180, 270 degrees)
        # tf.image.rot90 is deterministic, random_uniform decides how many times
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        rot90 = tf.image.rot90(image, k=k)
        save_img(f'{output_path}/rot90.jpg', rot90)

        # Image-specific augmentations (don't apply to mask)
        random_brightness = tf.image.random_brightness(image, max_delta=0.4)  # Adjust brightness by up to 40%
        save_img(f'{output_path}/random_brightness.jpg', random_brightness)
        random_contrast = tf.image.random_contrast(image, lower=1, upper=2)  # Adjust contrast by +/- 20%
        save_img(f'{output_path}/random_contrast.jpg', random_contrast)
        random_saturation = tf.image.random_saturation(image, lower=1, upper=2)  # Only for RGB images
        save_img(f'{output_path}/random_saturation.jpg', random_saturation)
        random_hue = tf.image.random_hue(image, max_delta=0.4)  # Only for RGB images
        save_img(f'{output_path}/random_hue.jpg', random_hue)

        # Clip values to ensure they stay within [0, 1] after augmentation
        augmented = tf.clip_by_value(image, 0.0, 1.0)
        save_img(f'{output_path}/augmented.jpg', augmented)

        plt.subplot(2, 4, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(2, 4, 2)
        plt.imshow(flip_left_right)
        plt.title('Flip Left Right')
        plt.axis('off')

        plt.subplot(2, 4, 3)
        plt.imshow(flip_up_down)
        plt.title('Flip Up Down')
        plt.axis('off')

        plt.subplot(2, 4, 4)
        plt.imshow(rot90)
        plt.title('Rotation(90 degrees)')
        plt.axis('off')

        plt.subplot(2, 4, 5)
        plt.imshow(random_brightness)
        plt.title('Brightness')
        plt.axis('off')

        plt.subplot(2, 4, 6)
        plt.imshow(random_contrast)
        plt.title('Contrast')
        plt.axis('off')

        plt.subplot(2, 4, 7)
        plt.imshow(random_saturation)
        plt.title('Saturation')
        plt.axis('off')

        plt.subplot(2, 4, 8)
        plt.imshow(random_hue)
        plt.title('Hue')
        plt.axis('off')

        plt.show()

def augmented_images(config_file):
    load_config(config_file)
    output_path = '../../output/augmentation/images'
    image_path = '../../output/augmentation/image.jpg'
    os.makedirs(output_path, exist_ok=True)

    image = load_image(image_path)

    save_img(f'{output_path}/image.jpg', image)

    flip_left_right = tf.image.flip_left_right(image)
    save_img(f'{output_path}/flip_left_right.jpg', flip_left_right)

    flip_up_down = tf.image.flip_up_down(image)
    save_img(f'{output_path}/flip_up_down.jpg', flip_up_down)

    # Random rotation (by 90, 180, 270 degrees)
    # tf.image.rot90 is deterministic, random_uniform decides how many times
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    rot90 = tf.image.rot90(image, k=k)
    save_img(f'{output_path}/rot90.jpg', rot90)

    # Image-specific augmentations (don't apply to mask)
    random_brightness = tf.image.random_brightness(image, max_delta=0.4)  # Adjust brightness by up to 40%
    save_img(f'{output_path}/random_brightness.jpg', random_brightness)
    random_contrast = tf.image.random_contrast(image, lower=1, upper=2)  # Adjust contrast by +/- 20%
    save_img(f'{output_path}/random_contrast.jpg', random_contrast)
    random_saturation = tf.image.random_saturation(image, lower=1, upper=2)  # Only for RGB images
    save_img(f'{output_path}/random_saturation.jpg', random_saturation)
    random_hue = tf.image.random_hue(image, max_delta=0.4)  # Only for RGB images
    save_img(f'{output_path}/random_hue.jpg', random_hue)

    # Clip values to ensure they stay within [0, 1] after augmentation
    augmented = tf.clip_by_value(image, 0.0, 1.0)
    save_img(f'{output_path}/augmented.jpg', augmented)

# if __name__ == '__main__':
#     config_path = '../unet_wsl/config.yaml'
#
#     train_dataset, _validation_dataset = load_datasets(config_path)
#     check_class_imbalance(train_dataset)
#

if __name__ == '__main__':
    config_path = '../unet_wsl/config.yaml'
    output_path = '../../output'

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
            mask_data = mask_batch[i].numpy().squeeze()
            # print(mask_data.shape)
            # np.savetxt(f"{output_path}/matrix_{i}.txt", mask_data, fmt='%d', delimiter=' ')
            display_sample(image_batch[i].numpy(), mask_data)


# if __name__ == '__main__':
#     config_file = '../unet_wsl/config.yaml'
#     plot_augmented_images(config_file)

# if __name__ == '__main__':
#     config_file = '../unet_wsl/config.yaml'
#     augmented_images(config_file)


