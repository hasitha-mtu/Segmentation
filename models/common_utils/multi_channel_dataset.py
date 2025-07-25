import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models.common_utils.config import load_config, ModelConfig

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

#  ["RED", "GREEN", "BLUE", "NDWI", "Canny", "LBP", "HSV Saturation", "HSV Value", "GradMag", "Shadow Mask"]
def load_datasets(config_file, config_loaded=False):
    if not config_loaded:
        load_config(config_file)
    rgb_dir = f'{ModelConfig.DATASET_PATH}/image'
    ndwi_dir = f'{ModelConfig.DATASET_PATH}/ndwi'
    canny_dir = f'{ModelConfig.DATASET_PATH}/canny'
    lbp_dir = f'{ModelConfig.DATASET_PATH}/lbp'
    hsv_saturation_dir = f'{ModelConfig.DATASET_PATH}/hsv_saturation'
    hsv_value_dir = f'{ModelConfig.DATASET_PATH}/hsv_value'
    grad_mag_dir = f'{ModelConfig.DATASET_PATH}/grad_mag'
    shadow_mask_dir = f'{ModelConfig.DATASET_PATH}/shadow_mask'
    lightness_dir = f'{ModelConfig.DATASET_PATH}/lightness'
    blue_yellow_dir = f'{ModelConfig.DATASET_PATH}/blue_yellow'
    green_red_dir = f'{ModelConfig.DATASET_PATH}/green_red'
    x_dir = f'{ModelConfig.DATASET_PATH}/x'
    y_dir = f'{ModelConfig.DATASET_PATH}/y'
    z_dir = f'{ModelConfig.DATASET_PATH}/z'

    masks_dir = f'{ModelConfig.DATASET_PATH}/mask'

    rgb_image_paths = sorted([os.path.join(rgb_dir, fname) for fname in os.listdir(rgb_dir)])
    ndwi_image_paths = sorted([os.path.join(ndwi_dir, fname) for fname in os.listdir(ndwi_dir)])
    canny_image_paths = sorted([os.path.join(canny_dir, fname) for fname in os.listdir(canny_dir)])

    lbp_image_paths = sorted([os.path.join(lbp_dir, fname) for fname in os.listdir(lbp_dir)])
    hsv_saturation_image_paths = sorted([os.path.join(hsv_saturation_dir, fname) for fname in os.listdir(hsv_saturation_dir)])
    hsv_value_image_paths = sorted([os.path.join(hsv_value_dir, fname) for fname in os.listdir(hsv_value_dir)])

    grad_mag_image_paths = sorted([os.path.join(grad_mag_dir, fname) for fname in os.listdir(grad_mag_dir)])
    shadow_mask_image_paths = sorted([os.path.join(shadow_mask_dir, fname) for fname in os.listdir(shadow_mask_dir)])
    lightness_image_paths = sorted([os.path.join(lightness_dir, fname) for fname in os.listdir(lightness_dir)])

    blue_yellow_image_paths = sorted([os.path.join(blue_yellow_dir, fname) for fname in os.listdir(blue_yellow_dir)])
    green_red_image_paths = sorted([os.path.join(green_red_dir, fname) for fname in os.listdir(green_red_dir)])

    x_image_paths = sorted([os.path.join(x_dir, fname) for fname in os.listdir(x_dir)])
    y_image_paths = sorted([os.path.join(y_dir, fname) for fname in os.listdir(y_dir)])
    z_image_paths = sorted([os.path.join(z_dir, fname) for fname in os.listdir(z_dir)])

    mask_paths = sorted([os.path.join(masks_dir, fname) for fname in os.listdir(masks_dir)])

    print(f"load_datasets|rgb_image_paths count:{len(rgb_image_paths)}")
    print(f"load_datasets|lbp_image_paths count:{len(lbp_image_paths)}")
    print(f"load_datasets|lightness_image_paths count:{len(lightness_image_paths)}")
    print(f"load_datasets|y_image_paths count:{len(y_image_paths)}")
    print(f"load_datasets|mask_paths count:{len(mask_paths)}")

    assert (len(rgb_image_paths) == len(ndwi_image_paths) == len(canny_image_paths) == len(lbp_image_paths)
            == len(hsv_saturation_image_paths) == len(hsv_value_image_paths) == len(grad_mag_image_paths)
            == len(shadow_mask_image_paths) == len(lightness_image_paths) == len(blue_yellow_image_paths)
            == len(green_red_image_paths) == len(x_image_paths) == len(y_image_paths) == len(z_image_paths)
            == len(mask_paths))

    all_data_paths = list(zip(rgb_image_paths, ndwi_image_paths, canny_image_paths, lbp_image_paths,
                              hsv_saturation_image_paths, hsv_value_image_paths, grad_mag_image_paths,
                              shadow_mask_image_paths, lightness_image_paths, blue_yellow_image_paths,
                              green_red_image_paths, x_image_paths, y_image_paths, z_image_paths,
                              mask_paths))

    train_paths, val_paths = train_test_split(all_data_paths,
                                              test_size=ModelConfig.TEST_SIZE,
                                              random_state=ModelConfig.SEED)

    (train_rgb, train_ndwi, train_canny, train_lbp, train_hsv_saturation, train_hsv_value, train_grad_mag,
     train_shadow_mask, train_lightness, train_blue_yellow, train_green_red, train_x, train_y, train_z,
     train_masks) = zip(*train_paths)

    (val_rgb, val_ndwi, val_canny, val_lbp, val_hsv_saturation, val_hsv_value, val_grad_mag,
     val_shadow_mask, val_lightness, val_blue_yellow, val_green_red, val_x, val_y, val_z,
     val_masks) = zip(*val_paths)

    train_rgb = list(train_rgb)
    train_ndwi = list(train_ndwi)
    train_canny = list(train_canny)
    train_lbp = list(train_lbp)
    train_hsv_saturation = list(train_hsv_saturation)
    train_hsv_value = list(train_hsv_value)
    train_grad_mag = list(train_grad_mag)
    train_shadow_mask = list(train_shadow_mask)
    train_lightness = list(train_lightness)
    train_blue_yellow = list(train_blue_yellow)
    train_green_red = list(train_green_red)
    train_x = list(train_x)
    train_y = list(train_y)
    train_z = list(train_z)
    train_masks = list(train_masks)

    val_rgb = list(val_rgb)
    val_ndwi = list(val_ndwi)
    val_canny = list(val_canny)
    val_lbp = list(val_lbp)
    val_hsv_saturation = list(val_hsv_saturation)
    val_hsv_value = list(val_hsv_value)
    val_grad_mag = list(val_grad_mag)
    val_shadow_mask = list(val_shadow_mask)
    val_lightness = list(val_lightness)
    val_blue_yellow = list(val_blue_yellow)
    val_green_red = list(val_green_red)
    val_x = list(val_x)
    val_y = list(val_y)
    val_z = list(val_z)
    val_masks = list(val_masks)

    print(f"Total samples: {len(all_data_paths)}")
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")

    train_dataset = tf.data.Dataset.from_tensor_slices((train_rgb, train_ndwi, train_canny, train_lbp,
                                                        train_hsv_saturation, train_hsv_value, train_grad_mag,
                                                        train_shadow_mask, train_lightness, train_blue_yellow,
                                                        train_green_red, train_x, train_y, train_z, train_masks)
    )

    train_dataset = train_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    # train_dataset = train_dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)

    train_dataset = train_dataset.cache()  # Cache training data
    train_dataset = train_dataset.shuffle(buffer_size=len(train_paths), seed=ModelConfig.SEED)  # Shuffle the entire training set
    train_dataset = train_dataset.batch(ModelConfig.BATCH_SIZE)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_rgb, val_ndwi, val_canny, val_lbp, val_hsv_saturation,
                                                      val_hsv_value, val_grad_mag, val_shadow_mask, val_lightness,
                                                      val_blue_yellow, val_green_red, val_x, val_y, val_z, val_masks)
    )
    val_dataset = val_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.cache()  # Cache validation data
    # Do NOT shuffle the validation set typically, but batch it.
    val_dataset = val_dataset.batch(ModelConfig.BATCH_SIZE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    print("\nTrain Dataset element spec:", train_dataset.element_spec)
    print("Validation Dataset element spec:", val_dataset.element_spec)

    for combined_images_batch, masks_batch in train_dataset.take(1):
        print(f"\nTrain batch combined input shape: {combined_images_batch.shape}")
        print(f"Train batch masks shape: {masks_batch.shape}")

    for combined_images_batch, masks_batch in val_dataset.take(1):
        print(f"\nValidation batch combined input shape: {combined_images_batch.shape}")
        print(f"Validation batch masks shape: {masks_batch.shape}")

    return train_dataset, val_dataset

def load_and_preprocess_image(rgb_image_path, ndwi_image_path, canny_image_path, lbp_image_path,
                              hsv_saturation_image_path, hsv_value_image_path, grad_mag_image_path,
                              shadow_mask_image_path, lightness_image_path, blue_yellow_image_path,
                              green_red_image_path, x_image_path, y_image_path, z_image_path,
                              mask_path):
    rgb_image = tf.io.read_file(rgb_image_path)
    rgb_image = tf.image.decode_image(rgb_image, channels=3, expand_animations=False)
    rgb_image = tf.image.resize(rgb_image, [ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH])
    rgb_image = tf.cast(rgb_image, tf.float32) / 255.0

    ndwi_image = tf.io.read_file(ndwi_image_path)
    ndwi_image = tf.image.decode_image(ndwi_image, channels=1, expand_animations=False)
    ndwi_image = tf.image.resize(ndwi_image, [ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH])
    ndwi_image = tf.cast(ndwi_image, tf.float32) / 255.0

    canny_image = tf.io.read_file(canny_image_path)
    canny_image = tf.image.decode_image(canny_image, channels=1, expand_animations=False)
    canny_image = tf.image.resize(canny_image, [ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH])
    canny_image = tf.cast(canny_image, tf.float32) / 255.0

    lbp_image = tf.io.read_file(lbp_image_path)
    lbp_image = tf.image.decode_image(lbp_image, channels=1, expand_animations=False)
    lbp_image = tf.image.resize(lbp_image, [ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH])
    lbp_image = tf.cast(lbp_image, tf.float32) / 255.0

    hsv_saturation_image = tf.io.read_file(hsv_saturation_image_path)
    hsv_saturation_image = tf.image.decode_image(hsv_saturation_image, channels=1, expand_animations=False)
    hsv_saturation_image = tf.image.resize(hsv_saturation_image, [ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH])
    hsv_saturation_image = tf.cast(hsv_saturation_image, tf.float32) / 255.0

    hsv_value_image = tf.io.read_file(hsv_value_image_path)
    hsv_value_image = tf.image.decode_image(hsv_value_image, channels=1, expand_animations=False)
    hsv_value_image = tf.image.resize(hsv_value_image, [ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH])
    hsv_value_image = tf.cast(hsv_value_image, tf.float32) / 255.0

    grad_mag_image = tf.io.read_file(grad_mag_image_path)
    grad_mag_image = tf.image.decode_image(grad_mag_image, channels=1, expand_animations=False)
    grad_mag_image = tf.image.resize(grad_mag_image, [ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH])
    grad_mag_image = tf.cast(grad_mag_image, tf.float32) / 255.0

    shadow_mask_image = tf.io.read_file(shadow_mask_image_path)
    shadow_mask_image = tf.image.decode_image(shadow_mask_image, channels=1, expand_animations=False)
    shadow_mask_image = tf.image.resize(shadow_mask_image, [ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH])
    shadow_mask_image = tf.cast(shadow_mask_image, tf.float32) / 255.0

    lightness_image = tf.io.read_file(lightness_image_path)
    lightness_image = tf.image.decode_image(lightness_image, channels=1, expand_animations=False)
    lightness_image = tf.image.resize(lightness_image, [ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH])
    lightness_image = tf.cast(lightness_image, tf.float32) / 255.0

    blue_yellow_image = tf.io.read_file(blue_yellow_image_path)
    blue_yellow_image = tf.image.decode_image(blue_yellow_image, channels=1, expand_animations=False)
    blue_yellow_image = tf.image.resize(blue_yellow_image, [ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH])
    blue_yellow_image = tf.cast(blue_yellow_image, tf.float32) / 255.0

    green_red_image = tf.io.read_file(green_red_image_path)
    green_red_image = tf.image.decode_image(green_red_image, channels=1, expand_animations=False)
    green_red_image = tf.image.resize(green_red_image, [ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH])
    green_red_image = tf.cast(green_red_image, tf.float32) / 255.0

    x_image = tf.io.read_file(x_image_path)
    x_image = tf.image.decode_image(x_image, channels=1, expand_animations=False)
    x_image = tf.image.resize(x_image, [ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH])
    x_image = tf.cast(x_image, tf.float32) / 255.0

    y_image = tf.io.read_file(y_image_path)
    y_image = tf.image.decode_image(y_image, channels=1, expand_animations=False)
    y_image = tf.image.resize(y_image, [ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH])
    y_image = tf.cast(y_image, tf.float32) / 255.0

    z_image = tf.io.read_file(z_image_path)
    z_image = tf.image.decode_image(z_image, channels=1, expand_animations=False)
    z_image = tf.image.resize(z_image, [ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH])
    z_image = tf.cast(z_image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
    mask = tf.image.resize(mask, [ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH],
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.cast(mask, tf.int32)

    combined_input = tf.concat([rgb_image, ndwi_image, canny_image, lbp_image, hsv_saturation_image,
                                hsv_value_image, grad_mag_image, shadow_mask_image, lightness_image,
                                blue_yellow_image, green_red_image, x_image, y_image, z_image], axis=-1)
    return combined_input, mask

def select_channels(input_tensor, output_mask):
    """
    Selects specific channels from the input tensor.

    Args:
        input_tensor: A tensor of shape (height, width, 16).
        output_mask: The corresponding segmentation mask (height, width, 1).

    Returns:
        A tuple (selected_input_tensor, output_mask) where selected_input_tensor
        has shape (height, width, 6).
    """
    # Use tf.gather to select channels along the last axis (axis=2)
    # The batch dimension (None) is handled automatically.
    selected_channel_indices = ModelConfig.CHANNELS
    selected_channel_indices_tf = tf.constant(selected_channel_indices, dtype=tf.int32)
    selected_input = tf.gather(input_tensor, selected_channel_indices_tf, axis=-1)
    return selected_input, output_mask

def filter_dataset(original_dataset):
    filtered_dataset = original_dataset.map(select_channels, num_parallel_calls=tf.data.AUTOTUNE)
    print("\nFiltered Dataset Element Spec:", filtered_dataset.element_spec)
    return filtered_dataset

# Channel Name List ["RED", "GREEN", "BLUE", "NDWI", "Canny", "LBP", "HSV Saturation", "HSV Value", "GradMag", "Shadow Mask", "Lightness", "Blue Yellow", "Green Red", "X", "Y", "Z"]
# Define index list in config.yaml based on above order
if __name__ == '__main__':
    config_path = '../unet_wsl/config.yaml'

    train_dataset, validation_dataset = load_datasets(config_path)

    filter_training_dataset = filter_dataset(train_dataset)
    filter_validation_dataset = filter_dataset(validation_dataset)

    print("For training dataset")
    for inputs, masks in filter_training_dataset.take(1):
        print("\nShape of inputs after channel selection:", inputs.shape)
        print("Shape of masks (unchanged):", masks.shape)

    print("For validation dataset")
    for inputs, masks in filter_training_dataset.take(1):
        print("\nShape of inputs after channel selection:", inputs.shape)
        print("Shape of masks (unchanged):", masks.shape)


