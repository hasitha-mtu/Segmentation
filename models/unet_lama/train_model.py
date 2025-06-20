import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.callbacks import (Callback,
                             CSVLogger)

from model import (unet_lama, Up, Down, FFCBlock, DoubleConv, OutConv, UNetWithLaMaFeaturesTF,
                   psnr_metric, ssim_metric)
from data import load_dataset
from models.unet_wsl.wsl_utils import show_image
from PIL import Image

LOG_DIR = "C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet_lama\logs"
CKPT_DIR = "C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet_lama\ckpt"
OUTPUT_DIR = "C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet_lama\output"

def train_model(epoch_count, batch_size, X_train, y_train, X_val, y_val, width, height,
                input_channels, output_channels, restore=True):
    print(f'X_train shape : {X_train.shape}')
    print(f'y_train shape : {y_train.shape}')

    os.makedirs(LOG_DIR, exist_ok=True)
    tensorboard = keras.callbacks.TensorBoard(
        log_dir = os.path.join(LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S")),
        histogram_freq=1
    )

    os.makedirs(CKPT_DIR, exist_ok=True)

    checkpoint_filepath = os.path.join(CKPT_DIR, 'unet_lama_best_model')

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_mae',  # Monitor validation MAE
        mode='min',  # Save when validation MAE is minimized
        save_best_only=True,  # Only save the best model found so far
        save_weights_only=False,  # Save the entire model (architecture + weights + optimizer)
        save_format="tf",  # CRUCIAL: Save in TensorFlow SavedModel format for subclassed models
        verbose=1  # Print messages when saving
    )

    cbs = [
        CSVLogger(LOG_DIR+'/unet_lama_logs.csv', separator=',', append=False),
        model_checkpoint_callback,
        tensorboard
    ]
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = make_or_restore_model(restore, width, height, input_channels, output_channels)

    history = model.fit(
                    X_train,
                    y_train,
                    epochs=epoch_count,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val),
                    callbacks=cbs
                )

    print(history.history)
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(accuracy) + 1)
    plt.plot(epochs, accuracy, "bo", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()
    return None

def make_or_restore_model(restore, width, height, input_channels, output_channels):
    if restore:
        custom_objects = {
            'FFCBlock': FFCBlock,
            'DoubleConv': DoubleConv,
            'Down': Down,
            'Up': Up,
            'OutConv': OutConv,
            'UNetWithLaMaFeaturesTF': UNetWithLaMaFeaturesTF,
            # Add the custom metric functions here as well if you want them re-associated when loading
            'psnr_metric': psnr_metric,
            'ssim_metric': ssim_metric
        }
        saved_model_path = os.path.join(CKPT_DIR, "unet_lama_best_model")
        print(f"Restoring from {saved_model_path}")
        return keras.models.load_model(saved_model_path,
                                       custom_objects = custom_objects,
                                       compile=True)
    else:
        print("Creating fresh model")
        return unet_lama(width, height, input_channels, output_channels)

def load_with_trained_model(X_val, y_val):
    custom_objects = {
        'FFCBlock': FFCBlock,
        'DoubleConv': DoubleConv,
        'Down': Down,
        'Up': Up,
        'OutConv': OutConv,
        'UNetWithLaMaFeaturesTF': UNetWithLaMaFeaturesTF,
        'psnr_metric': psnr_metric,
        'ssim_metric': ssim_metric
    }

    try:
        latest_checkpoint_path = os.path.join(CKPT_DIR, "unet_lama_best_model")
        if latest_checkpoint_path:
            loaded_model = tf.keras.models.load_model(
                latest_checkpoint_path,
                custom_objects=custom_objects,
                compile = True
            )
            print(f"Model loaded successfully from: {latest_checkpoint_path}")
            for i in range(len(X_val)):
                image_array = X_val[i]
                actual_mask = y_val[i]
                #
                # print(f"Original image shape: {image_array.shape}, dtype: {image_array.dtype}")
                # print(f"Original image pixel value example (top-left): {image_array[0, 0, :]}")
                #
                # # --- DEBUG PRINTS: Check image_array before normalization ---
                # print(f"DEBUG (pre-norm): image_array shape: {image_array.shape}")
                # print(f"DEBUG (pre-norm): image_array dtype: {image_array.dtype}")
                # print(f"DEBUG (pre-norm): image_array min value: {np.min(image_array)}")
                # print(f"DEBUG (pre-norm): image_array max value: {np.max(image_array)}")
                # # --- END DEBUG PRINTS ---
                #
                # # --- Step 2: Normalize the image to 0-1 range (ADJUSTED LOGIC) ---
                #
                # # Based on your debug output (`image_array max value: 1.0`),
                # # the image_array is ALREADY in the 0.0-1.0 range.
                # # Therefore, simply ensure it's float32 without dividing by 255.0.
                #
                # if image_array.max() > 1.0 or image_array.dtype == np.uint8:
                #     # This block executes if the image was NOT already normalized (e.g., 0-255 range)
                #     print("Detected unnormalized image (max > 1 or uint8 dtype). Normalizing now.")
                #     normalized_image_array = image_array.astype(np.float32) / 255.0
                # else:
                #     # This block executes if the image is ALREADY normalized (0.0-1.0 float values)
                #     print("Image appears to be already normalized (0.0-1.0 float). Skipping 255 division.")
                #     normalized_image_array = image_array.astype(np.float32)  # Just ensure it's float32
                #
                # print(f"Normalized image shape: {normalized_image_array.shape}, dtype: {normalized_image_array.dtype}")
                # print(f"Normalized image pixel value example (top-left): {normalized_image_array[0, 0, :]}")
                #
                # # --- Step 3: Add batch dimension (if your model expects it) ---
                #
                # # Most Keras models expect inputs in the format (batch_size, height, width, channels)
                # # If your image is (height, width, channels), you need to add a batch dimension.
                # input_for_model = np.expand_dims(normalized_image_array, axis=0)  # Adds a dimension at the beginning
                #
                # print(f"Input for model shape: {input_for_model.shape}, dtype: {input_for_model.dtype}")
                # print(f"Input for model min value: {np.min(input_for_model)}")
                # print(f"Input for model max value: {np.max(input_for_model)}")

                input_image_path = "../../input/samples/segnet_256/images/DJI_20250324092908_0001_V.jpg"
                image_pil = Image.open(input_image_path).convert('RGB')
                image_array = np.array(image_pil)

                if image_array.max() > 1.0 or image_array.dtype == np.uint8:
                    print("Detected unnormalized image (max > 1 or uint8 dtype). Normalizing now.")
                    normalized_image_array = image_array.astype(np.float32) / 255.0
                else:
                    print("Image appears to be already normalized (0.0-1.0 float). Skipping 255 division.")
                    normalized_image_array = image_array.astype(np.float32)

                print(f"Normalized image shape: {normalized_image_array.shape}, dtype: {normalized_image_array.dtype}")
                print(f"Normalized image pixel value example (top-left): {normalized_image_array[0, 0, :]}")

                # --- Step 3: Add batch dimension ---
                input_for_model = np.expand_dims(normalized_image_array, axis=0)

                pred_mask = loaded_model.predict(input_for_model)
                plt.figure(figsize=(10, 8))
                plt.subplot(1, 3, 1)
                rgb_image = image_array[:, :, :3]
                show_image(OUTPUT_DIR, rgb_image, index=i, title="Original_Image", save=True)
                plt.subplot(1, 3, 2)
                show_image(OUTPUT_DIR, actual_mask.squeeze(), index=i, title="Actual_Mask", save=True)
                plt.subplot(1, 3, 3)
                show_image(OUTPUT_DIR, pred_mask.squeeze(), index=i, title="Predicted_Mask", save=True)

                plt.tight_layout()
                plt.show()
        else:
            print("No checkpoints found to load.")

    except Exception as e:
        print(f"Failed to load model. Error: {e}")
        print("Ensure 'latest_checkpoint_path' is correct and custom_objects are properly defined.")

if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(f"physical_devices : {physical_devices}")
    print(tf.__version__)
    print(tf.executing_eagerly())
    image_size = (256, 256) # actual size is (5280, 3956)
    epochs = 25
    batch_size = 4
    # channels = ['RED', 'GREEN', 'BLUE', 'NDWI', 'Canny']
    channels = ['RED', 'GREEN', 'BLUE']
    channel_count = len(channels)
    if len(physical_devices) > 0:
        (X_train, y_train), (X_val, y_val) = load_dataset("../../input/samples/segnet_256/images",
                                                          size = image_size,
                                                          file_extension="jpg",
                                                          channels=channels,
                                                          percentage=0.7,
                                                          image_count=200)

        y_train = (y_train > 0.0).astype(np.float32)
        y_val = (y_val > 0.0).astype(np.float32)
        print(f"y_train unique values : {np.unique(y_train)}")
        print(f"y_val unique values : {np.unique(y_val)}")
        water_pixels = np.sum(y_val == 1.0)
        non_water_pixels = np.sum(y_val == 0.0)
        print(f"Water: {water_pixels}, Non-water: {non_water_pixels}")

        # train_model(25, batch_size, X_train, y_train, X_val, y_val,
        #             256, 256, 3, 3, restore=False)

        load_with_trained_model(X_val, y_val)

# if __name__ == "__main__":
#     print(tf.config.list_physical_devices('GPU'))
#     physical_devices = tf.config.experimental.list_physical_devices('GPU')
#     print(f"physical_devices : {physical_devices}")
#     print(tf.__version__)
#     print(tf.executing_eagerly())
#     if len(physical_devices) > 0:
#         num_samples = 100
#         sample_height = 256
#         sample_width = 256
#         sample_channels = 3
#         input_channels = 3
#         output_channels = 3
#
#         dummy_masked_images = tf.random.normal((num_samples, sample_height, sample_width, sample_channels))
#         # For original images, imagine they are the "clean" versions
#         dummy_original_images = tf.random.normal((num_samples, sample_height, sample_width, sample_channels))
#
#         dataset = tf.data.Dataset.from_tensor_slices((dummy_masked_images, dummy_original_images)).batch(
#             4)  # Batch size 4
#
#         print("\nStarting training (with dummy data):")
#         model = unet_lama(sample_width, sample_height, input_channels, output_channels)
#         history = model.fit(
#             dataset,
#             epochs=10,  # Number of training epochs
#             validation_data=dataset,  # In a real scenario, use a separate validation set
#             verbose=1
#         )
#         print("\nTraining complete!")