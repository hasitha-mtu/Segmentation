import os
import io
from datetime import datetime

import keras.callbacks_v1
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import (Callback,
                             CSVLogger)
import numpy as np
from numpy.random import randint

from models.segnet.data import load_dataset
from model import segnet_model
from models.common_utils.loss_functions import  recall_m, precision_m, f1_score, combined_loss_with_edge
from models.common_utils.accuracy_functions import calculate_accuracy, evaluate_prediction
from models.unet_wsl.wsl_utils import show_image
from model import MaxUnpooling2D, MaxPoolingWithArgmax2D

LOG_DIR = "C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\segnet\logs"
CKPT_DIR = "C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\segnet\ckpt"
OUTPUT_DIR = "C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\segnet\output"

def train_model(epoch_count, batch_size, X_train, y_train, X_val, y_val, num_channels, size = (256, 256), restore=True):
    print(f'X_train shape : {X_train.shape}')
    print(f'y_train shape : {y_train.shape}')

    os.makedirs(LOG_DIR, exist_ok=True)
    tensorboard = keras.callbacks.TensorBoard(
        log_dir = os.path.join(LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S")),
        histogram_freq=1
    )

    os.makedirs(CKPT_DIR, exist_ok=True)

    val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(4)
    debug_callback = DebugImageLogger(writer, val_data)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        f"{CKPT_DIR}/SegNet_best_model.h5",  # or "best_model.keras"
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,  # set to True if you want only weights
        mode='max',
        verbose=1
    )

    cbs = [
        CSVLogger(LOG_DIR+'/segnet_logs.csv', separator=',', append=False),
        checkpoint_cb,
        debug_callback,
        tensorboard
    ]
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = make_or_restore_model(restore, num_channels, size)

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

def make_or_restore_model(restore, num_channels, size):
    (width, height) = size
    if restore:
        saved_model_path = f"{CKPT_DIR}/SegNet_best_model.h5"
        print(f"Restoring from {saved_model_path}")
        return keras.models.load_model(saved_model_path,
                                    custom_objects={'recall_m': recall_m,
                                                    'precision_m': precision_m,
                                                    'f1_score': f1_score,
                                                    'combined_loss_with_edge': combined_loss_with_edge,
                                                    'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D,
                                                    'MaxUnpooling2D': MaxUnpooling2D})
    else:
        print("Creating fresh model")
        return segnet_model(width, height, num_channels)

def load_with_trained_model(X_val, y_val):
    saved_model_path = f"{CKPT_DIR}/SegNet_best_model.h5"
    print(f"Restoring from {saved_model_path}")
    model = keras.models.load_model(saved_model_path,
                                    custom_objects={'recall_m': recall_m,
                                                    'precision_m': precision_m,
                                                    'f1_score': f1_score,
                                                    'combined_loss_with_edge': combined_loss_with_edge,
                                                    'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D,
                                                    'MaxUnpooling2D': MaxUnpooling2D})
    for i in range(len(X_val)):
        actual_mask = y_val[i]
        image = X_val[i]
        new_image = np.expand_dims(image, axis=0)
        y_pred = model.predict(new_image)
        pred_mask = reconstruct_mask(y_pred, i)
        calculate_accuracy(actual_mask, y_pred)
        plt.figure(figsize=(10, 8))
        plt.subplot(1, 3, 1)
        rgb_image = image[:, :, :3]
        show_image(OUTPUT_DIR, rgb_image, index=i, title="Original_Image", save=False)
        plt.subplot(1, 3, 2)
        show_image(OUTPUT_DIR, actual_mask, index=i, title="Actual_Mask", save=False)
        plt.subplot(1, 3, 3)
        show_image(OUTPUT_DIR, pred_mask, index=i, title="Predicted_Mask", save=True)
        plt.tight_layout()
        plt.show()

def reconstruct_mask(y_pred, i):
    pred_mask = np.argmax(y_pred[0], axis=-1)  # shape: (H, W)
    # Convert class mask to color
    colormap = np.array([
        [0, 0, 0],  # class 0: background
        [0, 0, 255],  # class 1: water
    ], dtype=np.uint8)
    return colormap[pred_mask]


class DebugImageLogger(tf.keras.callbacks.Callback):
    def __init__(self, writer, val_data, freq=1):
        super().__init__()
        self.writer = writer
        self.val_data = val_data
        self.freq = freq

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.freq != 0:
            return

        x_val, y_val = next(iter(self.val_data))
        y_pred = self.model.predict(x_val)

        # Recompute mask if needed
        mask = tf.cast(tf.not_equal(y_val, 255.0), tf.float32)

        log_debug_images(self.writer, step=epoch, inputs=x_val,
                         y_true=y_val, y_pred=y_pred, mask=mask)

def log_debug_images(writer, step, inputs, y_true, y_pred, mask=None, max_images=3):
    """
    Log input, prediction, and ground truth side-by-side to TensorBoard.

    Args:
        writer: tf.summary.create_file_writer
        step: global training step
        inputs: model inputs (e.g., RGB or composite feature tensor) [B, H, W, C]
        y_true: ground truth labels [B, H, W, 1]
        y_pred: predictions [B, H, W, 1]
        mask: optional mask [B, H, W, 1]
        max_images: number of images to log
    """
    inputs = tf.image.convert_image_dtype(inputs, tf.uint8)
    y_true = tf.cast(y_true * 255, tf.uint8)
    y_pred = tf.cast(y_pred * 255, tf.uint8)

    if mask is not None:
        mask = tf.cast(mask * 255, tf.uint8)

    for i in range(min(max_images, inputs.shape[0])):
        fig, axes = plt.subplots(1, 4 if mask is not None else 3, figsize=(16, 4))

        axes[0].imshow(inputs[..., :3][i])
        axes[0].set_title("Input")
        axes[0].axis('off')

        axes[1].imshow(tf.squeeze(y_true[i]), cmap='gray')
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')

        axes[2].imshow(tf.squeeze(y_pred[i]), cmap='gray')
        axes[2].set_title("Prediction")
        axes[2].axis('off')

        if mask is not None:
            axes[3].imshow(tf.squeeze(mask[i]), cmap='gray')
            axes[3].set_title("Mask")
            axes[3].axis('off')

        # Convert to image tensor
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)  # [1, H, W, 4]
        with writer.as_default():
            tf.summary.image(f"Debug/Image_{i}", image, step=step)
        plt.close(fig)

if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(f"physical_devices : {physical_devices}")
    print(tf.__version__)
    print(tf.executing_eagerly())
    image_size = (256, 256) # actual size is (5280, 3956)
    epochs = 50
    batch_size = 4
    channels = ['RED', 'GREEN', 'BLUE', 'NDWI', 'Canny', 'LBP', 'HSV Saturation', 'HSV Value', 'GradMag',
                'Shadow Mask', 'Lightness', 'GreenRed', 'BlueYellow', 'X', 'Y', 'Z']
    channel_count = len(channels)
    if len(physical_devices) > 0:
        (X_train, y_train), (X_val, y_val) = load_dataset("../../input/samples/segnet_512/images",
                                                          size = image_size,
                                                          file_extension="jpg",
                                                          channels=channels,
                                                          percentage=0.7)
        log_dir = "logs/debug"
        writer = tf.summary.create_file_writer(log_dir)
        train_model(epochs, batch_size, X_train, y_train, X_val, y_val, channel_count,
                    size = image_size,
                    restore=False)
        load_with_trained_model(X_val, y_val)
#
# if __name__ == "__main__":
#     print(tf.config.list_physical_devices('GPU'))
#     physical_devices = tf.config.experimental.list_physical_devices('GPU')
#     print(f"physical_devices : {physical_devices}")
#     print(tf.__version__)
#     print(tf.executing_eagerly())
#     image_size = (512, 512) # actual size is (5280, 3956)
#     channels = ['RED', 'GREEN', 'BLUE', 'NDWI', 'Canny', 'LBP', 'HSV Saturation', 'HSV Value', 'GradMag',
#                 'Shadow Mask', 'Lightness', 'GreenRed', 'BlueYellow', 'X', 'Y', 'Z']
#     channel_count = len(channels)
#     if len(physical_devices) > 0:
#         (X_train, y_train), (X_val, y_val) = load_dataset("../../input/samples1/crookstown/images",
#                                                           size = image_size,
#                                                           file_extension="jpg",
#                                                           channels=channels,
#                                                           percentage=0.7)
#         print(f'X_train shape : {X_train.shape}')
#         print(f'y_train shape : {y_train.shape}')
#         print(f'X_val shape : {X_val.shape}')
#         print(f'y_val shape : {y_val.shape}')
#         load_with_trained_model(X_val, y_val)

# if __name__ == "__main__":
#     print(tf.config.list_physical_devices('GPU'))
#     physical_devices = tf.config.experimental.list_physical_devices('GPU')
#     print(f"physical_devices : {physical_devices}")
#     print(tf.__version__)
#     print(tf.executing_eagerly())
#     image_size = (512, 512) # actual size is (5280, 3956)
#     epochs = 2
#     batch_size = 2
#     channels = ['RED', 'GREEN', 'BLUE', 'NDWI', 'Canny', 'LBP', 'HSV Saturation', 'HSV Value', 'GradMag',
#                 'Shadow Mask', 'Lightness', 'GreenRed', 'BlueYellow', 'X', 'Y', 'Z']
#     channel_count = len(channels)
#     if len(physical_devices) > 0:
#         (X_train, y_train), (X_val, y_val) = load_dataset("../../input/samples1/crookstown/images",
#                                                           size = image_size,
#                                                           file_extension="jpg",
#                                                           channels=channels,
#                                                           percentage=0.7)
#         log_dir = "logs/debug"
#         writer = tf.summary.create_file_writer(log_dir)
#         train_model(epochs, batch_size, X_train, y_train, X_val, y_val, channel_count,
#                     size=image_size,
#                     restore=False)
#         load_with_trained_model(X_val, y_val)

