import os
from datetime import datetime

import keras.callbacks_v1
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import (Callback,
                             CSVLogger)
import numpy as np

from models.segnet.data import load_dataset
from model import segnet_model, build_rgb_segnet, extend_to_16_channel_model
from models.common_utils.loss_functions import  recall_m, precision_m, f1_score, combined_masked_dice_focal_loss
from models.common_utils.accuracy_functions import calculate_accuracy
from models.wsl.wsl_utils import show_image

LOG_DIR = "C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\segnet_VGG16\logs"
CKPT_DIR = "C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\segnet_VGG16\ckpt"
OUTPUT_DIR = "C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\segnet_VGG16\output"

def train_model(epoch_count, batch_size, X_train, y_train, X_val, y_val, num_channels, size = (256, 256), restore=True):
    print(f'X_train shape : {X_train.shape}')
    print(f'y_train shape : {y_train.shape}')

    os.makedirs(LOG_DIR, exist_ok=True)
    tensorboard = keras.callbacks.TensorBoard(
        log_dir = os.path.join(LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S")),
        histogram_freq=1
    )

    os.makedirs(CKPT_DIR, exist_ok=True)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        f"{CKPT_DIR}/SegNet_VGG16_best_model.h5",  # or "best_model.keras"
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,  # set to True if you want only weights
        mode='min',
        verbose=1
    )

    cbs = [
        CSVLogger(LOG_DIR+'/segnet_logs.csv', separator=',', append=False),
        checkpoint_cb,
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
        saved_model_path = f"{CKPT_DIR}/SegNet_VGG16_best_model.h5"
        print(f"Restoring from {saved_model_path}")
        return keras.models.load_model(saved_model_path,
                                    custom_objects={'recall_m': recall_m,
                                                    'precision_m': precision_m,
                                                    'f1_score': f1_score,
                                                    'combined_masked_dice_focal_loss': combined_masked_dice_focal_loss})
    else:
        print("Creating fresh model")
        return segnet_model(width, height, num_channels)

def load_with_trained_model(X_val, y_val):
    saved_model_path = f"{CKPT_DIR}/SegNet_VGG16_best_model.h5"
    print(f"Restoring from {saved_model_path}")
    model = keras.models.load_model(saved_model_path,
                                    custom_objects={'recall_m': recall_m,
                                                    'precision_m': precision_m,
                                                    'f1_score': f1_score,
                                                    'combined_masked_dice_focal_loss': combined_masked_dice_focal_loss})
    for i in range(len(X_val)):
        actual_mask = y_val[i]
        image = X_val[i]
        new_image = np.expand_dims(image, axis=0)
        y_pred = model.predict(new_image)
        pred_mask = reconstruct_mask(y_pred)
        calculate_accuracy(actual_mask, y_pred)
        # pred_class_map = np.argmax(pred_mask, axis=-1)
        plt.figure(figsize=(10, 8))
        plt.subplot(1, 3, 1)
        rgb_image = image[:, :, :3]
        show_image(OUTPUT_DIR, rgb_image, index=i, title="Original_Image", save=False)
        plt.subplot(1, 3, 2)
        show_image(OUTPUT_DIR, actual_mask, index=i, title="Actual_Mask", save=False)
        plt.subplot(1, 3, 3)
        show_image(OUTPUT_DIR, pred_mask, index=i, title="Predicted_Mask", save=False)
        plt.tight_layout()
        plt.show()

def reconstruct_mask(y_pred):
    pred_mask = np.argmax(y_pred[0], axis=-1)  # shape: (H, W)
    # Convert class mask to color
    colormap = np.array([
        [0, 0, 0],  # class 0: background
        [0, 0, 255],  # class 1: water
    ], dtype=np.uint8)
    return colormap[pred_mask]

def train_rgb_model(epoch_count, batch_size, X_train, y_train, X_val, y_val, input_shape=(512, 512, 3), num_classes=1, restore=False):
    print(f'X_train shape : {X_train.shape}')
    print(f'y_train shape : {y_train.shape}')

    os.makedirs(LOG_DIR, exist_ok=True)
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=os.path.join(LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S")),
        histogram_freq=1
    )

    os.makedirs(CKPT_DIR, exist_ok=True)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        f"{CKPT_DIR}/SegNet_VGG16_RGB_best_model.h5",  # or "best_model.keras"
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,  # set to True if you want only weights
        mode='min',
        verbose=1
    )

    cbs = [
        CSVLogger(LOG_DIR + '/segnet_vgg16_rgb_logs.csv', separator=',', append=False),
        checkpoint_cb,
        tensorboard
    ]
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    model = make_or_restore_model_rgb_model(restore, input_shape, num_classes)

    history = model.fit(
        X_train,
        y_train,
        epochs=epoch_count,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=cbs
    )

    print(f"SegNet-VGG16-RGB Model History: {history.history}")

def make_or_restore_model_rgb_model(restore, input_shape, num_classes):
    if restore:
        saved_model_path = f"{CKPT_DIR}/SegNet_VGG16_RGB_best_model.h5"
        print(f"Restoring from {saved_model_path}")
        return keras.models.load_model(saved_model_path)
    else:
        print("Creating fresh model")
        return build_rgb_segnet(input_shape=input_shape,
                                num_classes=num_classes)

def make_or_restore_16_model(restore, input_shape, pretrained_model):
    if restore:
        saved_model_path = f"{CKPT_DIR}/SegNet_VGG16_best_model.h5"
        print(f"Restoring from {saved_model_path}")
        return keras.models.load_model(saved_model_path,
                                    custom_objects={'recall_m': recall_m,
                                                    'precision_m': precision_m,
                                                    'f1_score': f1_score,
                                                    'combined_masked_dice_focal_loss': combined_masked_dice_focal_loss})
    else:
        print("Creating fresh model")
        channel_16_model = extend_to_16_channel_model(pretrained_model, input_shape=input_shape)
        return channel_16_model

def train_extend_to_16_channel_model(epoch_count, batch_size, rgb_model_path, input_shape=(512, 512, 16), restore=False):
    pretrained_model = keras.models.load_model(rgb_model_path)

    os.makedirs(LOG_DIR, exist_ok=True)
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=os.path.join(LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S")),
        histogram_freq=1
    )

    os.makedirs(CKPT_DIR, exist_ok=True)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        f"{CKPT_DIR}/SegNet_VGG16_best_model.h5",  # or "best_model.keras"
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,  # set to True if you want only weights
        mode='min',
        verbose=1
    )

    cbs = [
        CSVLogger(LOG_DIR + '/segnet_vgg16_logs.csv', separator=',', append=False),
        checkpoint_cb,
        tensorboard
    ]
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    model = make_or_restore_16_model(restore, input_shape, pretrained_model)

    history = model.fit(
        X_train,
        y_train,
        epochs=epoch_count,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=cbs
    )

    print(f"SegNet-VGG16-16-Channel Model History: {history.history}")

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

if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(f"physical_devices : {physical_devices}")
    print(tf.__version__)
    print(tf.executing_eagerly())
    if len(physical_devices) > 0:
        (RGB_X_train, RGB_y_train), (RGB_X_val, RGB_y_val) = load_dataset("../../input/samples/crookstown/images",
                                                          size = (512, 512),
                                                          file_extension="jpg",
                                                          channels=['RED', 'GREEN', 'BLUE'],
                                                          percentage=0.7)
        train_rgb_model(2, 2, RGB_X_train, RGB_y_train, RGB_X_val, RGB_y_val)
        (X_train, y_train), (X_val, y_val) = load_dataset("../../input/samples/crookstown/images",
                                                          size=(512, 512),
                                                          file_extension="jpg",
                                                          channels=['RED', 'GREEN', 'BLUE', 'NDWI', 'Canny', 'LBP',
                                                                    'HSV Saturation', 'HSV Value', 'GradMag',
                                                                    'Shadow Mask', 'Lightness', 'GreenRed',
                                                                    'BlueYellow', 'X', 'Y', 'Z'],
                                                          percentage=0.7)
        rgb_model_path = f"{CKPT_DIR}/SegNet_VGG16_RGB_best_model.h5"
        train_extend_to_16_channel_model(2, 2, rgb_model_path)
        load_with_trained_model(X_val, y_val)

# if __name__ == "__main__":
#     print(tf.config.list_physical_devices('GPU'))
#     physical_devices = tf.config.experimental.list_physical_devices('GPU')
#     print(f"physical_devices : {physical_devices}")
#     print(tf.__version__)
#     print(tf.executing_eagerly())
#     image_size = (256, 256) # actual size is (5280, 3956)
#     epochs = 25
#     batch_size = 4
#     channels = ['RED', 'GREEN', 'BLUE', 'NDWI', 'Canny', 'LBP', 'HSV Saturation', 'HSV Value', 'GradMag',
#                 'Shadow Mask', 'Lightness', 'GreenRed', 'BlueYellow', 'X', 'Y', 'Z']
#     channel_count = len(channels)
#     if len(physical_devices) > 0:
#         (X_train, y_train), (X_val, y_val) = load_dataset("../../input/samples/crookstown/images",
#                                                           size = image_size,
#                                                           file_extension="jpg",
#                                                           channels=channels,
#                                                           percentage=0.7)
#         train_model(epochs, batch_size, X_train, y_train, X_val, y_val, channel_count,
#                     size = image_size,
#                     restore=False)
#         load_with_trained_model(X_val, y_val)
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
#         train_model(epochs, batch_size, X_train, y_train, X_val, y_val, channel_count,
#                     size=image_size,
#                     restore=False)
#         load_with_trained_model(X_val, y_val)

