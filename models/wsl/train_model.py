import os
from datetime import datetime

import keras.callbacks_v1
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import (Callback,
                             ModelCheckpoint,
                             CSVLogger)
import numpy as np
from numpy.random import randint

from data import load_dataset
from model import unet_model
from models.common_utils.loss_functions import  recall_m, precision_m, f1_score, masked_dice_loss
from models.wsl.wsl_utils import show_image, overlay_mask_on_image, overlay_mask

LOG_DIR = "C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\wsl\logs"
CKPT_DIR = "C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\wsl\ckpt"
OUTPUT_DIR = "C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\wsl\output"

def train_model(epoch_count, batch_size, X_train, y_train, X_val, y_val, num_channels, size = (256, 256), restore=True):
    print(f'X_train shape : {X_train.shape}')
    print(f'y_train shape : {y_train.shape}')

    os.makedirs(LOG_DIR, exist_ok=True)
    tensorboard = keras.callbacks.TensorBoard(
        log_dir = os.path.join(LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S")),
        histogram_freq=1
    )

    os.makedirs(CKPT_DIR, exist_ok=True)

    cbs = [
        CSVLogger(LOG_DIR+'/wsl_unet_logs.csv', separator=',', append=False),
        ModelCheckpoint(CKPT_DIR+'/ckpt-{epoch}', save_freq="epoch"),
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
    accuracy = history.history["segmentation_output_accuracy"]
    val_accuracy = history.history["val_segmentation_output_accuracy"]
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
        checkpoints = [os.path.join(CKPT_DIR, name) for name in os.listdir("ckpt")]
        print(f"Checkpoints: {checkpoints}")
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            print(f"Restoring from {latest_checkpoint}")
            return keras.models.load_model(latest_checkpoint)
        else:
            print("Creating fresh model")
            return unet_model(width, height, num_channels)
    else:
        print("Creating fresh model")
        return unet_model(width, height, num_channels)

def load_with_trained_model(X_val, channels):
    checkpoints = [os.path.join(CKPT_DIR, name) for name in os.listdir("ckpt")]
    print(f"Checkpoints: {checkpoints}")
    weight_lists = []
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"Restoring from {latest_checkpoint}")
        model = keras.models.load_model(latest_checkpoint,
                                        custom_objects={'recall_m':recall_m,
                                                        'precision_m':precision_m,
                                                        'f1_score':f1_score,
                                                        'masked_dice_loss':masked_dice_loss})
        for i in range(len(X_val)):
            id = randint(len(X_val))
            image = X_val[id]
            pred_mask, attention_weights = model.predict(np.expand_dims(image, 0))
            plt.figure(figsize=(10, 8))
            plt.subplot(1, 2, 1)
            rgb_image = image[:, :, :3]
            show_image(OUTPUT_DIR, rgb_image, index=i, title="Original_Image", save=True)
            plt.subplot(1, 2, 2)
            blended_mask = overlay_mask(rgb_image, pred_mask, alpha=0.3)
            show_image(OUTPUT_DIR, blended_mask, index=i, title="Predicted_Mask", save=True)

            print(f'Attention weights: {attention_weights.flatten()}')
            weight_lists.append(attention_weights.flatten().tolist())
            plt.tight_layout()
            plt.show()
    else:
        print("No preloaded model")
    print(f"weight_lists: {weight_lists}")
    plot_attention_weights(channels, weight_lists)

def plot_attention_weights(channels, weight_lists):
    result = [sum(values) for values in zip(*weight_lists)]
    print(f"plot_attention_weights|result: {result}")
    percentages = [(value/sum(result))*100 for value in result]
    print(f"plot_attention_weights|percentages: {percentages}")
    plt.bar(range(1, len(channels) + 1), percentages)
    plt.xticks(range(1, len(channels) + 1), channels)
    plt.ylabel("Channel Attention Weight")
    plt.title("Learned Attention per Channel")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(f"physical_devices : {physical_devices}")
    print(tf.__version__)
    print(tf.executing_eagerly())
    image_size = (512, 512) # actual size is (5280, 3956)
    epochs = 50
    batch_size = 4
    # channels = ['RED', 'GREEN', 'BLUE', 'NDWI', 'Canny', 'LBP', 'HSV Saturation', 'HSV Value', 'GradMag', 'Shadow Mask']
    channels = ['RED', 'GREEN', 'BLUE', 'NDWI']
    channel_count = len(channels)
    if len(physical_devices) > 0:
        (X_train, y_train), (X_val, y_val) = load_dataset("../../input/samples/crookstown/images",
                                                          size = image_size,
                                                          file_extension="jpg",
                                                          channels=channels,
                                                          percentage=0.7)
        train_model(epochs, batch_size, X_train, y_train, X_val, y_val, channel_count,
                    size = image_size,
                    restore=False)
        load_with_trained_model(X_val, channels)

