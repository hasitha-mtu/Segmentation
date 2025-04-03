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
from tensorflow.python.ops.metrics_impl import percentage_below

from data import load_dataset
from model import unet_model
from models.common_utils.loss_functions import  recall_m, precision_m, f1_score, masked_dice_loss

LOG_DIR = "C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\wsl\logs"
CKPT_DIR = "C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\wsl\ckpt"

def train_model(X_train, y_train, X_val, y_val, num_channels, restore=True):
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
        model = make_or_restore_model(restore, num_channels)

    history = model.fit(
                    X_train,
                    y_train,
                    epochs=20,
                    batch_size=16,
                    validation_data=(X_val, y_val),
                    callbacks=cbs
                )

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

def make_or_restore_model(restore, num_channels):
    if restore:
        checkpoints = [os.path.join(CKPT_DIR, name) for name in os.listdir("ckpt")]
        print(f"Checkpoints: {checkpoints}")
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            print(f"Restoring from {latest_checkpoint}")
            return keras.models.load_model(latest_checkpoint)
        else:
            print("Creating fresh model")
            return unet_model(256, 256, num_channels)
    else:
        print("Creating fresh model")
        return unet_model(256, 256, num_channels)

def load_with_trained_model(X_val, y_val, count=5):
    checkpoints = [os.path.join(CKPT_DIR, name) for name in os.listdir("ckpt")]
    print(f"Checkpoints: {checkpoints}")
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"Restoring from {latest_checkpoint}")
        model = keras.models.load_model(latest_checkpoint,
                                        custom_objects={'recall_m':recall_m,
                                                        'precision_m':precision_m,
                                                        'f1_score':f1_score,
                                                        'masked_dice_loss':masked_dice_loss})
        for i in range(count):
            id = randint(len(X_val))
            image = X_val[id]
            mask = y_val[id]
            pred_mask = model.predict(np.expand_dims(image, 0))[0]
            plt.figure(figsize=(10, 8))
            plt.subplot(1, 3, 1)
            rgb_image = image[:, :, :3]
            show_image(rgb_image, title="Original Image")
            plt.subplot(1, 3, 2)
            show_image(mask, title="Original Mask")
            plt.subplot(1, 3, 3)
            show_image(pred_mask, title="Predicted Mask")
            plt.tight_layout()
            plt.show()
    else:
        print("No preloaded model")
    return None

def display_mask(pred):
    mask = np.argmax(pred, axis=-1)
    mask *= 127
    plt.axis("off")
    plt.imshow(mask)
    plt.show()

def show_image(image, title=None):
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')

if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(f"physical_devices : {physical_devices}")
    print(tf.__version__)
    print(tf.executing_eagerly())

    if len(physical_devices) > 0:
        (X_train, y_train), (X_val, y_val) = load_dataset("../../data/test/images", file_extension="JPG", num_channels=5, percentage=0.7)
        train_model(X_train, y_train, X_val, y_val, 5, restore=False)
        load_with_trained_model(X_val, y_val, count=10)

