import tensorflow as tf

from models.unet2.model import unet_model
from preprocessing import load_drone_dataset
import os
import keras
from datetime import datetime
from keras.callbacks import (Callback,
                             ModelCheckpoint,
                             CSVLogger)
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint

LOG_DIR = "C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet2\logs"
CKPT_DIR = "C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet2\ckpt"

def train_model(X_train, y_train, X_val, y_val):
    print(f'X_train shape : {X_train.shape}')
    print(f'y_train shape : {y_train.shape}')

    os.makedirs(LOG_DIR, exist_ok=True)
    tensorboard = keras.callbacks.TensorBoard(
        log_dir = os.path.join(LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S")),
        histogram_freq=1
    )

    os.makedirs(CKPT_DIR, exist_ok=True)
    model = unet_model(output_channels=3)

    cbs = [
        CSVLogger(LOG_DIR + '/unet_logs.csv', separator=',', append=False),
        ModelCheckpoint(CKPT_DIR + '/ckpt-{epoch}', save_freq="epoch"),
        tensorboard
    ]

    history = model.fit(
                    X_train,
                    y_train,
                    epochs=50,
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

def show_image(image, title=None):
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')

def load_with_trained_model(X_val, y_val, count=5):
    checkpoints = [os.path.join(CKPT_DIR, name) for name in os.listdir("ckpt")]
    print(f"Checkpoints: {checkpoints}")
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"Restoring from {latest_checkpoint}")
        model = keras.models.load_model(latest_checkpoint)
        for i in range(count):
            id = randint(len(X_val))
            image = X_val[id]
            mask = y_val[id]
            pred_mask = model.predict(np.expand_dims(image, 0))[0]
            plt.figure(figsize=(10, 8))
            plt.subplot(1, 3, 1)
            show_image(image, title="Original Image")
            plt.subplot(1, 3, 2)
            show_image(mask, title="Original Mask")
            plt.subplot(1, 3, 3)
            show_image(pred_mask, title="Predicted Mask")
            plt.tight_layout()
            plt.show()
    else:
        print("No preloaded model")
    return None

if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(f"physical_devices : {physical_devices}")
    if len(physical_devices) > 0:
        (X_train, y_train), (X_val, y_val) = load_drone_dataset("../../input/drone/images")
        train_model(X_train, y_train, X_val, y_val)
        load_with_trained_model(X_val, y_val, count=10)