import tensorflow as tf

from models.unet2.model import unet_model
from preprocessing import load_drone_dataset
import os
import keras
from datetime import datetime
from keras.callbacks import (Callback,
                             CSVLogger)
import matplotlib.pyplot as plt

LOG_DIR = "C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet2\logs"


def train_model(X_train, y_train, X_val, y_val):
    print(f'X_train shape : {X_train.shape}')
    print(f'y_train shape : {y_train.shape}')

    os.makedirs(LOG_DIR, exist_ok=True)
    tensorboard = keras.callbacks.TensorBoard(
        log_dir = os.path.join(LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S")),
        histogram_freq=1
    )

    model = unet_model(output_channels=3)

    cbs = [
        CSVLogger(LOG_DIR + '/unet_logs.csv', separator=',', append=False),
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

if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(f"physical_devices : {physical_devices}")
    if len(physical_devices) > 0:
        (X_train, y_train), (X_val, y_val) = load_drone_dataset("../../input/drone/images")
        train_model(X_train, y_train, X_val, y_val)