import os
from datetime import datetime
import keras.callbacks_v1
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import (Callback,
                             CSVLogger)
import numpy as np
import random
from models.unet_plus_plus.data import load_dataset
from models.unet_plus_plus.model import unet_plus_plus
from models.common_utils.loss_functions import  recall_m, precision_m, f1_score
from models.unet_plus_plus.loss_functions import BCEDiceLoss
from models.common_utils.images import show_image
from tensorflow.keras.callbacks import ModelCheckpoint

LOG_DIR = "C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet_plus_plus\logs"
CKPT_DIR = "C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet_plus_plus\ckpt"
OUTPUT_DIR = "C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet_plus_plus\output"

def train_model(epoch_count, batch_size, X_train, y_train, X_val, y_val, num_channels,
                size = (256, 256),
                restore=True):
    print(f'X_train shape : {X_train.shape}')
    print(f'y_train shape : {y_train.shape}')

    os.makedirs(LOG_DIR, exist_ok=True)
    tensorboard = keras.callbacks.TensorBoard(
        log_dir = os.path.join(LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S")),
        histogram_freq=1
    )

    os.makedirs(CKPT_DIR, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        f"{CKPT_DIR}/unet_plus_plus_best_model.h5",  # or "best_model.keras"
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,  # set to True if you want only weights
        mode='max',
        verbose=1
    )

    cbs = [
        CSVLogger(LOG_DIR+'/unet_plus_plus_logs.csv', separator=',', append=False),
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

def load_saved_model():
    saved_model_path = os.path.join(CKPT_DIR, "unet_plus_plus_best_model.h5")
    print(f"Restoring from {saved_model_path}")
    return keras.models.load_model(saved_model_path,
                                   custom_objects={'recall_m': recall_m,
                                                   'precision_m': precision_m,
                                                   'f1_score': f1_score,
                                                   'BCEDiceLoss': BCEDiceLoss},
                                   compile=True)

def make_or_restore_model(restore, num_channels, size):
    (width, height) = size
    if restore:
        return load_saved_model()
    else:
        print("Creating fresh model")
        return unet_plus_plus(width, height, num_channels)

def load_with_trained_model(X_val, y_val):
    saved_model_path = os.path.join(CKPT_DIR, "unet_plus_plus_best_model.h5")
    print(f"Restoring from {saved_model_path}")
    model = keras.models.load_model(saved_model_path,
                                    custom_objects={'recall_m': recall_m,
                                                    'precision_m': precision_m,
                                                    'f1_score': f1_score,
                                                    'BCEDiceLoss': BCEDiceLoss},
                                    compile=True)
    for i in range(len(X_val)):
        image = X_val[i]
        actual_mask = y_val[i]
        formated_image = np.expand_dims(image, 0)
        pred_mask = model.predict(formated_image)
        plt.figure(figsize=(10, 8))
        plt.subplot(1, 3, 1)
        rgb_image = image[:, :, :3]
        show_image(OUTPUT_DIR, rgb_image, index=i, title="Original_Image", save=True)
        plt.subplot(1, 3, 2)
        show_image(OUTPUT_DIR, actual_mask.squeeze(), index=i, title="Actual_Mask", save=True)
        plt.subplot(1, 3, 3)
        show_image(OUTPUT_DIR, pred_mask.squeeze(), index=i, title="Predicted_Mask", save=True)

        plt.tight_layout()
        plt.show()


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

    SEED = 42
    os.environ['PYTHONHASHSEED'] = str(SEED)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # # Optional: For full reproducibility (if supported by your TF version)
    # tf.config.experimental.enable_op_determinism()

    image_size = (512, 512) # actual size is (5280, 3956)
    epochs = 25
    batch_size = 4
    channels = ['RED', 'GREEN', 'BLUE']
    channel_count = len(channels)
    if len(physical_devices) > 0:
        (X_train, y_train), (X_val, y_val) = load_dataset("../../input/samples/segnet_512/images",
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

        train_model(epochs, batch_size, X_train, y_train, X_val, y_val, channel_count,
                    size = image_size,
                    restore=False)
        load_with_trained_model(X_val, y_val)

