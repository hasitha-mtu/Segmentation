import os
from datetime import datetime
import keras.callbacks_v1
import tensorflow as tf
from keras.callbacks import (Callback,
                             CSVLogger)
import numpy as np
import random
from models.unet_wsl.data import load_dataset
from models.unet_wsl.model import unet_model
from models.common_utils.loss_functions import  recall_m, precision_m, f1_score, masked_dice_loss
from models.common_utils.plot import plot_model_history, plot_prediction
from tensorflow.keras.callbacks import ModelCheckpoint
from models.common_utils.dataset import load_datasets, set_seed
from models.common_utils.config import load_config, ModelConfig

MODEL_FILE_NAME = 'unet_wsl_best_model1.h5'
DATASET_PATH = '../../input/samples/segnet_512/images'

LOG_DIR = "C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet_wsl\logs"
CKPT_DIR = "C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet_wsl\ckpt"
OUTPUT_DIR = "C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet_wsl\output"

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
        f"{CKPT_DIR}/{MODEL_FILE_NAME}",  # or "best_model.keras"
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,  # set to True if you want only weights
        mode='max',
        verbose=1
    )

    cbs = [
        CSVLogger(LOG_DIR+'/unet_wsl_logs.csv', separator=',', append=False),
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

    plot_model_history(history)

def load_saved_model():
    saved_model_path = os.path.join(CKPT_DIR, MODEL_FILE_NAME)
    print(f"Restoring from {saved_model_path}")
    return keras.models.load_model(saved_model_path,
                                   custom_objects={'recall_m': recall_m,
                                                   'precision_m': precision_m,
                                                   'f1_score': f1_score,
                                                   'masked_dice_loss': masked_dice_loss},
                                   compile=True)


def make_or_restore_model(restore, num_channels, size):
    if restore:
        return load_saved_model()
    else:
        print("Creating fresh model")
        return unet_model(size[0], size[1], num_channels)

def load_with_trained_model(X_val, y_val):
    model = load_saved_model()
    for i in range(len(X_val)):
        image = X_val[i]
        actual_mask = y_val[i]
        formated_image = np.expand_dims(image, 0)
        pred_mask = model.predict(formated_image)
        plot_prediction(i, image[:, :, :3], actual_mask.squeeze(), pred_mask.squeeze(), OUTPUT_DIR)

if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(f"physical_devices : {physical_devices}")
    print(tf.__version__)
    print(tf.executing_eagerly())
    config_file = 'config.yaml'
    load_config(config_file)
    set_seed(ModelConfig.SEED)
    # channels = ['RED', 'GREEN', 'BLUE', 'NDWI', 'Canny', 'LBP', 'HSV Saturation', 'HSV Value', 'GradMag',
    #             'Shadow Mask', 'Lightness', 'GreenRed', 'BlueYellow', 'X', 'Y', 'Z']
    channels = ['RED', 'GREEN', 'BLUE']
    channel_count = len(channels)
    image_size = (ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH)
    if len(physical_devices) > 0:
        train_dataset, validation_dataset = load_datasets(config_file)
        train_model(ModelConfig.TRAINING_EPOCHS,
                    ModelConfig.BATCH_SIZE,
                    train_dataset,
                    validation_dataset,
                    channel_count,
                    size = image_size,
                    restore=False)
        load_with_trained_model(validation_dataset)

# if __name__ == "__main__":
#     print(tf.config.list_physical_devices('GPU'))
#     physical_devices = tf.config.experimental.list_physical_devices('GPU')
#     print(f"physical_devices : {physical_devices}")
#     print(tf.__version__)
#     print(tf.executing_eagerly())
#
#     _set_seed(42)
#
#     image_size = (512, 512) # actual size is (5280, 3956)
#     epochs = 50
#     batch_size = 4
#     # channels = ['RED', 'GREEN', 'BLUE', 'NDWI', 'Canny', 'LBP', 'HSV Saturation', 'HSV Value', 'GradMag',
#     #             'Shadow Mask', 'Lightness', 'GreenRed', 'BlueYellow', 'X', 'Y', 'Z']
#     channels = ['RED', 'GREEN', 'BLUE']
#     channel_count = len(channels)
#     if len(physical_devices) > 0:
#         (X_train, y_train), (X_val, y_val) = load_dataset(DATASET_PATH,
#                                                           size = image_size,
#                                                           file_extension="jpg",
#                                                           channels=channels,
#                                                           percentage=0.7,
#                                                           image_count=200)
#         train_model(epochs, batch_size, X_train, y_train, X_val, y_val, channel_count,
#                     size = image_size,
#                     restore=False)
#         load_with_trained_model(X_val, y_val)
