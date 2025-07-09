import os
from datetime import datetime

import keras.callbacks_v1
import tensorflow as tf
from keras.callbacks import (Callback,
                             CSVLogger)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from models.common_utils.plot import plot_model_history, plot_prediction

from models.common_utils.dataset import load_datasets, set_seed
from models.common_utils.config import load_config, ModelConfig

def train_model(epoch_count, batch_size, train_dataset, validation_dataset, num_channels,
                make_or_restore_model,
                size = (256, 256),
                restore=True):

    os.makedirs(ModelConfig.LOG_DIR, exist_ok=True)

    tensorboard = keras.callbacks.TensorBoard(
        log_dir = os.path.join(ModelConfig.LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S")),
        histogram_freq=1
    )

    os.makedirs(ModelConfig.MODEL_SAVE_DIR, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        f"{ModelConfig.MODEL_SAVE_DIR}/{ModelConfig.SAVED_FILE_NAME}",  # or "best_model.keras"
        monitor=ModelConfig.CHECKPOINT_CALLBACK_MONITOR,
        save_best_only=ModelConfig.CHECKPOINT_CALLBACK_SAVE_BEST_ONLY,
        save_weights_only=ModelConfig.CHECKPOINT_CALLBACK_SAVE_WEIGHTS_ONLY,  # set to True if you want only weights
        mode=ModelConfig.CHECKPOINT_CALLBACK_MODE,
        verbose=1
    )

    # --- Define Early Stopping (Optional) ---
    # Stop training if val_loss doesn't improve for 5 consecutive epochs
    early_stopping_cb = EarlyStopping(
        monitor=ModelConfig.EARLY_STOPPING_CALLBACK_MONITOR,
        patience=ModelConfig.EARLY_STOPPING_CALLBACK_PATIENCE,  # Number of epochs with no improvement after which training will be stopped
        mode=ModelConfig.EARLY_STOPPING_CALLBACK_MODE,
        restore_best_weights=ModelConfig.EARLY_STOPPING_CALLBACK_RESTORE_BEST_WEIGHTS,  # Restores model weights from the epoch with the best value of the monitored metric.
        verbose=1
    )

    cbs = [
        CSVLogger(ModelConfig.LOG_DIR+'/model_logs.csv', separator=',', append=False),
        checkpoint_cb,
        early_stopping_cb,
        tensorboard
    ]
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = make_or_restore_model(restore, num_channels, size)

    history = model.fit(
                    train_dataset,
                    epochs=epoch_count,
                    batch_size=batch_size,
                    validation_data=validation_dataset,
                    callbacks=cbs,
                    verbose=1
                )

    os.makedirs(ModelConfig.OUTPUT_DIR, exist_ok=True)
    plot_model_history(history, ModelConfig.OUTPUT_DIR)

def load_with_trained_model(load_saved_model, dataset, num_display=2):
    model = load_saved_model()
    for images, true_masks in dataset.take(1):  # Take one batch
        predicted_masks = model.predict(images)
        for i in range(min(num_display, ModelConfig.BATCH_SIZE)):
            actual_image = images[i] # EagerTensor
            actual_mask = true_masks[i] # EagerTensor
            predicted_mask = predicted_masks[i] # nd array
            plot_prediction(i, actual_image.numpy(),
                            actual_mask.numpy(),
                            predicted_mask,
                            ModelConfig.OUTPUT_DIR)


def execute_model(config_file, make_or_restore_model, load_saved_model):
    print(tf.config.list_physical_devices('GPU'))
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(f"physical_devices : {physical_devices}")
    print(tf.__version__)
    print(tf.executing_eagerly())
    load_config(config_file)
    set_seed(ModelConfig.SEED, ModelConfig.ENABLE_OP_DETERMINISM)

    channels = ModelConfig.CHANNELS
    channel_count = len(channels)
    if len(physical_devices) > 0:
        train_dataset, validation_dataset = load_datasets(config_file, True)
        print(f'train_dataset: {train_dataset}')
        print(f'validation_dataset: {validation_dataset}')
        train_model(ModelConfig.TRAINING_EPOCHS,
                    ModelConfig.BATCH_SIZE,
                    train_dataset,
                    validation_dataset,
                    channel_count,
                    make_or_restore_model,
                    size=(ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH),
                    restore=False)
        load_with_trained_model(load_saved_model, validation_dataset, 4)

