import os
from datetime import datetime

import keras.callbacks_v1
import tensorflow as tf
from keras.callbacks import (Callback,
                             CSVLogger)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

from models.common_utils.plot import plot_model_history, plot_prediction

from models.common_utils.dataset import set_seed, load_datasets
from models.common_utils.config import load_config, ModelConfig
from models.common_utils.model_utils import get_model_save_file_name

from models.common_utils.loss_functions import  recall_m, precision_m, f1_score, combined_loss_function

def train_model(epoch_count, batch_size, train_dataset, validation_dataset, num_channels,
                make_or_restore_model,
                config_file,
                size = (256, 256),
                restore=True):

    os.makedirs(ModelConfig.LOG_DIR, exist_ok=True)

    tensorboard = keras.callbacks.TensorBoard(
        log_dir = os.path.join(ModelConfig.LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S")),
        histogram_freq=1
    )

    os.makedirs(ModelConfig.MODEL_SAVE_DIR, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        get_model_save_file_name(None, None),  # or "best_model.keras"
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

    class NaNChecker(tf.keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            if tf.math.is_nan(logs["loss"]):
                if tf.math.is_nan(logs["loss"]):
                    print(f"NaN detected at batch {batch}, stopping training...")
                    self.model.stop_training = True

    class NaNInspector(tf.keras.callbacks.Callback):
        def __init__(self, train_data, val_data=None, max_samples=2):
            super().__init__()
            self.train_data = train_data
            self.val_data = val_data
            self.max_samples = max_samples

        def on_batch_end(self, batch, logs=None):
            if logs is not None and "loss" in logs:
                loss_val = logs["loss"]
                if tf.math.is_nan(loss_val):
                    print(f"❌ NaN detected at batch {batch}, stopping training...")

                    # Extract a problematic batch
                    try:
                        inputs, masks = next(iter(self.train_data))
                    except Exception:
                        print("⚠️ Could not fetch batch for inspection.")
                        self.model.stop_training = True
                        return

                    print(f"Input shape: {inputs.shape}, Mask shape: {masks.shape}")
                    print(f"Unique mask values: {np.unique(masks)}")
                    print(f"Mask dtype: {masks.dtype}")

                    # Show a few samples
                    for i in range(min(self.max_samples, inputs.shape[0])):
                        plt.figure(figsize=(8, 4))

                        plt.subplot(1, 2, 1)
                        plt.title("Input RGB")
                        plt.imshow(inputs[i].numpy().astype(np.uint8))  # assuming 0–255 RGB

                        plt.subplot(1, 2, 2)
                        plt.title("Mask")
                        plt.imshow(masks[i].numpy().squeeze(), cmap="gray")

                        plt.show()

                    self.model.stop_training = True

    class DebugValCallback(tf.keras.callbacks.Callback):
        def __init__(self, val_dataset, loss_fn):
            super().__init__()
            self.val_dataset = val_dataset
            self.loss_fn = loss_fn

        def on_epoch_end(self, epoch, logs=None):
            for step, (x, y) in enumerate(self.val_dataset):
                preds = self.model(x, training=False)
                # Debug numerics here
                tf.debugging.check_numerics(preds, f"Preds contain NaN/Inf at val batch {step}")
                tf.debugging.check_numerics(y, f"Labels contain NaN/Inf at val batch {step}")

                loss = self.loss_fn(y, preds)
                if tf.reduce_any(tf.math.is_nan(loss)):
                    print(f"⚠️ NaN loss at epoch {epoch}, val batch {step}")
                    break

    cbs = [
        CSVLogger(ModelConfig.LOG_DIR + '/model_logs.csv', separator=',', append=False),
        DebugValCallback(validation_dataset, loss_fn=combined_loss_function),
        NaNChecker(),
        NaNInspector(train_dataset),
        checkpoint_cb,
        early_stopping_cb,
        tensorboard
    ]

    if ModelConfig.ENABLE_CLRS == True:
        reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        cbs.append(reduce_lr_on_plateau)


    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = make_or_restore_model(restore, num_channels, size, config_file)

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

def load_with_trained_model(load_saved_model, dataset, config_file, num_display=2):
    model = load_saved_model(config_file)
    for images, true_masks in dataset.take(1):  # Take one batch
        predicted_masks = model.predict(images)
        for i in range(min(num_display, ModelConfig.BATCH_SIZE)):
            actual_image = images[i] # EagerTensor
            actual_image = tf.gather(actual_image, tf.constant([0, 1, 2], dtype=tf.int32), axis=-1) # Use only RGB channels
            print(f'load_with_trained_model|actual_image shape:{actual_image.shape}')
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

        for val_images, val_masks in validation_dataset.take(1):
            print("Val batch shapes:", val_images.shape, val_masks.shape)
            print("Unique values in val masks:", np.unique(val_masks))
            print("Any NaNs in val masks?", np.isnan(val_masks.numpy()).any())

        for imgs, masks in validation_dataset:
            if tf.reduce_sum(masks) == 0:
                print("⚠️ Empty mask in validation set")
        # train_dataset = filter_dataset(train_dataset)
        # validation_dataset = filter_dataset(validation_dataset)
        #
        # print(f'filtered train_dataset: {train_dataset}')
        # print(f'filtered validation_dataset: {validation_dataset}')

        train_model(ModelConfig.TRAINING_EPOCHS,
                    ModelConfig.BATCH_SIZE,
                    train_dataset,
                    validation_dataset,
                    channel_count,
                    make_or_restore_model,
                    config_file,
                    size=(ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH),
                    restore=False)
        load_with_trained_model(load_saved_model, validation_dataset, config_file,  4)

