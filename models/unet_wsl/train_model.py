import os
import keras.callbacks_v1
from models.unet_wsl.model import unet_model
from models.common_utils.loss_functions import recall_m, precision_m, f1_score, masked_dice_loss

from models.common_utils.config import ModelConfig
from models.train_model_utils import execute_model


def load_saved_model(saved_model_path = None):
    if saved_model_path is None:
        saved_model_path = os.path.join(ModelConfig.MODEL_SAVE_DIR, ModelConfig.SAVED_FILE_NAME)
    if os.path.exists(saved_model_path):
        return loading_model(saved_model_path)
    else:
        print(f"Saved model file '{saved_model_path}' does not exist.")
        config_file = os.path.join(ModelConfig.MODEL_DIR, 'config.yaml')
        model_execution(config_file)
        return loading_model(saved_model_path)

def loading_model(saved_model_path):
    print(f"Restoring from {saved_model_path}")
    custom_objects = {'recall_m': recall_m,
                      'precision_m': precision_m,
                      'f1_score': f1_score,
                      'masked_dice_loss': masked_dice_loss}
    return keras.models.load_model(saved_model_path,
                                   custom_objects=custom_objects,
                                   compile=True)

def make_or_restore_model(restore, num_channels, size):
    if restore:
        return load_saved_model()
    else:
        print("Creating fresh model")
        return unet_model(size[0], size[1], num_channels)


def model_execution(config_file):
    execute_model(config_file, make_or_restore_model, load_saved_model)


if __name__ == "__main__":
    config_file = 'config.yaml'
    execute_model(config_file, make_or_restore_model, load_saved_model)

# def train_model(epoch_count, batch_size, train_dataset, validation_dataset, num_channels,
#                 size = (256, 256),
#                 restore=True):
#
#     os.makedirs(ModelConfig.LOG_DIR, exist_ok=True)
#     tensorboard = keras.callbacks.TensorBoard(
#         log_dir = os.path.join(ModelConfig.LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S")),
#         histogram_freq=1
#     )
#
#     os.makedirs(ModelConfig.MODEL_SAVE_DIR, exist_ok=True)
#
#     checkpoint_cb = ModelCheckpoint(
#         f"{ModelConfig.MODEL_SAVE_DIR}/{ModelConfig.SAVED_FILE_NAME}",  # or "best_model.keras"
#         monitor=ModelConfig.CHECKPOINT_CALLBACK_MONITOR,
#         save_best_only=ModelConfig.CHECKPOINT_CALLBACK_SAVE_BEST_ONLY,
#         save_weights_only=ModelConfig.CHECKPOINT_CALLBACK_SAVE_WEIGHTS_ONLY,  # set to True if you want only weights
#         mode=ModelConfig.CHECKPOINT_CALLBACK_MODE,
#         verbose=1
#     )
#
#     # --- Define Early Stopping (Optional) ---
#     # Stop training if val_loss doesn't improve for 5 consecutive epochs
#     early_stopping_cb = EarlyStopping(
#         monitor=ModelConfig.EARLY_STOPPING_CALLBACK_MONITOR,
#         patience=ModelConfig.EARLY_STOPPING_CALLBACK_PATIENCE,  # Number of epochs with no improvement after which training will be stopped
#         mode=ModelConfig.EARLY_STOPPING_CALLBACK_MODE,
#         restore_best_weights=ModelConfig.EARLY_STOPPING_CALLBACK_RESTORE_BEST_WEIGHTS,  # Restores model weights from the epoch with the best value of the monitored metric.
#         verbose=1
#     )
#
#     cbs = [
#         CSVLogger(ModelConfig.LOG_DIR+'/model_logs.csv', separator=',', append=False),
#         checkpoint_cb,
#         early_stopping_cb,
#         tensorboard
#     ]
#     # Create a MirroredStrategy.
#     strategy = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())
#     print("Number of devices: {}".format(strategy.num_replicas_in_sync))
#
#     with strategy.scope():
#         model = make_or_restore_model(restore, num_channels, size)
#
#     history = model.fit(
#                     train_dataset,
#                     epochs=epoch_count,
#                     batch_size=batch_size,
#                     validation_data=validation_dataset,
#                     callbacks=cbs,
#                     verbose=1
#                 )
#
#     plot_model_history(history, ModelConfig.OUTPUT_DIR)
#
# def load_saved_model():
#     saved_model_path = os.path.join(ModelConfig.MODEL_SAVE_DIR, ModelConfig.SAVED_FILE_NAME)
#     print(f"Restoring from {saved_model_path}")
#     return keras.models.load_model(saved_model_path,
#                                    custom_objects={'recall_m': recall_m,
#                                                    'precision_m': precision_m,
#                                                    'f1_score': f1_score,
#                                                    'masked_dice_loss': masked_dice_loss},
#                                    compile=True)
#
#
# def make_or_restore_model(restore, num_channels, size):
#     if restore:
#         return load_saved_model()
#     else:
#         print("Creating fresh model")
#         return unet_model(size[0], size[1], num_channels)
#
# def load_with_trained_model(dataset, num_display=2):
#     model = load_saved_model()
#     for images, true_masks in dataset.take(1):  # Take one batch
#         predicted_masks = model.predict(images)
#         for i in range(min(num_display, ModelConfig.BATCH_SIZE)):
#             actual_image = images[i] # EagerTensor
#             actual_mask = true_masks[i] # EagerTensor
#             predicted_mask = predicted_masks[i] # nd array
#             plot_prediction(i, actual_image.numpy(),
#                             actual_mask.numpy(),
#                             predicted_mask,
#                             ModelConfig.OUTPUT_DIR)
#
# if __name__ == "__main__":
#     print(tf.config.list_physical_devices('GPU'))
#     physical_devices = tf.config.experimental.list_physical_devices('GPU')
#     print(f"physical_devices : {physical_devices}")
#     print(tf.__version__)
#     print(tf.executing_eagerly())
#     config_file = 'config.yaml'
#     load_config(config_file)
#     set_seed(ModelConfig.SEED)
#
#     channels = ['RED', 'GREEN', 'BLUE']
#     channel_count = len(channels)
#     image_size = (ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH)
#     if len(physical_devices) > 0:
#         train_dataset, validation_dataset = load_datasets(config_file, True)
#         print(f'train_dataset: {train_dataset}')
#         print(f'validation_dataset: {validation_dataset}')
#         train_model(ModelConfig.TRAINING_EPOCHS,
#                     ModelConfig.BATCH_SIZE,
#                     train_dataset,
#                     validation_dataset,
#                     channel_count,
#                     size = image_size,
#                     restore=False)
#         load_with_trained_model(validation_dataset, 4)
