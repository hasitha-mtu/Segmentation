import os
import keras
from models.segnet_VGG16.model import segnet_vgg16
from models.segnet_VGG16.loss_function import combined_masked_dice_bce_loss
from models.common_utils.loss_functions import  recall_m, precision_m, f1_score

from models.common_utils.config import ModelConfig, load_config
from models.train_model_utils import execute_model

def load_saved_model(config_file):
    load_config(config_file)
    saved_model_path = os.path.join(ModelConfig.MODEL_SAVE_DIR, ModelConfig.SAVED_FILE_NAME)
    if os.path.exists(saved_model_path):
        return loading_model(saved_model_path)
    else:
        print(f"Saved model file '{saved_model_path}' does not exist.")
        model_execution(config_file)
        return loading_model(saved_model_path)

def loading_model(saved_model_path):
    print(f"Restoring from {saved_model_path}")
    custom_objects = {'recall_m': recall_m,
                      'precision_m': precision_m,
                      'f1_score': f1_score,
                      'combined_masked_dice_bce_loss': combined_masked_dice_bce_loss}
    return keras.models.load_model(saved_model_path,
                                   custom_objects=custom_objects,
                                   compile=True)


def make_or_restore_model(restore, num_channels, size, config_file):
    (width, height) = size
    if restore:
        return load_saved_model(config_file)
    else:
        print("Creating fresh model")
        return segnet_vgg16(width, height, num_channels)


def model_execution(config_file):
    execute_model(config_file, make_or_restore_model, load_saved_model)


if __name__ == "__main__":
    config_file = 'config.yaml'
    execute_model(config_file, make_or_restore_model, load_saved_model)


# def train_model(epoch_count, batch_size, X_train, y_train, X_val, y_val, width, height,
#                 input_channels, restore=True):
#     print(f'X_train shape : {X_train.shape}')
#     print(f'y_train shape : {y_train.shape}')
#
#     os.makedirs(LOG_DIR, exist_ok=True)
#     tensorboard = keras.callbacks.TensorBoard(
#         log_dir = os.path.join(LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S")),
#         histogram_freq=1
#     )
#
#     os.makedirs(CKPT_DIR, exist_ok=True)
#
#     model_checkpoint_callback = ModelCheckpoint(
#         f"{CKPT_DIR}/{MODEL_FILE_NAME}",  # or "best_model.keras"
#         monitor='val_accuracy',
#         save_best_only=True,
#         save_weights_only=False,  # set to True if you want only weights
#         mode='max',
#         verbose=1
#     )
#
#     cbs = [
#         CSVLogger(LOG_DIR+'/segnet_VGG16_logs.csv', separator=',', append=False),
#         model_checkpoint_callback,
#         tensorboard
#     ]
#     # Create a MirroredStrategy.
#     strategy = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())
#     print("Number of devices: {}".format(strategy.num_replicas_in_sync))
#
#     with strategy.scope():
#         model = make_or_restore_model(restore, width, height, input_channels)
#
#     history = model.fit(
#                     X_train,
#                     y_train,
#                     epochs=epoch_count,
#                     batch_size=batch_size,
#                     validation_data=(X_val, y_val),
#                     callbacks=cbs
#                 )
#
#     print(history.history)
#     accuracy = history.history["accuracy"]
#     val_accuracy = history.history["val_accuracy"]
#     loss = history.history["loss"]
#     val_loss = history.history["val_loss"]
#     epochs = range(1, len(accuracy) + 1)
#     plt.plot(epochs, accuracy, "bo", label="Training accuracy")
#     plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
#     plt.title("Training and validation accuracy")
#     plt.legend()
#     plt.figure()
#     plt.plot(epochs, loss, "bo", label="Training loss")
#     plt.plot(epochs, val_loss, "b", label="Validation loss")
#     plt.title("Training and validation loss")
#     plt.legend()
#     plt.show()
#     return None
#
# def load_saved_model():
#     custom_objects = {'recall_m': recall_m,
#                       'precision_m': precision_m,
#                       'f1_score': f1_score,
#                       'combined_masked_dice_bce_loss': combined_masked_dice_bce_loss}
#     saved_model_path = f"{CKPT_DIR}/segnet_VGG16_model.h5"
#     print(f"Restoring from {saved_model_path}")
#     return keras.models.load_model(saved_model_path,
#                                    custom_objects=custom_objects,
#                                    compile=True)
#
# def make_or_restore_model(restore, width, height, input_channels):
#     if restore:
#         return load_saved_model()
#     else:
#         print("Creating fresh model")
#         return segnet_vgg16(width, height, input_channels)
#
# def load_with_trained_model(X_val, y_val):
#     loaded_model = load_saved_model()
#     for i in range(len(X_val)):
#         image_array = X_val[i]
#         actual_mask = y_val[i]
#
#         input_for_model = np.expand_dims(image_array, axis=0)
#
#         pred_mask = loaded_model.predict(input_for_model)
#         plt.figure(figsize=(10, 8))
#         plt.subplot(1, 3, 1)
#         rgb_image = image_array[:, :, :3]
#         show_image(OUTPUT_DIR, rgb_image, index=i, title="Original_Image", save=True)
#         plt.subplot(1, 3, 2)
#         show_image(OUTPUT_DIR, actual_mask.squeeze(), index=i, title="Actual_Mask", save=True)
#         plt.subplot(1, 3, 3)
#         show_image(OUTPUT_DIR, pred_mask.squeeze(), index=i, title="Predicted_Mask", save=True)
#
#         plt.tight_layout()
#         plt.show()
#
# if __name__ == "__main__":
#     print(tf.config.list_physical_devices('GPU'))
#     physical_devices = tf.config.experimental.list_physical_devices('GPU')
#     print(f"physical_devices : {physical_devices}")
#     print(tf.__version__)
#     print(tf.executing_eagerly())
#
#     SEED = 42
#     os.environ['PYTHONHASHSEED'] = str(SEED)
#     tf.random.set_seed(SEED)
#     np.random.seed(SEED)
#     random.seed(SEED)
#     # # Optional: For full reproducibility (if supported by your TF version)
#     # tf.config.experimental.enable_op_determinism()
#
#     image_size = (512, 512) # actual size is (5280, 3956)
#     epochs = 50
#     batch_size = 4
#     channels = ['RED', 'GREEN', 'BLUE']
#     channel_count = len(channels)
#     if len(physical_devices) > 0:
#         (X_train, y_train), (X_val, y_val) = load_dataset(DATASET_PATH,
#                                                           size = image_size,
#                                                           file_extension="jpg",
#                                                           channels=channels,
#                                                           percentage=0.7)
#
#         y_train = (y_train > 0.0).astype(np.float32)
#         y_val = (y_val > 0.0).astype(np.float32)
#         print(f"y_train unique values : {np.unique(y_train)}")
#         print(f"y_val unique values : {np.unique(y_val)}")
#         water_pixels = np.sum(y_val == 1.0)
#         non_water_pixels = np.sum(y_val == 0.0)
#         print(f"Water: {water_pixels}, Non-water: {non_water_pixels}")
#
#         train_model(epochs, batch_size, X_train, y_train, X_val, y_val,
#                     512, 512, 3, restore=False)
#
#         load_with_trained_model(X_val, y_val)
