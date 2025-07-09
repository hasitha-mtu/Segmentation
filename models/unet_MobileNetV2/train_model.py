import tensorflow as tf
import os
import keras
from models.unet_MobileNetV2.model import unet_mobilenet_v2
from models.unet_MobileNetV2.loss_function import combined_masked_dice_bce_loss
from models.common_utils.loss_functions import  recall_m, precision_m, f1_score

from models.common_utils.dataset import load_datasets, set_seed
from models.common_utils.config import load_config, ModelConfig
from models.train_model_utils import train_model, load_with_trained_model


def load_saved_model():
    custom_objects = {'recall_m': recall_m,
                      'precision_m': precision_m,
                      'f1_score': f1_score,
                      'combined_masked_dice_bce_loss': combined_masked_dice_bce_loss}
    saved_model_path = os.path.join(ModelConfig.MODEL_SAVE_DIR, ModelConfig.SAVED_FILE_NAME)
    print(f"Restoring from {saved_model_path}")
    return keras.models.load_model(saved_model_path,
                                   custom_objects=custom_objects,
                                   compile=True)


def make_or_restore_model(restore, num_channels, size):
    (width, height) = size
    if restore:
        return load_saved_model()
    else:
        print("Creating fresh model")
        return unet_mobilenet_v2(width, height, num_channels)


if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(f"physical_devices : {physical_devices}")
    print(tf.__version__)
    print(tf.executing_eagerly())
    config_file = 'config.yaml'
    load_config(config_file)
    set_seed(ModelConfig.SEED, ModelConfig.ENABLE_OP_DETERMINISM)

    channels = ModelConfig.CHANNELS
    channel_count = len(channels)
    if len(physical_devices) > 0:
        info0 = tf.config.experimental.get_memory_info('GPU:0')
        print(f"GPU0 Current: {info0['current']} bytes, Peak: {info0['peak']} bytes")
        info1 = tf.config.experimental.get_memory_info('GPU:1')
        print(f"GPU1 Current: {info1['current']} bytes, Peak: {info1['peak']} bytes")
        train_dataset, validation_dataset = load_datasets(config_file, True)
        print(f'train_dataset: {train_dataset}')
        print(f'validation_dataset: {validation_dataset}')
        train_model(ModelConfig.TRAINING_EPOCHS,
                    ModelConfig.BATCH_SIZE,
                    train_dataset,
                    validation_dataset,
                    channel_count,
                    make_or_restore_model,
                    size = (ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH),
                    restore=False)
        load_with_trained_model(load_saved_model, validation_dataset, 4)

