import os
import keras.callbacks_v1
import tensorflow as tf
from models.unet_ResNet50.model import unet_model
from models.common_utils.loss_functions import  recall_m, precision_m, f1_score, unet_resnet50_loss_function

from models.common_utils.dataset import load_datasets, set_seed
from models.common_utils.config import load_config, ModelConfig
from models.train_model_utils import train_model, load_with_trained_model


def load_saved_model():
    saved_model_path = os.path.join(ModelConfig.MODEL_SAVE_DIR, ModelConfig.SAVED_FILE_NAME)
    return keras.models.load_model(saved_model_path,
                                    custom_objects={'recall_m': recall_m,
                                                    'precision_m': precision_m,
                                                    'f1_score': f1_score,
                                                    'unet_resnet50_loss_function': unet_resnet50_loss_function})

def make_or_restore_model(restore, num_channels, size):
    (width, height) = size
    if restore:
        return load_saved_model()
    else:
        print("Creating fresh model")
        return unet_model(width, height, num_channels)

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

