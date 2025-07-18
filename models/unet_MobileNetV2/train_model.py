import os
import keras
from models.unet_MobileNetV2.model import unet_mobilenet_v2
from models.unet_MobileNetV2.loss_function import combined_masked_dice_bce_loss
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
        return unet_mobilenet_v2(width, height, num_channels)


def model_execution(config_file):
    execute_model(config_file, make_or_restore_model, load_saved_model)


if __name__ == "__main__":
    config_file = 'config.yaml'
    execute_model(config_file, make_or_restore_model, load_saved_model)


