import tensorflow_addons as tfa
import tensorflow as tf
import os
from models.common_utils.config import ModelConfig

def get_optimizer():
    initial_learning_rate = 1e-4  # A smaller initial learning rate
    weight_decay = 1e-5  # A small, but effective weight decay
    if ModelConfig.TRAINING_OPTIMIZER == 'AdamW':
        return tfa.optimizers.AdamW(
            weight_decay=weight_decay,
            learning_rate=initial_learning_rate
        )
    else:
        return tf.optimizers.Adam()

def get_model_save_file_name():
    if ModelConfig.ENABLE_CLRS == True:
        saved_model_path = os.path.join(ModelConfig.MODEL_SAVE_DIR,
                                        f'{ModelConfig.SAVED_FILE_NAME}_{ModelConfig.TRAINING_OPTIMIZER}_with_CLRS.h5')
    else:
        saved_model_path = os.path.join(ModelConfig.MODEL_SAVE_DIR,
                                        f'{ModelConfig.SAVED_FILE_NAME}_{ModelConfig.TRAINING_OPTIMIZER}_without_CLRS.h5')

    return saved_model_path

