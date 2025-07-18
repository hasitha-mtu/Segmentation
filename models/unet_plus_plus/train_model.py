import os
import keras.callbacks_v1
from models.unet_plus_plus.model import unet_plus_plus
from models.common_utils.loss_functions import  recall_m, precision_m, f1_score
from models.unet_plus_plus.loss_functions import BCEDiceLoss

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
    return keras.models.load_model(saved_model_path,
                                   custom_objects={'recall_m': recall_m,
                                                   'precision_m': precision_m,
                                                   'f1_score': f1_score,
                                                   'BCEDiceLoss': BCEDiceLoss},
                                   compile=True)

def make_or_restore_model(restore, num_channels, size, config_file):
    (width, height) = size
    if restore:
        return load_saved_model(config_file)
    else:
        print("Creating fresh model")
        return unet_plus_plus(width, height, num_channels)


def model_execution(config_file):
    execute_model(config_file, make_or_restore_model, load_saved_model)


if __name__ == "__main__":
    config_file = 'config.yaml'
    execute_model(config_file, make_or_restore_model, load_saved_model)

# def plot_attention_weights(channels, weight_lists):
#     result = [sum(values) for values in zip(*weight_lists)]
#     print(f"plot_attention_weights|result: {result}")
#     percentages = [(value/sum(result))*100 for value in result]
#     print(f"plot_attention_weights|percentages: {percentages}")
#     plt.bar(range(1, len(channels) + 1), percentages)
#     plt.xticks(range(1, len(channels) + 1), channels)
#     plt.ylabel("Channel Attention Weight")
#     plt.title("Learned Attention per Channel")
#     plt.tight_layout()
#     plt.show()
#
# if __name__ == "__main__":
#     print(tf.config.list_physical_devices('GPU'))
#     physical_devices = tf.config.experimental.list_physical_devices('GPU')
#     print(f"physical_devices : {physical_devices}")
#     print(tf.__version__)
#     print(tf.executing_eagerly())
#     config_file = 'config.yaml'
#     load_config(config_file)
#     set_seed(ModelConfig.SEED, ModelConfig.ENABLE_OP_DETERMINISM)
#
#     channels = ModelConfig.CHANNELS
#     channel_count = len(channels)
#     if len(physical_devices) > 0:
#         info0 = tf.config.experimental.get_memory_info('GPU:0')
#         print(f"GPU0 Current: {info0['current']} bytes, Peak: {info0['peak']} bytes")
#         info1 = tf.config.experimental.get_memory_info('GPU:1')
#         print(f"GPU1 Current: {info1['current']} bytes, Peak: {info1['peak']} bytes")
#         train_dataset, validation_dataset = load_datasets(config_file, True)
#         print(f'train_dataset: {train_dataset}')
#         print(f'validation_dataset: {validation_dataset}')
#         train_model(ModelConfig.TRAINING_EPOCHS,
#                     ModelConfig.BATCH_SIZE,
#                     train_dataset,
#                     validation_dataset,
#                     channel_count,
#                     make_or_restore_model,
#                     size = (ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH),
#                     restore=False)
#         load_with_trained_model(load_saved_model, validation_dataset, 4)



