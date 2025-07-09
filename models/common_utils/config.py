import yaml
import os

# Define a class or module to hold your "constants"
class ModelConfig:
    pass

def load_config(config_file):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    # Assign values from config_data to our Config class
    # Accessing nested values
    ModelConfig.MODEL_NAME = config_data.get('model', {}).get('name', 'Unet')
    ModelConfig.MODEL_INPUT_CHANNELS = config_data.get('model', {}).get('input_channels', 3)
    ModelConfig.MODEL_OUTPUT_CHANNELS = config_data.get('model', {}).get('output_channels', 1)

    ModelConfig.DATASET_PATH = config_data.get('data', {}).get('dataset_path', '../../input/updated_samples/segnet_512')

    ModelConfig.IMAGE_HEIGHT = config_data.get('data', {}).get('image_size', {}).get('height', 512)
    ModelConfig.IMAGE_WIDTH = config_data.get('data', {}).get('image_size', {}).get('width', 512)

    ModelConfig.BATCH_SIZE = config_data.get('data', {}).get('batch_size', 4)
    ModelConfig.BUFFER_SIZE = config_data.get('data', {}).get('buffer_size', 100)
    ModelConfig.SEED = config_data.get('data', {}).get('seed', 42)
    ModelConfig.TEST_SIZE = config_data.get('data', {}).get('test_size', 0.2)

    ModelConfig.AUGMENTATION_ROTATE = config_data.get('data', {}).get('augmentation', {}).get('rotate', True)
    ModelConfig.AUGMENTATION_FLIP_HORIZONTAL = config_data.get('data', {}).get('augmentation', {}).get('flip_horizontal', True)
    ModelConfig.AUGMENTATION_ZOOM_RANGE = config_data.get('data', {}).get('augmentation', {}).get('zoom_range', 0.1)

    ModelConfig.TRAINING_EPOCHS = config_data.get('training', {}).get('epochs', 100)
    ModelConfig.TRAINING_LR = config_data.get('training', {}).get('learning_rate', 0.0001)
    ModelConfig.TRAINING_OPTIMIZER = config_data.get('training', {}).get('optimizer', 'Adam')
    ModelConfig.TRAINING_SPLIT = config_data.get('training', {}).get('training_split', 0.2)

    ModelConfig.CHECKPOINT_CALLBACK_MONITOR = config_data.get('training', {}).get('checkpoint_callback', {}).get('monitor', 'val_accuracy')
    ModelConfig.CHECKPOINT_CALLBACK_MODE = config_data.get('training', {}).get('checkpoint_callback', {}).get('mode', 'max')
    ModelConfig.CHECKPOINT_CALLBACK_SAVE_BEST_ONLY = config_data.get('training', {}).get('checkpoint_callback', {}).get('save_best_only', True)
    ModelConfig.CHECKPOINT_CALLBACK_SAVE_WEIGHTS_ONLY = config_data.get('training', {}).get('checkpoint_callback', {}).get('save_weights_only', False)

    ModelConfig.EARLY_STOPPING_CALLBACK_MONITOR = config_data.get('training', {}).get('early_stopping_callback', {}).get('monitor', 'val_loss')
    ModelConfig.EARLY_STOPPING_CALLBACK_MODE = config_data.get('training', {}).get('early_stopping_callback', {}).get('mode', 'min')
    ModelConfig.EARLY_STOPPING_CALLBACK_PATIENCE = config_data.get('training', {}).get('early_stopping_callback', {}).get('patience', 5)
    ModelConfig.EARLY_STOPPING_CALLBACK_RESTORE_BEST_WEIGHTS = config_data.get('training', {}).get('early_stopping_callback', {}).get('restore_best_weights', True)

    ModelConfig.MODEL_SAVE_DIR = config_data.get('paths', {}).get('model_save_dir', './ckpt')
    ModelConfig.LOG_DIR = config_data.get('paths', {}).get('log_dir', './logs')
    ModelConfig.OUTPUT_DIR = config_data.get('paths', {}).get('output_dir', './output')
    ModelConfig.SAVED_FILE_NAME = config_data.get('paths', {}).get('saved_file_name', 'model.h5')

    print(f"Loaded config from {config_file}")

# --- Usage ---
if __name__ == "__main__":
    try:
        load_config('../unet_wsl/config.yaml')

        print(f"MODEL_NAME (YAML): {ModelConfig.MODEL_NAME}")
        print(f"DATASET_PATH (YAML): {ModelConfig.DATASET_PATH}")
        print(f"IMAGE_HEIGHT (YAML): {ModelConfig.IMAGE_HEIGHT}")
        print(f"BATCH_SIZE (YAML): {ModelConfig.BATCH_SIZE}")
        print(f"AUGMENTATION_ROTATE (YAML): {ModelConfig.AUGMENTATION_ROTATE}")
        print(f"TRAINING_EPOCHS (YAML): {ModelConfig.TRAINING_EPOCHS}")
        print(f"MODEL_SAVE_DIR (YAML): {ModelConfig.MODEL_SAVE_DIR}")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")


# if __name__ == "__main__":
#     config_file = '../unet_wsl/config.yaml'
#     cfg = load_config(config_file)
#
#     print(f"Loaded configuration from {config_file}:")
#     print(cfg)
#     print("-" * 60)
#
#     # --- 2. Access and use hyperparameters ---
#
#     # Model Parameters
#     model_name = cfg['model']['name']
#     input_channels = cfg['model']['input_channels']
#     output_channels = cfg['model']['output_channels']
#
#     print(f"Model Name: {model_name}")
#
#     # Data Parameters
#     dataset_path = cfg['data']['dataset_path']
#     image_size = tuple(cfg['data']['image_size']) # Convert list to tuple if needed
#     batch_size = cfg['data']['batch_size']
#     do_rotate = cfg['data']['augmentation']['rotate']
#
#     print(f"Dataset Path: {dataset_path}")
#     print(f"Image Size: {image_size}")
#     print(f"Apply Rotation: {do_rotate}")
#
#     # Training Parameters
#     epochs = cfg['training']['epochs']
#     learning_rate = cfg['training']['learning_rate']
#     optimizer_type = cfg['training']['optimizer']
#
#     print(f"Epochs: {epochs}")
#     print(f"Learning Rate: {learning_rate}")
#     print(f"Optimizer: {optimizer_type}")
#
#     # Path Parameters
#     model_save_dir = cfg['paths']['model_save_dir']
#     log_dir = cfg['paths']['log_dir']
#
#     # Ensure directories exist
#     print(f"Model will be saved to: {model_save_dir}")
#     print(f"Logs will be written to: {log_dir}")
