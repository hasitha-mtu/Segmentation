import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    config_file = '../unet_wsl/config.yaml'
    cfg = load_config(config_file)

    print(f"Loaded configuration from {config_file}:")
    print(cfg)
    print("-" * 60)

    # --- 2. Access and use hyperparameters ---

    # Model Parameters
    model_name = cfg['model']['name']
    input_channels = cfg['model']['input_channels']
    output_channels = cfg['model']['output_channels']

    print(f"Model Name: {model_name}")

    # Data Parameters
    dataset_path = cfg['data']['dataset_path']
    image_size = tuple(cfg['data']['image_size']) # Convert list to tuple if needed
    batch_size = cfg['data']['batch_size']
    do_rotate = cfg['data']['augmentation']['rotate']

    print(f"Dataset Path: {dataset_path}")
    print(f"Image Size: {image_size}")
    print(f"Apply Rotation: {do_rotate}")

    # Training Parameters
    epochs = cfg['training']['epochs']
    learning_rate = cfg['training']['learning_rate']
    optimizer_type = cfg['training']['optimizer']

    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Optimizer: {optimizer_type}")

    # Path Parameters
    model_save_dir = cfg['paths']['model_save_dir']
    log_dir = cfg['paths']['log_dir']

    # Ensure directories exist
    print(f"Model will be saved to: {model_save_dir}")
    print(f"Logs will be written to: {log_dir}")
