import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
import keras
import os
from models.common_utils.loss_functions import  recall_m, precision_m, f1_score, combined_loss_function
from models.memory_usage import estimate_model_memory_usage
from models.common_utils.config import load_config, ModelConfig
from models.common_utils.model_utils import get_optimizer, estimate_flops

def unet_with_resnet50(input_shape=(512, 512, 16), num_classes=1):
    # Step 1: Define 16-channel input
    inputs = tf.keras.Input(shape=input_shape)

    # Step 2: Project 16 channels to 3 channels for ResNet compatibility
    x_proj = layers.Conv2D(3, kernel_size=1, padding='same', name='project_to_rgb')(inputs)

    # Step 3: Build ResNet50 with input=x_proj but do NOT use input_tensor=
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(512, 512, 3))

    # Get skip connection outputs
    skip1 = base_model.get_layer('conv1_relu').output
    skip2 = base_model.get_layer('conv2_block3_out').output
    skip3 = base_model.get_layer('conv3_block4_out').output
    skip4 = base_model.get_layer('conv4_block6_out').output
    bottleneck = base_model.get_layer('conv5_block3_out').output

    # Create a model from the x_proj input to the bottleneck output
    encoder = models.Model(inputs=base_model.input, outputs=[skip1, skip2, skip3, skip4, bottleneck])

    # Step 4: Get encoder outputs
    s1, s2, s3, s4, b = encoder(x_proj)

    # Decoder
    x = layers.UpSampling2D()(b)
    x = layers.Concatenate()([x, s4])
    x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)

    x = layers.UpSampling2D()(x)
    x = layers.Concatenate()([x, s3])
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)

    x = layers.UpSampling2D()(x)
    x = layers.Concatenate()([x, s2])
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)

    x = layers.UpSampling2D()(x)
    x = layers.Concatenate()([x, s1])
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)

    x = layers.UpSampling2D()(x)

    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[inputs],
                           outputs=[outputs],
                           name=ModelConfig.MODEL_NAME)

    model.compile(
        optimizer=get_optimizer(),
        loss=combined_loss_function,
        metrics=['accuracy', f1_score, precision_m, recall_m]  # Metrics only for the segmentation output
    )

    print(f"Model summary : {model.summary()}")

    estimate_model_memory_usage(model, batch_size=ModelConfig.BATCH_SIZE)

    keras.utils.plot_model(model, os.path.join(ModelConfig.MODEL_DIR, "UNET-ResNet50_model.png"), show_shapes=True)

    estimate_flops(model)

    return model

def unet_model(width, height, num_channels):
    return unet_with_resnet50(input_shape=(width, height, num_channels))

if __name__ == '__main__':
    config_file = 'config.yaml'
    load_config(config_file)
    unet_with_resnet50(input_shape=(ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH, ModelConfig.MODEL_INPUT_CHANNELS))