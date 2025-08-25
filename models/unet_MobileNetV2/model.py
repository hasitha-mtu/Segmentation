import keras
from tensorflow.keras.layers import (Conv2D, Activation, BatchNormalization,
                                     UpSampling2D, Input, Concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
import os

from models.common_utils.loss_functions import  recall_m, precision_m, f1_score, combined_loss_function
from models.unet_MobileNetV2.loss_function import combined_masked_dice_bce_loss
from models.memory_usage import estimate_model_memory_usage
from models.common_utils.config import load_config, ModelConfig
from models.common_utils.model_utils import get_optimizer, estimate_flops

def UnetMobileNetV2(shape):
    inputs = Input(shape=shape, name='input_image')

    # Pre-trained Encoder
    encoder = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False, alpha=0.35)
    print(f'MobileNetV2 summary: {encoder.summary()}')
    skip_connection_names = ['input_image', 'block_1_expand_relu', 'block_3_expand_relu',
                             'block_6_expand_relu', 'block_7_expand_relu']
    encoder_output = encoder.get_layer('block_14_expand_relu').output
    print(f'encoder_output shape: {encoder_output.shape}')

    filter_sizes = [16, 32, 64, 128, 256]
    x = encoder_output
    for i in range(1, len(filter_sizes)+1, 1):
        print(skip_connection_names[-i])
        x_skip = encoder.get_layer(skip_connection_names[-i]).output
        print(f'x_skip shape: {x_skip.shape}')
        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_skip])

        x = Conv2D(filter_sizes[-i], (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filter_sizes[-i], (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = Conv2D(1, (1, 1),  padding='same')(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs=inputs, outputs=x,  name=ModelConfig.MODEL_NAME)

    print(f"Model summary : {model.summary()}")

    estimate_model_memory_usage(model, batch_size=ModelConfig.BATCH_SIZE)

    keras.utils.plot_model(model, os.path.join(ModelConfig.MODEL_DIR, "Unet-MobileNetV2_model.png"), show_shapes=True)

    estimate_flops(model)

    return model

def unet_mobilenet_v2(width, height, input_channels):
    input_shape = (width, height, input_channels)
    model = UnetMobileNetV2(input_shape)
    model.compile(
        optimizer=get_optimizer(),
        loss=combined_loss_function,
        metrics=['accuracy', f1_score, precision_m, recall_m]
    )
    return model

if __name__=='__main__':
    config_file = 'config.yaml'
    load_config(config_file)
    unet_mobilenet_v2(ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH, ModelConfig.MODEL_INPUT_CHANNELS)

