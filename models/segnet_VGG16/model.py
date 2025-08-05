from models.common_utils.loss_functions import  recall_m, precision_m, f1_score, combined_loss_function
from models.segnet_VGG16.loss_function import combined_masked_dice_bce_loss
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
import keras
import os
from models.memory_usage import estimate_model_memory_usage
from models.common_utils.config import load_config, ModelConfig
from models.common_utils.model_utils import get_optimizer

def SegNetVGG16(input_shape):
    inputs = Input(shape=input_shape)
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=inputs)

    """ Encoder """
    x = vgg16.get_layer('block5_conv3').output  # 32x32

    # Decoder
    x = UpSampling2D(size=(2, 2))(x)  # 32x32 → 64x64
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(size=(2, 2))(x)  # 64x64 → 128x128
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(size=(2, 2))(x)  # 128x128 → 256x256
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(size=(2, 2))(x)  # 128x128 → 256x256
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Final layer
    x = Conv2D(ModelConfig.MODEL_OUTPUT_CHANNELS, (1, 1), padding='valid')(x)
    outputs = Activation('sigmoid')(x)  # Use 'sigmoid' if binary segmentation

    model = Model(inputs=inputs,
                  outputs=outputs,
                  name=ModelConfig.MODEL_NAME)
    print("Model output shape:", model.output_shape)
    model.summary()

    estimate_model_memory_usage(model, batch_size=ModelConfig.BATCH_SIZE)

    keras.utils.plot_model(model, os.path.join(ModelConfig.MODEL_DIR, "SegNet-VGG16_model.png"), show_shapes=True)

    return model

def segnet_vgg16(width, height, input_channels):
    input_shape = (width, height, input_channels)
    model = SegNetVGG16(input_shape)
    model.compile(
        optimizer=get_optimizer(),
        loss=combined_loss_function,
        metrics=['accuracy', f1_score, precision_m, recall_m]
    )
    return model

if __name__=='__main__':
    config_file = 'config.yaml'
    load_config(config_file)
    segnet_vgg16(ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH, ModelConfig.MODEL_INPUT_CHANNELS)