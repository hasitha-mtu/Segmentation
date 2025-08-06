
from tensorflow.keras.layers import (Conv2D, BatchNormalization, AveragePooling2D, UpSampling2D, Concatenate)
from tensorflow.keras.models import Model
import keras
import os
import tensorflow as tf

from models.common_utils.loss_functions import  recall_m, precision_m, f1_score, combined_loss_function
from models.deeplabv3_plus.loss_function import combined_masked_dice_bce_loss
from models.memory_usage import estimate_model_memory_usage
from models.common_utils.config import load_config, ModelConfig
from models.common_utils.model_utils import get_optimizer

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    use_bias=False):
    x = Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
        interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3Plus(input_shape):
    model_input = keras.Input(shape=input_shape)
    resnet50 = keras.applications.ResNet50(
        weights=None, include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = UpSampling2D(
        size=(input_shape[0] // 4 // x.shape[1], input_shape[1] // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = UpSampling2D(
        size=(input_shape[0] // x.shape[1], input_shape[1] // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = Conv2D(ModelConfig.MODEL_OUTPUT_CHANNELS, kernel_size=(1, 1), padding="same", activation='sigmoid')(x)
    model = Model(inputs=model_input, outputs=model_output)
    print("Model output shape:", model.output_shape)
    print(f"Model summary : {model.summary()}")

    estimate_model_memory_usage(model, batch_size=ModelConfig.BATCH_SIZE)

    keras.utils.plot_model(model, os.path.join(ModelConfig.MODEL_DIR, "DeepLabV3Plus_model.png"), show_shapes=True)

    print(f'Model type: {type(model)}')
    print(f'Model output shape: {model.output.shape}')
    return model

def deeplab_v3_plus(width, height, input_channels):
    input_shape = (width, height, input_channels)
    model = DeeplabV3Plus(input_shape)
    model.compile(
        optimizer=get_optimizer(),
        loss=combined_loss_function,
        metrics=['accuracy', f1_score, precision_m, recall_m]
    )
    return model

if __name__ == '__main__':
    config_file = 'config.yaml'
    load_config(config_file)
    deeplab_v3_plus(ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH, ModelConfig.MODEL_INPUT_CHANNELS)
