import keras
from tensorflow.keras.layers import (Conv2D, Activation, BatchNormalization,
                                     Conv2DTranspose, Input, Concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16

from models.common_utils.loss_functions import  recall_m, precision_m, f1_score
from models.unet_VGG16.loss_function import combined_masked_dice_bce_loss

from models.memory_usage import estimate_model_memory_usage
from models.common_utils.config import load_config, ModelConfig

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2,2), strides=2, padding='same')(inputs) # 64x64
    print(f'x shape: {x.shape}')
    print(f'skip_features shape: {skip_features.shape}')
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def UnetVGG16(input_shape):
    inputs = Input(shape=input_shape)
    print(f'inputs shape:{inputs.shape}')
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=inputs)
    vgg16.summary()

    """ Encoder """
    s1 = vgg16.get_layer('block1_conv2').output # 512x512
    print(f's1 shape: {s1.shape}')
    s2 = vgg16.get_layer('block2_conv2').output # 256x256
    print(f's2 shape: {s2.shape}')
    s3 = vgg16.get_layer('block3_conv3').output # 128x128
    print(f's3 shape: {s3.shape}')
    s4 = vgg16.get_layer('block4_conv3').output # 64x64
    print(f's4 shape: {s4.shape}')

    """ Bottleneck """
    b1 = vgg16.get_layer('block5_conv3').output  # 32x32
    print(f'b1 shape: {b1.shape}')

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)
    print(f'd1 shape: {d1.shape}')
    d2 = decoder_block(d1, s3, 256)
    print(f'd2 shape: {d2.shape}')
    d3 = decoder_block(d2, s2, 128)
    print(f'd3 shape: {d3.shape}')
    d4 = decoder_block(d3, s1, 64)
    print(f'd4 shape: {d4.shape}')

    """ Output """
    output = Conv2D(ModelConfig.MODEL_OUTPUT_CHANNELS, 1, padding='same', activation='sigmoid')(d4)
    model = Model(inputs=inputs,
                  outputs=output,
                  name=ModelConfig.MODEL_NAME)

    model.summary()

    estimate_model_memory_usage(model, batch_size=ModelConfig.BATCH_SIZE)

    keras.utils.plot_model(model, "Unet-VGG16_model.png", show_shapes=True)

    return model


def unet_vgg16(width, height, input_channels):
    input_shape = (width, height, input_channels)
    model = UnetVGG16(input_shape)
    model.compile(
        optimizer=ModelConfig.TRAINING_OPTIMIZER,
        loss=combined_masked_dice_bce_loss,
        metrics=['accuracy', f1_score, precision_m, recall_m]
    )
    return model

if __name__=='__main__':
    config_file = 'config.yaml'
    load_config(config_file)
    unet_vgg16(512, 512, 3)

