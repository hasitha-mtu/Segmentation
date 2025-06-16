import keras
from tensorflow.keras.layers import (GlobalAveragePooling2D, Reshape, Dense, Multiply,
                                     Conv2D, BatchNormalization, Add, Activation,
                                     MaxPooling2D, UpSampling2D, Concatenate)
from tensorflow.keras.models import Model
from tensorflow.python.keras import Input

from models.common_utils.loss_functions import  recall_m, precision_m, f1_score
from loss_function import combined_masked_dice_bce_loss


def SE(inputs, ratio=8):
    ## [8, H, W, 32]
    channel_axis = -1
    num_filters = inputs.shape[channel_axis]
    se_shape = (1, 1, num_filters)

    x = GlobalAveragePooling2D()(inputs) ## [8, 32]
    x = Reshape(se_shape)(x)
    x = Dense(num_filters // ratio, activation='relu', use_bias=False)(x)
    x = Dense(num_filters, activation='sigmoid', use_bias=False)(x)

    x = Multiply()([inputs, x])
    return x

def stem_block(inputs, num_filters):
    ## Conv 1
    x = Conv2D(num_filters, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, 3, padding='same')(x)

    ## Shortcut
    s = Conv2D(num_filters, 1, padding='same')(inputs)

    ## Add
    x = Add()([x, s])

    return x

def resnet_block(inputs, num_filters, strides=1):
    ## SE
    inputs = SE(inputs)

    ## Conv 1
    x = BatchNormalization()(inputs)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, 3, padding='same', strides=strides)(x)

    ## Conv 2
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, 3, padding='same', strides=1)(x)

    ## Shortcut
    s = Conv2D(num_filters, 1, padding='same', strides=strides)(inputs)

    ## Add
    x = Add()([x, s])

    return x

def aspp_block(inputs, num_filters):
    x1 = Conv2D(num_filters, 3, dilation_rate=6, padding='same')(inputs)
    x1 = BatchNormalization()(x1)

    x2 = Conv2D(num_filters, 3, dilation_rate=12, padding='same')(inputs)
    x2 = BatchNormalization()(x2)

    x3 = Conv2D(num_filters, 3, dilation_rate=18, padding='same')(inputs)
    x3 = BatchNormalization()(x3)

    x4 = Conv2D(num_filters, (3, 3), padding='same')(inputs)
    x4 = BatchNormalization()(x4)

    y = Add()([x1, x2, x3, x4])
    y = Conv2D(num_filters, 1, padding='same')(y)

    return y

def attention_block(x1, x2):
    num_filters = x2.shape[-1]

    x1_conv = BatchNormalization()(x1)
    x1_conv = Activation('relu')(x1_conv)
    x1_conv = Conv2D(num_filters, 3, padding='same')(x1_conv)
    x1_pool = MaxPooling2D((2, 2))(x1_conv)

    x2_conv = BatchNormalization()(x2)
    x2_conv = Activation('relu')(x2_conv)
    x2_conv = Conv2D(num_filters, 3, padding='same')(x2_conv)

    x = Add()([x1_pool, x2_conv])

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, 3, padding='same')(x)

    x = Multiply()([x, x2])

    return x


def ResUnetPlusPlus(input_shape):
    """" Inputs """
    inputs = Input(shape=input_shape)

    """" Encoder """
    c1 = stem_block(inputs, 16)
    c2 = resnet_block(c1, 32, strides=2)
    c3 = resnet_block(c2, 64, strides=2)
    c4 = resnet_block(c3, 128, strides=2)

    """" Bridge """
    b1 = aspp_block(c4, 256)

    """" Decoder """
    d1 = attention_block(c3, b1)
    d1 = UpSampling2D((2, 2))(d1)
    d1 = Concatenate()([d1, c3])
    d1 = resnet_block(d1, 128)

    d2 = attention_block(c2, d1)
    d2 = UpSampling2D((2, 2))(d2)
    d2 = Concatenate()([d2, c2])
    d2 = resnet_block(d2, 64)

    d3 = attention_block(c1, d2)
    d3 = UpSampling2D((2, 2))(d3)
    d3 = Concatenate()([d3, c1])
    d3 = resnet_block(d3, 32)

    """" Output """
    outputs = aspp_block(d3, 16)
    outputs = Conv2D(1, 1, padding='same')(outputs)
    outputs = Activation('sigmoid')(outputs)

    """" Model """
    model = Model(inputs=inputs, outputs=outputs)
    print(f"Model summary : {model.summary()}")
    keras.utils.plot_model(model, "ResUnetPlusPlus_model.png", show_shapes=True)

    return model


def res_unet_plus_plus(width, height, input_channels):
    input_shape = (width, height, input_channels)
    model = ResUnetPlusPlus(input_shape)
    model.compile(
        optimizer='adam',
        loss=combined_masked_dice_bce_loss,
        metrics=['accuracy', f1_score, precision_m, recall_m]
    )
    return model

if __name__=='__main__':
    res_unet_plus_plus(512, 512, 3)

