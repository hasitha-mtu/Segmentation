import keras
from tensorflow.keras.layers import (GlobalAveragePooling2D, Reshape, Dense, Multiply)
from tensorflow.keras.models import Model

from models.common_utils.loss_functions import  recall_m, precision_m, f1_score
from loss_function import combined_masked_dice_bce_loss

def SE(inputs, ratio=0.8):
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


def res_unet_plus_plus(width, height, input_channels):
    input_shape = (width, height, input_channels)
    model = ResUNET(input_shape)
    model.compile(
        optimizer='adam',
        loss=combined_masked_dice_bce_loss,
        metrics=['accuracy', f1_score, precision_m, recall_m]
    )
    return model

if __name__=='__main__':
    res_unet_plus_plus(512, 512, 3)

