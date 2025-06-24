
from tensorflow.keras.layers import (Conv2D, BatchNormalization, Activation, UpSampling2D,
                                     AveragePooling2D, Concatenate, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
import keras

from models.common_utils.loss_functions import  recall_m, precision_m, f1_score
from models.deeplabv3_plus.loss_function import combined_masked_dice_bce_loss
from models.memory_usage import estimate_model_memory_usage

def ASPP(inputs):
    shape = inputs.shape
    y_pool = AveragePooling2D(pool_size=(shape[1], shape[2]))(inputs)
    print(f'AveragePooling2D y_pool shape:{y_pool.shape}')
    y_pool = Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(y_pool)
    print(f'Conv2D y_pool shape:{y_pool.shape}')
    y_pool = BatchNormalization()(y_pool)
    y_pool = Activation('relu')(y_pool)
    y_pool = UpSampling2D((shape[1], shape[2]), interpolation='bilinear')(y_pool)
    print(f'UpSampling2D y_pool shape:{y_pool.shape}')

    y_1 = Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(inputs)
    print(f'Conv2D y_1 shape:{y_1.shape}')
    y_1 = BatchNormalization()(y_1)
    y_1 = Activation('relu')(y_1)

    y_6 = Conv2D(filters=256, kernel_size=1, dilation_rate=6, padding='same', use_bias=False)(inputs)
    print(f'Conv2D y_6 shape:{y_6.shape}')
    y_6 = BatchNormalization()(y_6)
    y_6 = Activation('relu')(y_6)

    y_12 = Conv2D(filters=256, kernel_size=1, dilation_rate=12, padding='same', use_bias=False)(inputs)
    print(f'Conv2D y_12 shape:{y_12.shape}')
    y_12 = BatchNormalization()(y_12)
    y_12 = Activation('relu')(y_12)

    y_18 = Conv2D(filters=256, kernel_size=1, dilation_rate=18, padding='same', use_bias=False)(inputs)
    print(f'Conv2D y_18 shape:{y_18.shape}')
    y_18 = BatchNormalization()(y_18)
    y_18 = Activation('relu')(y_18)

    y = Concatenate()([y_pool, y_1, y_6, y_12, y_18])
    print(f'Concatenate y shape:{y.shape}')

    y = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same', use_bias=False)(y)
    print(f'Conv2D y shape:{y.shape}')
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    return y

def DeepLabV3Plus(shape):
    inputs = Input(shape=shape)

    #  Pre-trained ResNet50
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)

    # Pre-trained ResNet50 output
    image_features = base_model.get_layer('conv4_block6_out').output
    print(f'image_features shape:{image_features.shape}')

    x_a = ASPP(image_features)
    print(f'x_a shape:{x_a.shape}')
    x_a = UpSampling2D((4, 4), interpolation='bilinear')(x_a)
    print(f'UpSampling2D x_a shape:{x_a.shape}')

    # Get low-level features
    x_b = base_model.get_layer('conv2_block2_out').output
    x_b = Conv2D(filters=48, kernel_size=1, padding='same', use_bias=False)(x_b)
    print(f'Conv2D x_b shape:{x_b.shape}')
    x_b = BatchNormalization()(x_b)
    x_b = Activation('relu')(x_b)

    x = Concatenate()([x_a, x_b])
    print(f'Concatenate x shape:{x.shape}')

    x = Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(x)
    print(f'Conv2D x shape:{x.shape}')
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(x)
    print(f'Conv2D x shape:{x.shape}')
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D((4, 4), interpolation='bilinear')(x)
    print(f'UpSampling2D x shape:{x.shape}')

    # Output
    x = Conv2D(1, (1, 1), name='output_layer')(x)
    x = Activation('sigmoid')(x)

    # Model
    model = Model(inputs=inputs, outputs=x)
    print("Model output shape:", model.output_shape)
    print(f"Model summary : {model.summary()}")

    estimate_model_memory_usage(model, batch_size=4)

    keras.utils.plot_model(model, "DeepLabV3Plus_model.png", show_shapes=True)

    return model

def deeplab_v3_plus(width, height, input_channels):
    input_shape = (width, height, input_channels)
    model = DeepLabV3Plus(input_shape)
    model.compile(
        optimizer='adam',
        loss=combined_masked_dice_bce_loss,
        metrics=['accuracy', f1_score, precision_m, recall_m]
    )
    return model

if __name__ == '__main__':
    input_shape = (256, 256, 3)
    DeepLabV3Plus(input_shape)
