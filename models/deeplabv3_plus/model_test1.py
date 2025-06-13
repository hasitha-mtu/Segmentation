import tensorflow as tf
from tensorflow.keras import layers
import keras

from models.common_utils.loss_functions import  recall_m, precision_m, f1_score
from loss_function import combined_masked_dice_bce_loss

def conv_bn_relu(x, filters, kernel_size=3, strides=1, dilation=1, padding='same', name=None):
    x = layers.Conv2D(filters, kernel_size, strides=strides, dilation_rate=dilation,
                      padding=padding, use_bias=False,
                      kernel_initializer='he_normal', name=None if not name else name+'_conv')(x)
    x = layers.BatchNormalization(name=None if not name else name+'_bn')(x)
    x = layers.Activation('relu', name=None if not name else name+'_relu')(x)
    return x

def bottleneck_block(x, filters, strides=1, dilation=1, use_projection=False, name=None):
    shortcut = x
    if use_projection:
        shortcut = layers.Conv2D(filters * 4, 1, strides=strides, use_bias=False,
                                 kernel_initializer='he_normal', name=None if not name else name+'_proj_conv')(x)
        shortcut = layers.BatchNormalization(name=None if not name else name+'_proj_bn')(shortcut)

    x = layers.Conv2D(filters, 1, strides=1, use_bias=False,
                      kernel_initializer='he_normal', name=None if not name else name+'_conv1')(x)
    x = layers.BatchNormalization(name=None if not name else name+'_bn1')(x)
    x = layers.Activation('relu', name=None if not name else name+'_relu1')(x)

    x = layers.Conv2D(filters, 3, strides=strides, dilation_rate=dilation, padding='same', use_bias=False,
                      kernel_initializer='he_normal', name=None if not name else name+'_conv2')(x)
    x = layers.BatchNormalization(name=None if not name else name+'_bn2')(x)
    x = layers.Activation('relu', name=None if not name else name+'_relu2')(x)

    x = layers.Conv2D(filters * 4, 1, use_bias=False,
                      kernel_initializer='he_normal', name=None if not name else name+'_conv3')(x)
    x = layers.BatchNormalization(name=None if not name else name+'_bn3')(x)

    x = layers.Add(name=None if not name else name+'_add')([shortcut, x])
    x = layers.Activation('relu', name=None if not name else name+'_out')(x)
    return x

def resnet_layer(x, filters, blocks, strides=1, dilation=1, name=None):
    # First block with projection if strides>1 or input/output channels mismatch
    x = bottleneck_block(x, filters, strides=strides, dilation=dilation, use_projection=True,
                         name=None if not name else name+'_block1')
    for i in range(2, blocks + 1):
        x = bottleneck_block(x, filters, strides=1, dilation=dilation, use_projection=False,
                             name=None if not name else f'{name}_block{i}')
    return x

def ASPP(x, filters, rate_list=[6, 12, 18], name=None):
    dims = x.shape
    pool = layers.GlobalAveragePooling2D(name=None if not name else name+'_gap')(x)
    pool = layers.Reshape((1,1,dims[-1]), name=None if not name else name+'_gap_reshape')(pool)
    pool = layers.Conv2D(filters, 1, padding='same', use_bias=False,
                         kernel_initializer='he_normal', name=None if not name else name+'_gap_conv')(pool)
    pool = layers.BatchNormalization(name=None if not name else name+'_gap_bn')(pool)
    pool = layers.Activation('relu', name=None if not name else name+'_gap_relu')(pool)
    pool = layers.UpSampling2D(size=(dims[1], dims[2]), interpolation='bilinear',
                               name=None if not name else name+'_gap_upsample')(pool)

    conv_1x1 = layers.Conv2D(filters, 1, padding='same', use_bias=False,
                             kernel_initializer='he_normal', name=None if not name else name+'_conv_1x1')(x)
    conv_1x1 = layers.BatchNormalization(name=None if not name else name+'_conv_1x1_bn')(conv_1x1)
    conv_1x1 = layers.Activation('relu', name=None if not name else name+'_conv_1x1_relu')(conv_1x1)

    conv_3x3_1 = layers.Conv2D(filters, 3, padding='same', dilation_rate=rate_list[0], use_bias=False,
                              kernel_initializer='he_normal', name=None if not name else f'{name}_conv_3x3_1')(x)
    conv_3x3_1 = layers.BatchNormalization(name=None if not name else f'{name}_conv_3x3_1_bn')(conv_3x3_1)
    conv_3x3_1 = layers.Activation('relu', name=None if not name else f'{name}_conv_3x3_1_relu')(conv_3x3_1)

    conv_3x3_2 = layers.Conv2D(filters, 3, padding='same', dilation_rate=rate_list[1], use_bias=False,
                              kernel_initializer='he_normal', name=None if not name else f'{name}_conv_3x3_2')(x)
    conv_3x3_2 = layers.BatchNormalization(name=None if not name else f'{name}_conv_3x3_2_bn')(conv_3x3_2)
    conv_3x3_2 = layers.Activation('relu', name=None if not name else f'{name}_conv_3x3_2_relu')(conv_3x3_2)

    conv_3x3_3 = layers.Conv2D(filters, 3, padding='same', dilation_rate=rate_list[2], use_bias=False,
                              kernel_initializer='he_normal', name=None if not name else f'{name}_conv_3x3_3')(x)
    conv_3x3_3 = layers.BatchNormalization(name=None if not name else f'{name}_conv_3x3_3_bn')(conv_3x3_3)
    conv_3x3_3 = layers.Activation('relu', name=None if not name else f'{name}_conv_3x3_3_relu')(conv_3x3_3)

    x = layers.Concatenate(name=None if not name else name+'_concat')([pool, conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3])
    x = layers.Conv2D(filters, 1, padding='same', use_bias=False,
                      kernel_initializer='he_normal', name=None if not name else name+'_conv_out')(x)
    x = layers.BatchNormalization(name=None if not name else name+'_conv_out_bn')(x)
    x = layers.Activation('relu', name=None if not name else name+'_conv_out_relu')(x)
    x = layers.Dropout(0.5, name=None if not name else name+'_dropout')(x)
    return x

def DeepLabV3Plus(input_shape=(512, 512, 5), num_classes=1):
    inputs = layers.Input(shape=input_shape)

    # Initial conv + BN + ReLU + MaxPool
    x = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False, kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Encoder (ResNet blocks)
    # Conv2_x
    x = resnet_layer(x, filters=64, blocks=3, strides=1, name='conv2')
    # Conv3_x
    x = resnet_layer(x, filters=128, blocks=4, strides=2, name='conv3')
    # Save low-level features here for decoder skip connection
    low_level_feat = x

    # Conv4_x (with dilation)
    x = resnet_layer(x, filters=256, blocks=6, strides=1, dilation=2, name='conv4')
    # Conv5_x (with dilation)
    x = resnet_layer(x, filters=512, blocks=3, strides=1, dilation=4, name='conv5')

    # ASPP
    x = ASPP(x, 256, name='ASPP')

    # Decoder
    x = layers.UpSampling2D(size=(4,4), interpolation='bilinear')(x)  # Upsample ASPP output

    # Low-level feature processing (reduce channels)
    low_level_feat = layers.Conv2D(48, 1, padding='same', use_bias=False,
                                  kernel_initializer='he_normal')(low_level_feat)
    low_level_feat = layers.BatchNormalization()(low_level_feat)
    low_level_feat = layers.Activation('relu')(low_level_feat)

    # Concatenate
    x = layers.Concatenate()([x, low_level_feat])

    # Decoder conv layers
    x = layers.Conv2D(256, 3, padding='same', use_bias=False,
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(256, 3, padding='same', use_bias=False,
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Final upsampling to input size
    x = layers.UpSampling2D(size=(4,4), interpolation='bilinear')(x)

    # Output segmentation mask
    if num_classes == 1:
        outputs = layers.Conv2D(num_classes, 1, padding='same', activation='sigmoid')(x)
    else:
        outputs = layers.Conv2D(num_classes, 1, padding='same', activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='DeepLabV3Plus_Custom')

    model.compile(
        optimizer='adam',
        loss=combined_masked_dice_bce_loss,
        metrics=['accuracy', f1_score, precision_m, recall_m]
    )

    print("Model output shape:", model.output_shape)

    print(f"Model summary : {model.summary()}")

    keras.utils.plot_model(model, "DeepLabV3Plus_model.png", show_shapes=True)

    return model

if __name__ == "__main__":
    DeepLabV3Plus(input_shape=(512, 512, 5), num_classes=1)
