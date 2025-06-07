import tensorflow as tf
from tensorflow.keras import layers, Model
import keras
from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from models.common_utils.loss_functions import (recall_m, precision_m, f1_score,
                                                combined_masked_dice_focal_loss)

from tensorflow.keras.applications import VGG16

def build_rgb_segnet(input_shape=(512, 512, 3), num_classes=1):
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Encoder outputs
    _x1 = vgg16.get_layer('block1_pool').output
    _x2 = vgg16.get_layer('block2_pool').output
    _x3 = vgg16.get_layer('block3_pool').output
    _x4 = vgg16.get_layer('block4_pool').output
    x5 = vgg16.get_layer('block5_pool').output

    # Decoder (mirroring VGG16 structure)
    x = UpSampling2D((2, 2))(x5)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Dropout(0.2)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Dropout(0.2)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Dropout(0.2)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Dropout(0.2)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Conv2D(num_classes, (1, 1), padding='same')(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs=vgg16.input, outputs=x, name="SegNet-RGB-VGG16")

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print(f"SegNet-RGB-VGG16 Model : {model.summary()}")

    return model

def extend_to_16_channel_model(pretrained_model, input_shape=(512, 512, 16)):
    # New input layer for 16-channel input
    inputs = Input(shape=input_shape)

    # Project 16 channels to 3 to match original VGG16
    projected = Conv2D(3, (1, 1), padding='same', activation='relu', name='input_projection')(inputs)

    # Feed into the original model (excluding its input layer)
    x = pretrained_model(projected)

    model = Model(inputs=inputs, outputs=x, name='segnet_16ch')

    model.compile(
        optimizer='adam',
        loss=combined_masked_dice_focal_loss,  # Use `None` to skip the second loss
        metrics=[['accuracy', f1_score, precision_m, recall_m], []]  # Metrics only for the segmentation output
    )

    print(f"SegNet-16Ch-VGG16 Model : {model.summary()}")

    keras.utils.plot_model(model, "SegNet-16Ch-VGG16.png", show_shapes=True)

    return model


# -------------------------------------------------------------------------------------------------------------------

def segnet_encoder_block(inputs, filters):
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x, indices = tf.nn.max_pool_with_argmax(x, ksize=2, strides=2, padding='SAME', output_dtype=tf.int32)
    return x, indices

def segnet_decoder_block(inputs, indices, filters, output_shape):
    x = tf.keras.layers.Lambda(
        lambda args: tf.scatter_nd(
            tf.expand_dims(tf.reshape(args[1], [-1]), axis=1),
            tf.reshape(args[0], [-1]),
            [tf.reduce_prod(output_shape)]
        )
    )([inputs, indices])
    x = tf.reshape(x, output_shape)
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    return x

def build_segnet(input_shape=(512, 512, 16), num_classes=1):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x1, idx1 = segnet_encoder_block(inputs, 64)
    x2, idx2 = segnet_encoder_block(x1, 128)
    x3, idx3 = segnet_encoder_block(x2, 256)
    x4, idx4 = segnet_encoder_block(x3, 512)
    x5, idx5 = segnet_encoder_block(x4, 512)

    # Decoder (reverse of encoder)
    d5 = tf.image.resize(x5, tf.shape(x4)[1:3], method='nearest')  # simulate unpooling
    d5 = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(d5)

    d4 = tf.image.resize(d5, tf.shape(x3)[1:3], method='nearest')
    d4 = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(d4)

    d3 = tf.image.resize(d4, tf.shape(x2)[1:3], method='nearest')
    d3 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(d3)

    d2 = tf.image.resize(d3, tf.shape(x1)[1:3], method='nearest')
    d2 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(d2)

    d1 = tf.image.resize(d2, tf.shape(inputs)[1:3], method='nearest')
    d1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(d1)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(d1)

    model = Model(inputs, outputs, name="SegNet-VGG16")

    model.compile(
        optimizer='adam',
        loss=combined_masked_dice_focal_loss,  # Use `None` to skip the second loss
        metrics=[['accuracy', f1_score, precision_m, recall_m], []]  # Metrics only for the segmentation output
    )

    print(f"Model : {model.summary()}")

    keras.utils.plot_model(model, "SegNet-VGG16.png", show_shapes=True)

    return model

def segnet_model(width, height, num_channels):
    input_shape = (width, height, num_channels)
    return build_segnet(input_shape=input_shape, num_classes=1)

if __name__ == "__main__":
    segnet_model(512, 512, 1)

