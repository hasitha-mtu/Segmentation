import keras
import tensorflow as tf
from models.common_utils.loss_functions import  recall_m, precision_m, f1_score, masked_dice_loss


def encoding_block(inputs, filters, dropout, batch_normalization=True, pooling=True, kernel_size=(3,3), activation="relu",
                   kernel_initializer="he_normal", padding="same"):
    if batch_normalization:
        inputs = tf.keras.layers.BatchNormalization()(inputs)

    c = tf.keras.layers.Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer,
                               padding=padding)(inputs)
    c = tf.keras.layers.Dropout(dropout)(c)
    c = tf.keras.layers.Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer,
                               padding=padding)(c)
    if pooling:
        p = tf.keras.layers.MaxPooling2D((2, 2))(c)
        return c, p
    else:
        return c


def decoding_block(inputs, conv, filters, batch_normalization=True, kernel_size=(3,3), strides=(2,2), activation="relu",
                   kernel_initializer="he_normal", padding="same"):
    if batch_normalization:
        inputs = tf.keras.layers.BatchNormalization()(inputs)

    u = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(inputs)
    u = tf.keras.layers.concatenate([u, conv])
    c = tf.keras.layers.Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer,
                               padding=padding)(u)
    c = tf.keras.layers.Dropout(0.2)(c)
    c = tf.keras.layers.Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer,
                               padding=padding)(c)
    return c

def unet_model(image_width, image_height, image_channels):
    inputs = tf.keras.Input(shape=(image_height, image_width, image_channels))  # 8-channel input
    x, attention_weights = channel_attention_block(inputs)  # Apply attention
    # Encoding
    c1, p1 = encoding_block(x, 16, 0.3)
    c2, p2 = encoding_block(p1, 32, 0.3)
    c3, p3 = encoding_block(p2, 64, 0.3)
    c4, p4 = encoding_block(p3, 128, 0.3)
    c5, p5 = encoding_block(p4, 256, 0.3)
    c6, p6 = encoding_block(p5, 512, 0.3)
    c7 = encoding_block(p6, 1024, 0.3, pooling=False)
    # Decoding
    u1 = decoding_block(c7, c6, 512)
    u2 = decoding_block(u1, c5, 256)
    u3 = decoding_block(u2, c4, 128)
    u4 = decoding_block(u3, c3, 64)
    u5 = decoding_block(u4, c2, 32)
    u6 = decoding_block(u5, c1, 16)

    outputs = tf.keras.layers.Conv2D(3, kernel_size=3, activation='sigmoid', padding='same')(u6)

    model = tf.keras.Model(inputs=inputs,
                           outputs=[outputs, attention_weights],
                           name="unet_with_attention")

    model.compile(
        optimizer='adam',
        loss=[masked_dice_loss, None], # Use `None` to skip the second loss
        metrics=[['accuracy', f1_score, precision_m, recall_m], []] # Metrics only for the segmentation output
    )

    print(f"Model : {model.summary()}")

    keras.utils.plot_model(model, "unet_model.png", show_shapes=True)

    return model


def channel_attention_block(inputs, reduction_ratio=8, name="channel_attention"):
    channels = inputs.shape[-1]

    # Global average pooling: (B, H, W, C) → (B, C)
    x = tf.keras.layers.GlobalAveragePooling2D(name=f"{name}_gap")(inputs)

    # Fully connected layers: (B, C) → (B, C//r) → (B, C)
    x = tf.keras.layers.Dense(channels // reduction_ratio, activation='relu', name=f"{name}_fc1")(x)
    x = tf.keras.layers.Dense(channels, activation='sigmoid', name=f"{name}_fc2")(x)  # channel weights

    # Reshape to (B, 1, 1, C) and scale the input
    x = tf.keras.layers.Reshape((1, 1, channels), name=f"{name}_reshape")(x)
    scaled = tf.keras.layers.Multiply(name=f"{name}_scale")([inputs, x])

    return scaled, x  # also return the attention weights for inspection

if __name__ == '__main__':
    unet_model(512, 512, 5)



