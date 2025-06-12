import tensorflow as tf
from tensorflow.keras import layers
import keras
from models.common_utils.loss_functions import  recall_m, precision_m, f1_score
from loss_function import combined_masked_dice_bce_loss

# --- Fast Fourier Convolution (FFC) block ---
class FFC(tf.keras.layers.Layer):
    def __init__(self, filters, ratio_g=0.5, **kwargs):
        super(FFC, self).__init__(**kwargs)
        self.ratio_g = ratio_g
        self.filters = filters
        self.conv_l2l = None
        self.conv_l2g = None
        self.conv_g2l = None
        self.conv_g2g = None

    def build(self, input_shape):
        in_channels = input_shape[-1]
        filters_g = max(1, int(in_channels * self.ratio_g)) if self.ratio_g > 0 else 0
        filters_l = self.filters - filters_g

        self.filters_g = filters_g
        self.filters_l = filters_l

        self.conv_l2l = tf.keras.layers.Conv2D(filters_l, 3, padding="same", activation="relu")
        self.conv_l2g = tf.keras.layers.Conv2D(filters_g, 3, padding="same", activation="relu")

        self.conv_g2l = tf.keras.layers.Conv2D(filters_l, 3, padding="same", activation="relu")
        self.conv_g2g = tf.keras.layers.Conv2D(filters_g, 3, padding="same", activation="relu")

    def call(self, x):
        # Safe split if input channels are known at runtime
        try:
            split_l = x.shape[-1] - self.filters_g  # Static shape for splitting
            split_g = self.filters_g
            if split_l > 0 and split_g > 0:
                x_l, x_g = tf.split(x, [split_l, split_g], axis=-1)
            else:
                x_l = x
                x_g = None
        except:
            x_l = x
            x_g = None

        l2l = self.conv_l2l(x_l)
        l2g = self.conv_l2g(x_l)

        if x_g is not None:
            fft = tf.signal.fft2d(tf.cast(x_g, tf.complex64))
            fft_real = tf.math.real(fft)
            g2l = self.conv_g2l(fft_real)
            g2g = self.conv_g2g(fft_real)
        else:
            g2l = 0
            g2g = 0

        out_l = l2l + g2l
        out_g = l2g + g2g if self.filters_g > 0 else None

        return tf.concat([out_l, out_g], axis=-1) if out_g is not None else out_l

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "ratio_g": self.ratio_g,
        })
        return config

# --- Updated encoding block with optional FFC ---
def encoding_block(inputs, filters, dropout, use_ffc=False, batch_normalization=True, pooling=True):
    if batch_normalization:
        inputs = tf.keras.layers.BatchNormalization()(inputs)

    if use_ffc:
        c = FFC(filters=filters)(inputs)
        c = tf.keras.layers.Dropout(dropout)(c)
        c = FFC(filters=filters)(c)
    else:
        c = tf.keras.layers.Conv2D(filters, 3, activation="relu", padding="same")(inputs)
        c = tf.keras.layers.Dropout(dropout)(c)
        c = tf.keras.layers.Conv2D(filters, 3, activation="relu", padding="same")(c)

    if pooling:
        p = tf.keras.layers.MaxPooling2D((2, 2))(c)
        return c, p
    else:
        return c

# --- Decoding block remains unchanged ---
def decoding_block(inputs, conv, filters):
    inputs = tf.keras.layers.BatchNormalization()(inputs)
    u = tf.keras.layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding="same")(inputs)
    u = tf.keras.layers.concatenate([u, conv])
    c = tf.keras.layers.Conv2D(filters, 3, activation="relu", padding="same")(u)
    c = tf.keras.layers.Dropout(0.2)(c)
    c = tf.keras.layers.Conv2D(filters, 3, activation="relu", padding="same")(c)
    return c

# --- Final UNet model with FFC in deeper encoders ---
def unet_model(image_width, image_height, image_channels):
    inputs = tf.keras.Input(shape=(image_height, image_width, image_channels))

    c1, p1 = encoding_block(inputs, 16, 0.3, use_ffc=False)
    c2, p2 = encoding_block(p1, 32, 0.3, use_ffc=False)
    c3, p3 = encoding_block(p2, 64, 0.3, use_ffc=True)
    c4, p4 = encoding_block(p3, 128, 0.3, use_ffc=True)
    c5, p5 = encoding_block(p4, 256, 0.3, use_ffc=True)
    c6, p6 = encoding_block(p5, 512, 0.3, use_ffc=True)
    c7 = encoding_block(p6, 1024, 0.3, use_ffc=True, pooling=False)

    u1 = decoding_block(c7, c6, 512)
    u2 = decoding_block(u1, c5, 256)
    u3 = decoding_block(u2, c4, 128)
    u4 = decoding_block(u3, c3, 64)
    u5 = decoding_block(u4, c2, 32)
    u6 = decoding_block(u5, c1, 16)

    outputs = tf.keras.layers.Conv2D(3, kernel_size=3, activation='sigmoid', padding='same', name="segmentation_output")(u6)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name="UNET_FFC")

    model.compile(
        optimizer='adam',
        loss=combined_masked_dice_bce_loss,
        metrics=['accuracy', f1_score, precision_m, recall_m]
    )

    print("Model output shape:", model.output_shape)

    print(f"Model summary : {model.summary()}")

    keras.utils.plot_model(model, "UNET_FFC.png", show_shapes=True)

    return model


if __name__ == '__main__':
    unet_model(512, 512, 5)