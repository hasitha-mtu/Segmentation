import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# --- LaMa-inspired Fast Fourier Convolution (FFC) Block (Simplified) ---
# This block aims to capture global context by processing features in the frequency domain.
# A full FFC implementation is more complex, involving detailed splits,
# concatenations, and handling of complex tensors.
# --- LaMa-inspired Fast Fourier Convolution (FFC) Block (Simplified) ---
class FFCBlock(keras.Model):
    def __init__(self, out_channels, kernel_size=3, padding='same', ratio_global=0.5, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.ratio_global = ratio_global
        self.local_channels = int(out_channels * (1 - ratio_global))
        self.global_channels = out_channels - self.local_channels

        self.conv_local = keras.Sequential([
            layers.Conv2D(self.local_channels, kernel_size=kernel_size, padding=padding),
            layers.ReLU()
        ])

        self.conv_global_initial = layers.Conv2D(self.global_channels, kernel_size=1, padding='valid')
        self.relu_global_initial = layers.ReLU()
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.fc_global = layers.Dense(self.global_channels)
        self.relu_global_processed = layers.ReLU()

    def call(self, inputs):
        x_local = self.conv_local(inputs)
        x_global_initial = self.relu_global_initial(self.conv_global_initial(inputs))
        x_global_pooled = self.avg_pool(x_global_initial)
        x_global_processed = self.relu_global_processed(self.fc_global(x_global_pooled))

        # Ensure x_global_initial has defined shape for `h` and `w` if inputs is symbolic tensor
        batch_size = tf.shape(x_global_initial)[0]
        h = tf.shape(x_global_initial)[1]
        w = tf.shape(x_global_initial)[2]

        x_global_spatial = tf.expand_dims(tf.expand_dims(x_global_processed, 1), 1)
        x_global_spatial = tf.tile(x_global_spatial, [1, h, w, 1])

        output = tf.concat([x_local, x_global_spatial], axis=-1)
        return output


class DoubleConv(keras.Model):
    def __init__(self, out_channels, mid_channels=None, **kwargs):
        super().__init__(**kwargs)
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = layers.Conv2D(mid_channels, kernel_size=3, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class Down(keras.Model):
    def __init__(self, out_channels, use_ffc=False, **kwargs):
        super().__init__(**kwargs)
        self.maxpool = layers.MaxPool2D(pool_size=2, strides=2)
        if use_ffc:
            self.conv_block = FFCBlock(out_channels)
        else:
            self.conv_block = DoubleConv(out_channels)

    def call(self, inputs):
        x = self.maxpool(inputs)
        return self.conv_block(x)


class Up(keras.Model):
    def __init__(self, out_channels, bilinear=True, **kwargs):
        super().__init__(**kwargs)
        self.bilinear = bilinear
        if bilinear:
            self.up = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
            self.conv_block = DoubleConv(out_channels // 2)
        else:
            self.up = layers.Conv2DTranspose(out_channels // 2, kernel_size=2, strides=2, padding='valid')
            self.conv_block = DoubleConv(out_channels)

    def call(self, x1, x2):
        x1 = self.up(x1)

        diff_y = tf.shape(x2)[1] - tf.shape(x1)[1]
        diff_x = tf.shape(x2)[2] - tf.shape(x1)[2]

        # CORRECTED SECTION: Using tf.cond for graph-compatible control flow
        # Define a lambda function for the true branch (padding)
        def true_fn():
            padding = [[0, 0],
                       [diff_y // 2, diff_y - diff_y // 2],
                       [diff_x // 2, diff_x - diff_x // 2],
                       [0, 0]]
            return tf.pad(x1, paddings=padding)

        # Define a lambda function for the false branch (no padding, return x1 as is)
        def false_fn():
            return x1

        # The condition needs to be a scalar boolean tensor
        condition = tf.logical_or(tf.greater(diff_y, 0), tf.greater(diff_x, 0))
        x1 = tf.cond(condition, true_fn, false_fn)

        x = tf.concat([x2, x1], axis=-1)
        return self.conv_block(x)


class OutConv(keras.Model):
    def __init__(self, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(out_channels, kernel_size=1, padding='valid')

    def call(self, inputs):
        return self.conv(inputs)


class UNetWithLaMaFeaturesTF(keras.Model):
    def __init__(self, n_channels, n_classes, bilinear=True, **kwargs):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(64)
        self.down1 = Down(128)
        self.down2 = Down(256)

        self.down3 = Down(512, use_ffc=True)
        self.down4 = Down(1024, use_ffc=True)

        self.up1 = Up(out_channels=512, bilinear=self.bilinear)
        self.up2 = Up(out_channels=256, bilinear=self.bilinear)
        self.up3 = Up(out_channels=128, bilinear=self.bilinear)
        self.up4 = Up(out_channels=64, bilinear=self.bilinear)
        self.outc = OutConv(n_classes)

    def call(self, inputs):
        x1 = self.inc(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_channels": self.n_channels,
            "n_classes": self.n_classes
        })
        return config

# --- Custom Metric Functions ---
def psnr_metric(y_true, y_pred):
    # Ensure y_true and y_pred are within the expected range (e.g., 0 to 1)
    # Clamp values to avoid NaN/Inf for PSNR if predictions go slightly out of range
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0) # Adjust 0.0, 1.0 based on your data normalization
    return tf.image.psnr(y_true, y_pred, max_val=1.0) # Set max_val according to your data range

def ssim_metric(y_true, y_pred):
    # Clamp values for SSIM as well
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0) # Adjust 0.0, 1.0 based on your data normalization
    return tf.image.ssim(y_true, y_pred, max_val=1.0) # Set max_val according to your data range

def unet_lama(input_width, input_height, input_channels, output_channels):
    model = UNetWithLaMaFeaturesTF(input_channels, output_channels)
    model.build(input_shape=(None, input_height, input_width, input_channels))

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss_function = tf.keras.losses.MeanAbsoluteError()

    metrics = [
        'accuracy',
        tf.keras.metrics.MeanAbsoluteError(name='mae'),
        tf.keras.metrics.MeanSquaredError(name='mse'),
        psnr_metric,  # Use the custom wrapper function here
        ssim_metric  # You can also add SSIM this way
    ]

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=metrics)
    print("Model compiled successfully!")

    print(f"Model summary : {model.summary()}")

    keras.utils.plot_model(model, "UNET-LaMa_model.png", show_shapes=True)

    return model

# Example Usage:
if __name__ == "__main__":
    # Example: Inpainting a 256x256 RGB image, outputting a 3-channel image
    input_channels = 3
    output_channels = 3  # For inpainting, output channels typically match input
    input_height = 256
    input_width = 256

    # Create a dummy input image (batch_size, height, width, channels)
    # TensorFlow uses NWHC format by default for Conv2D.
    dummy_input = tf.random.normal((1, input_height, input_width, input_channels))

    # Instantiate the model
    model = UNetWithLaMaFeaturesTF(input_channels, output_channels)

    # Build the model with a dummy input shape
    # This is important for Keras models to create their weights
    model.build(input_shape=(None, input_height, input_width, input_channels))

    # Pass the dummy input through the model
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Verify that the output shape matches the input shape for inpainting
    assert output.shape == dummy_input.shape
    print("Model architecture created successfully and shapes match!")

    # You can also print a summary of the model
    model.summary()

    # In a real inpainting task, you would prepare your masked image
    # For instance:
    # mask = tf.zeros_like(dummy_input)
    # mask = tf.tensor_scatter_nd_update(mask, [[0, i, j, k] for i in range(50,150) for j in range(50,150) for k in range(input_channels)], [1.0]*(100*100*input_channels)) # Example mask
    # masked_image = dummy_input * mask
    # Then feed masked_image to the model for training/inference.