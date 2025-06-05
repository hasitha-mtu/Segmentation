from tensorflow.keras.layers import Input
import keras
import tensorflow as tf
from tensorflow.keras import layers

# Encoder 13 Conv layers
# Decoder 13 Conv layers
# Activation function is ReLU
# Max-pooling with a 2 × 2 window and stride 2

class MaxPoolingWithArgmax2D(tf.keras.layers.Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='SAME', **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def call(self, inputs):
        pooled, argmax = tf.nn.max_pool_with_argmax(
            inputs,
            ksize=[1, *self.pool_size, 1],
            strides=[1, *self.strides, 1],
            padding=self.padding,
            output_dtype=tf.int64
        )
        return pooled, argmax

class MaxUnpooling2D(layers.Layer):
    def __init__(self, pool_size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.pool_size = pool_size

    def call(self, inputs, argmax, output_shape=None):
        input_shape = tf.shape(inputs)

        if output_shape is None:
            output_shape = tf.stack([
                input_shape[0],
                input_shape[1] * self.pool_size[0],
                input_shape[2] * self.pool_size[1],
                input_shape[3]
            ])
        else:
            if isinstance(output_shape, tf.TensorShape):
                output_shape = output_shape.as_list()

            for i, dim in enumerate(output_shape):
                if dim is None:
                    output_shape[i] = tf.shape(inputs)[i]

            output_shape = tf.stack(output_shape)
            output_shape = tf.cast(output_shape, tf.int64)

        flat_input = tf.reshape(inputs, [-1])
        flat_argmax = tf.reshape(argmax, [-1])
        batch_size = output_shape[0]
        height = output_shape[1]
        width = output_shape[2]
        channels = output_shape[3]

        output_elements_per_batch = height * width * channels

        batch_range = tf.range(batch_size, dtype=tf.int64)
        batch_offset = batch_range * output_elements_per_batch
        batch_offset = tf.reshape(batch_offset, [-1, 1])

        num_pool_elements = tf.cast(tf.shape(flat_argmax)[0], tf.int64) // batch_size
        batch_offset = tf.tile(batch_offset, [1, num_pool_elements])
        batch_offset = tf.reshape(batch_offset, [-1])

        flat_argmax = flat_argmax + batch_offset

        output_shape_flat = tf.reduce_prod(output_shape)
        output = tf.scatter_nd(
            indices=tf.expand_dims(flat_argmax, 1),
            updates=flat_input,
            shape=[output_shape_flat]
        )

        output = tf.reshape(output, output_shape)
        return output

def segnet_model(width, height, num_channels):
    input_shape = (width, height, num_channels)
    return segnet(input_shape, n_classes=1)

def segnet(input_shape=(256, 256, 3), n_classes=2):
    inputs = tf.keras.Input(shape=input_shape)

    # --- Encoder ---
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    x, idx1 = MaxPoolingWithArgmax2D()(x)
    x1_shape = x.shape

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    x, idx2 = MaxPoolingWithArgmax2D()(x)
    x2_shape = x.shape

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    x, idx3 = MaxPoolingWithArgmax2D()(x)
    x3_shape = x.shape

    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)

    x, idx4 = MaxPoolingWithArgmax2D()(x)
    x4_shape = x.shape

    # --- Decoder ---
    x = MaxUnpooling2D()(x, idx4, output_shape=x4_shape)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)

    x = MaxUnpooling2D()(x, idx3, output_shape=x3_shape)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    x = MaxUnpooling2D()(x, idx2, output_shape=x2_shape)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    x = MaxUnpooling2D()(x, idx1, output_shape=x1_shape)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    # Final classifier
    if n_classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    outputs = layers.Conv2D(n_classes, (1, 1), padding='same', activation=activation)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print(f"Model : {model.summary()}")

    keras.utils.plot_model(model, "segnet_model.png", show_shapes=True)

    return model

if __name__ == '__main__':
    segnet_model(512, 512, 3)

