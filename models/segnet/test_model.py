import tensorflow as tf
from tensorflow.keras import layers, models

# ---- MaxPooling with Argmax ----
class MaxPoolingWithArgmax2D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)

    def call(self, inputs):
        output, argmax = tf.nn.max_pool_with_argmax(
            inputs,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            output_dtype=tf.int64
        )
        return output, argmax

# ---- MaxUnpooling with argmax ----
class MaxUnpooling2D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)

    def call(self, inputs, argmax, output_shape=None):
        assert inputs.shape == argmax.shape
        input_shape = tf.shape(inputs)
        if output_shape is None:
            output_shape = (input_shape[0], input_shape[1]*2, input_shape[2]*2, input_shape[3])

        flat_input = tf.reshape(inputs, [-1])
        flat_argmax = tf.reshape(argmax, [-1])

        output_shape_tensor = tf.reduce_prod(output_shape)
        output = tf.scatter_nd(tf.expand_dims(flat_argmax, 1), flat_input, [output_shape_tensor])
        output = tf.reshape(output, output_shape)
        return output

# ---- SegNet Model ----
def build_segnet(input_shape=(128, 128, 5), num_classes=1):
    inputs = tf.keras.Input(shape=input_shape)

    # ---- Encoder ----
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x, idx1 = MaxPoolingWithArgmax2D()(x)

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x, idx2 = MaxPoolingWithArgmax2D()(x)

    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x, idx3 = MaxPoolingWithArgmax2D()(x)

    # ---- Decoder ----
    x = MaxUnpooling2D()(x, idx3)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)

    x = MaxUnpooling2D()(x, idx2)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)

    x = MaxUnpooling2D()(x, idx1)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)

    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs)

if __name__ == '__main__':
    # Example model creation
    model = build_segnet(input_shape=(256, 256, 5), num_classes=1)
    model.summary()