from tensorflow.keras.layers import Input
import keras
import tensorflow as tf
from tensorflow.keras import layers

# Encoder 13 Conv layers
# Decoder 13 Conv layers
# Activation function is ReLU
# Max-pooling with a 2 × 2 window and stride 2

# --------- Custom MaxUnpooling2D layer ---------
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
        batch_offset = batch_range * tf.cast(output_elements_per_batch, tf.int64)
        batch_offset = tf.reshape(batch_offset, [-1, 1])
        num_pool_elements = tf.cast(tf.shape(flat_argmax)[0], tf.int32) // batch_size
        # num_pool_elements = tf.cast(tf.shape(flat_argmax)[0], tf.int64) // batch_size
        batch_offset = tf.tile(batch_offset, [1, num_pool_elements])
        batch_offset = tf.reshape(batch_offset, [-1])

        flat_argmax = flat_argmax + batch_offset
        output_flat = tf.scatter_nd(indices=tf.expand_dims(flat_argmax, 1), updates=flat_input, shape=[tf.reduce_prod(output_shape)])
        output = tf.reshape(output_flat, output_shape)
        return output

# --------- MaxPooling with Indices ---------
def max_pool_with_argmax(x):
    pool, argmax = tf.nn.max_pool_with_argmax(x, ksize=2, strides=2, padding='SAME', output_dtype=tf.int64)
    return pool, argmax

# --------- Build SegNet ---------
def build_segnet(input_shape=(224, 224, 3), num_classes=21):
    inputs = layers.Input(shape=input_shape)

    # Encoder Block 1
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x, ind1 = max_pool_with_argmax(x)

    # Encoder Block 2
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x, ind2 = max_pool_with_argmax(x)

    # Encoder Block 3
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x, ind3 = max_pool_with_argmax(x)

    # Encoder Block 4
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x, ind4 = max_pool_with_argmax(x)

    # Encoder Block 5
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x, ind5 = max_pool_with_argmax(x)

    # Decoder Block 5
    x = MaxUnpooling2D()(x, ind5)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Decoder Block 4
    x = MaxUnpooling2D()(x, ind4)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Decoder Block 3
    x = MaxUnpooling2D()(x, ind3)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Decoder Block 2
    x = MaxUnpooling2D()(x, ind2)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Decoder Block 1
    x = MaxUnpooling2D()(x, ind1)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Final Classifier
    if num_classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    outputs = layers.Conv2D(num_classes, (1, 1), padding='same', activation=activation)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='SegNet')
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(f"Model : {model.summary()}")

    keras.utils.plot_model(model, "segnet_model.png", show_shapes=True)

    return model

def segnet_model(width, height, num_channels):
    input_shape = (width, height, num_channels)
    return build_segnet(input_shape=input_shape, num_classes=3)


if __name__ == '__main__':
    segnet_model(512, 512, 16)

