import numpy as np
from tensorflow.keras import layers, Input, Model
import tensorflow as tf

def conv_block(inputs, model_width, kernel, multiplier):
    x = layers.Conv2D(model_width * multiplier, kernel, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def trans_conv(inputs, model_width, multiplier):
    x = layers.Conv2DTranspose(model_width * multiplier, (2, 2), strides=(2, 2), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def concat_block(inputs1, *args):
    concat = inputs1
    for arg in range(0, len(args)):
        concat = layers.concatenate([concat, args[arg]], axis=-1)
    return concat

def up_conv_block(inputs, size=(2, 2)):
    up = layers.UpSampling2D(size=size)(inputs)
    return up

def feature_extraction_block(inputs, model_width, feature_number):
    shape = inputs.shape
    latent = layers.Flatten()(inputs)
    latent = layers.Dense(feature_number, name="features")(latent)
    latent = layers.Dense(model_width * shape[1] * shape[2])(latent)
    latent = layers.Reshape((shape[1], shape[2], model_width))(latent)
    return latent

def attention_block(skip_connection, gating_signal, num_filters, multiplier):
    conv1x1_1 = layers.Conv2D(num_filters * multiplier, (1, 1), strides=(2, 2))(skip_connection)
    conv1x1_1 = layers.BatchNormalization()(conv1x1_1)
    conv1x1_2 = layers.Conv2D(num_filters * multiplier, (1, 1), strides=(1, 1))(gating_signal)
    conv1x1_2 = layers.BatchNormalization()(conv1x1_2)
    conv1_2 = layers.add([conv1x1_1, conv1x1_2])
    conv1_2 = layers.Activation('relu')(conv1_2)
    conv1_2 = layers.Conv2D(1, (1, 1), strides=(1, 1))(conv1_2)
    conv1_2 = layers.BatchNormalization()(conv1_2)
    conv1_2 = layers.Activation('sigmoid')(conv1_2)
    resampler1 = up_conv_block(conv1_2)
    resampler2 = trans_conv(conv1_2, 1, 1)
    resampler = layers.add([resampler1, resampler2])
    out = skip_connection * resampler
    return out

class UNet:
    def __init__(self, length, width, model_depth, num_channel, model_width, kernel_size, problem_type='Regression',
                 output_nums=1, ds=0, ae=0, ag=0, lstm=0, feature_number=1024, is_transconv=True):
        # length: Input Signal Length
        # width: Input Image Width (y-dim) [Normally same as the x-dim i.e., Square shape]
        # model_depth: Depth of the Model
        # model_width: Width of the Input Layer of the Model
        # num_channel: Number of Channels allowed by the Model
        # kernel_size: Kernel or Filter Size of the Convolutional Layers
        # problem_type: Classification (Binary or Multiclass) or Regression
        # output_nums: Output Classes (Classification Mode) or Features (Regression Mode)
        # ds: Checks where Deep Supervision is active or not, either 0 or 1 [Default value set as 0]
        # ae: Enables or diables the AutoEncoder Mode, either 0 or 1 [Default value set as 0]
        # ag: Checks where Attention Guided is active or not, either 0 or 1 [Default value set as 0]
        # lstm: Checks where Bidirectional LSTM is active or not, either 0 or 1 [Default value set as 0]
        # feature_number: Number of Features or Embeddings to be extracted from the AutoEncoder in the A_E Mode
        # is_transconv: (TRUE - Transposed Convolution, FALSE - UpSampling) in the Encoder Layer
        self.length = length
        self.width = width
        self.model_depth = model_depth
        self.num_channel = num_channel
        self.model_width = model_width
        self.kernel_size = kernel_size
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.D_S = ds
        self.A_E = ae
        self.A_G = ag
        self.LSTM = lstm
        self.feature_number = feature_number
        self.is_transconv = is_transconv

    def UNet(self):
        # Variable UNet Model Design
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []

        # Encoding
        inputs = Input((self.length, self.width, self.num_channel))
        pool = inputs

        for i in range(1, (self.model_depth + 1)):
            conv = conv_block(pool, self.model_width, self.kernel_size, 2 ** (i - 1))
            conv = conv_block(conv, self.model_width, self.kernel_size, 2 ** (i - 1))
            pool = layers.MaxPooling2D(pool_size=(2, 2))(conv)
            convs["conv%s" % i] = conv

        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            pool = feature_extraction_block(pool, self.model_width, self.feature_number)

        conv = conv_block(pool, self.model_width, self.kernel_size, 2 ** self.model_depth)
        conv = conv_block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)

        # Decoding
        deconv = conv
        convs_list = list(convs.values())

        for j in range(0, self.model_depth):
            skip_connection = convs_list[self.model_depth - j - 1]
            if self.A_G == 1:
                skip_connection = attention_block(convs_list[self.model_depth - j - 1], deconv, self.model_width, 2 ** (self.model_depth - j - 1))
            if self.D_S == 1:
                # For Deep Supervision
                level = layers.Conv2D(1, (1, 1), name=f'level{self.model_depth - j}')(deconv)
                levels.append(level)
            if self.is_transconv:
                deconv = trans_conv(deconv, self.model_width, 2 ** (self.model_depth - j - 1))
            elif not self.is_transconv:
                deconv = up_conv_block(deconv)
            if self.LSTM == 1:
                x1 = layers.Reshape(target_shape=(1, np.int32(self.length / (2 ** (self.model_depth - j - 1))), np.int32(self.width / (2 ** (self.model_depth - j - 1))), np.int32(self.model_width * (2 ** (self.model_depth - j - 1)))))(skip_connection)
                x2 = layers.Reshape(target_shape=(1, np.int32(self.length / (2 ** (self.model_depth - j - 1))), np.int32(self.width / (2 ** (self.model_depth - j - 1))), np.int32(self.model_width * (2 ** (self.model_depth - j - 1)))))(deconv)
                merge = layers.concatenate([x1, x2], axis=-1)
                deconv = layers.ConvLSTM2D(filters=np.int32(self.model_width * (2 ** (self.model_depth - j - 2))),
                                           kernel_size=(3, 3),
                                           padding='same',
                                           return_sequences=False,
                                           go_backwards=True,
                                           kernel_initializer='he_normal')(merge)
            elif self.LSTM == 0:
                deconv = concat_block(deconv, skip_connection)
            deconv = conv_block(deconv, self.model_width, self.kernel_size, 2 ** (self.model_depth - j - 1))
            deconv = conv_block(deconv, self.model_width, self.kernel_size, 2 ** (self.model_depth - j - 1))

        # Output
        outputs = []

        if self.problem_type == 'Classification':
            outputs = layers.Conv2D(self.output_nums, (1, 1), activation='softmax', name="out")(deconv)
        elif self.problem_type == 'Regression':
            outputs = layers.Conv2D(self.output_nums, (1, 1), activation='linear', name="out")(deconv)

        model = Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = Model(inputs=[inputs], outputs=levels)

        return model

if __name__ == '__main__':
    # Configurations
    length = 224  # Length of the Image (2D Signal)
    width = 224  # Width of the Image
    model_name = 'UNet'  # Name of the Segmentation Model
    model_depth = 5  # Number of Levels in the Segmentation Model
    model_width = 64  # Width of the Initial Layer, subsequent layers depend on this
    kernel_size = 3  # Size of the Kernels/Filter
    num_channel = 1  # Number of Channels in the Model
    D_S = 1  # Turn on Deep Supervision
    A_E = 0  # Turn on AutoEncoder Mode for Feature Extraction
    A_G = 1  # Turn on for Guided Attention
    LSTM = 1  # Turn on for BiConvLSTM
    problem_type = 'Regression'  # Problem Type: Classification or Regression
    output_nums = 1  # Number of Class for Classification Problems, always '1' for Regression Problems
    is_transconv = True  # True: Transposed Convolution, False: UpSampling
    '''Only required if the AutoEncoder Mode is turned on'''
    feature_number = 1024  # Number of Features to be Extracted
    #
    Model = UNet(length, width, model_depth, num_channel, model_width, kernel_size, problem_type=problem_type, output_nums=output_nums,
                 ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, is_transconv=is_transconv).UNet()
    Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
                  loss=tf.keras.losses.MeanAbsoluteError(),
                  metrics=tf.keras.metrics.MeanSquaredError())
    Model.summary()


