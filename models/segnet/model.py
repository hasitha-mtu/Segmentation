from models.common_utils.loss_functions import  recall_m, precision_m, f1_score
from models.segnet.loss_function import combined_masked_dice_bce_loss
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from tensorflow.keras.models import Model
import keras
import os
from models.memory_usage import estimate_model_memory_usage
from models.common_utils.config import load_config, ModelConfig

def SegNet(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)

    # Encoder
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)  # size reduced to 128x128

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)  # size reduced to 64x64

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)  # size reduced to 32x32

    # Decoder
    x = UpSampling2D(size=(2, 2))(x)  # 32x32 → 64x64
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(size=(2, 2))(x)  # 64x64 → 128x128
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(size=(2, 2))(x)  # 128x128 → 256x256
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Final layer
    x = Conv2D(ModelConfig.MODEL_OUTPUT_CHANNELS, (1, 1), padding='valid')(x)
    outputs = Activation('sigmoid')(x)  # Use 'sigmoid' if binary segmentation

    model = Model(inputs=inputs,
                  outputs=outputs,
                  name=ModelConfig.MODEL_NAME)
    print("Model output shape:", model.output_shape)
    model.summary()

    estimate_model_memory_usage(model, batch_size=ModelConfig.BATCH_SIZE)

    keras.utils.plot_model(model, os.path.join(ModelConfig.MODEL_DIR, "SegNet_model.png"), show_shapes=True)

    return model

def segnet(width, height, input_channels):
    input_shape = (width, height, input_channels)
    model = SegNet(input_shape)
    model.compile(
        optimizer=ModelConfig.TRAINING_OPTIMIZER,
        loss=combined_masked_dice_bce_loss,
        metrics=['accuracy', f1_score, precision_m, recall_m]
    )
    return model

if __name__=='__main__':
    config_file = 'config.yaml'
    load_config(config_file)
    segnet(512, 512, 3)