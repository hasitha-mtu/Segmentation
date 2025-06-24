from tensorflow.keras import layers, models
import keras
from models.unet_plus_plus.loss_functions import BCEDiceLoss
from models.common_utils.loss_functions import  recall_m, precision_m, f1_score
from models.memory_usage import estimate_model_memory_usage

def conv_block(x, filters, kernel_size=(3,3), activation='relu', padding='same'):
    x = layers.Conv2D(filters, kernel_size, activation=activation, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, kernel_size, activation=activation, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    return x

def build_model(batch_size, input_shape=(512, 512, 3), num_classes=1):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x00 = conv_block(inputs, 64)
    p0 = layers.MaxPooling2D((2, 2))(x00)

    x10 = conv_block(p0, 128)
    p1 = layers.MaxPooling2D((2, 2))(x10)

    x20 = conv_block(p1, 256)
    p2 = layers.MaxPooling2D((2, 2))(x20)

    x30 = conv_block(p2, 512)
    p3 = layers.MaxPooling2D((2, 2))(x30)

    x40 = conv_block(p3, 1024)

    # Decoder (Nested Dense Skip Connections)
    x01 = conv_block(layers.concatenate([x00, layers.UpSampling2D((2, 2))(x10)]), 64)
    x11 = conv_block(layers.concatenate([x10, layers.UpSampling2D((2, 2))(x20)]), 128)
    x21 = conv_block(layers.concatenate([x20, layers.UpSampling2D((2, 2))(x30)]), 256)
    x31 = conv_block(layers.concatenate([x30, layers.UpSampling2D((2, 2))(x40)]), 512)

    x02 = conv_block(layers.concatenate([x00, x01, layers.UpSampling2D((2, 2))(x11)]), 64)
    x12 = conv_block(layers.concatenate([x10, x11, layers.UpSampling2D((2, 2))(x21)]), 128)
    x22 = conv_block(layers.concatenate([x20, x21, layers.UpSampling2D((2, 2))(x31)]), 256)

    x03 = conv_block(layers.concatenate([x00, x01, x02, layers.UpSampling2D((2, 2))(x12)]), 64)
    x13 = conv_block(layers.concatenate([x10, x11, x12, layers.UpSampling2D((2, 2))(x22)]), 128)

    x04 = conv_block(layers.concatenate([x00, x01, x02, x03, layers.UpSampling2D((2, 2))(x13)]), 64)

    # Final output (from the last node in the topmost dense path)
    if num_classes == 1:
        output = layers.Conv2D(1, (1, 1), activation='sigmoid')(x04)
    else:
        output = layers.Conv2D(num_classes, (1, 1), activation='softmax')(x04)

    model = models.Model(inputs, output)

    loss_fn = BCEDiceLoss(global_batch_size=batch_size)

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy', f1_score, precision_m, recall_m])

    print("Model output shape:", model.output_shape)

    print(f"Model summary : {model.summary()}")

    estimate_model_memory_usage(model, batch_size=4)

    keras.utils.plot_model(model, "UNET++_model.png", show_shapes=True)

    return model


def unet_plus_plus(width, height, num_channels, batch_size=4):
    return build_model(batch_size, input_shape=(width, height, num_channels))


if __name__ == '__main__':
    build_model(4, input_shape=(512, 512, 16), num_classes=1)
