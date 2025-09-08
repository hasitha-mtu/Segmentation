
from tensorflow.keras.layers import (Conv2D, BatchNormalization, Activation, UpSampling2D,
                                     AveragePooling2D, Concatenate, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
import keras
import os
from models.common_utils.loss_functions import  recall_m, precision_m, f1_score, combined_loss_function
# from models.deeplabv3_plus.loss_function import combined_masked_dice_bce_loss
from models.memory_usage import estimate_model_memory_usage
from models.common_utils.config import load_config, ModelConfig
from models.common_utils.model_utils import get_optimizer, estimate_flops

def ASPP(inputs):
    shape = inputs.shape
    y_pool = AveragePooling2D(pool_size=(shape[1], shape[2]))(inputs)
    print(f'AveragePooling2D y_pool shape:{y_pool.shape}')
    y_pool = Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(y_pool)
    print(f'Conv2D y_pool shape:{y_pool.shape}')
    y_pool = BatchNormalization()(y_pool)
    y_pool = Activation('relu')(y_pool)
    y_pool = UpSampling2D((shape[1], shape[2]), interpolation='bilinear')(y_pool)
    print(f'UpSampling2D y_pool shape:{y_pool.shape}')

    y_1 = Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(inputs)
    print(f'Conv2D y_1 shape:{y_1.shape}')
    y_1 = BatchNormalization()(y_1)
    y_1 = Activation('relu')(y_1)

    y_6 = Conv2D(filters=256, kernel_size=1, dilation_rate=6, padding='same', use_bias=False)(inputs)
    print(f'Conv2D y_6 shape:{y_6.shape}')
    y_6 = BatchNormalization()(y_6)
    y_6 = Activation('relu')(y_6)

    y_12 = Conv2D(filters=256, kernel_size=1, dilation_rate=12, padding='same', use_bias=False)(inputs)
    print(f'Conv2D y_12 shape:{y_12.shape}')
    y_12 = BatchNormalization()(y_12)
    y_12 = Activation('relu')(y_12)

    y_18 = Conv2D(filters=256, kernel_size=1, dilation_rate=18, padding='same', use_bias=False)(inputs)
    print(f'Conv2D y_18 shape:{y_18.shape}')
    y_18 = BatchNormalization()(y_18)
    y_18 = Activation('relu')(y_18)

    y = Concatenate()([y_pool, y_1, y_6, y_12, y_18])
    print(f'Concatenate y shape:{y.shape}')

    y = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same', use_bias=False)(y)
    print(f'Conv2D y shape:{y.shape}')
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    return y

def DeepLabV3Plus(shape):
    inputs = Input(shape=shape)

    #  Pre-trained ResNet50
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)

    # Pre-trained ResNet50 output
    image_features = base_model.get_layer('conv4_block6_out').output
    print(f'image_features shape:{image_features.shape}')

    x_a = ASPP(image_features)
    print(f'x_a shape:{x_a.shape}')
    x_a = UpSampling2D((4, 4), interpolation='bilinear')(x_a)
    print(f'UpSampling2D x_a shape:{x_a.shape}')

    # Get low-level features
    x_b = base_model.get_layer('conv2_block2_out').output
    x_b = Conv2D(filters=48, kernel_size=1, padding='same', use_bias=False)(x_b)
    print(f'Conv2D x_b shape:{x_b.shape}')
    x_b = BatchNormalization()(x_b)
    x_b = Activation('relu')(x_b)

    x = Concatenate()([x_a, x_b])
    print(f'Concatenate x shape:{x.shape}')

    x = Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(x)
    print(f'Conv2D x shape:{x.shape}')
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(x)
    print(f'Conv2D x shape:{x.shape}')
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D((4, 4), interpolation='bilinear')(x)
    print(f'UpSampling2D x shape:{x.shape}')

    # Output
    outputs = Conv2D(ModelConfig.MODEL_OUTPUT_CHANNELS, (1, 1), name='output_layer')(x)
    outputs = Activation('sigmoid')(outputs)

    # Model
    model = Model(inputs=inputs,
                  outputs=outputs,
                  name=ModelConfig.MODEL_NAME)
    print("Model output shape:", model.output_shape)
    print(f"Model summary : {model.summary()}")

    estimate_model_memory_usage(model, batch_size=ModelConfig.BATCH_SIZE)

    keras.utils.plot_model(model, os.path.join(ModelConfig.MODEL_DIR, "DeepLabV3Plus_model.png"), show_shapes=True)


    print(f'Model type: {type(model)}')
    print(f'Model output shape: {model.output.shape}')

    estimate_flops(model)

    return model

def deeplab_v3_plus(width, height, input_channels):
    input_shape = (width, height, input_channels)
    model = DeepLabV3Plus(input_shape)
    model.compile(
        optimizer=get_optimizer(ModelConfig.TRAINING_LR),
        loss=combined_loss_function,
        metrics=['accuracy', f1_score, precision_m, recall_m]
    )
    return model

# if __name__ == '__main__':
#     config_file = 'config.yaml'
#     load_config(config_file)
#     input_shape = (ModelConfig.IMAGE_HEIGHT, ModelConfig.IMAGE_WIDTH, ModelConfig.MODEL_INPUT_CHANNELS)
#     DeepLabV3Plus(input_shape)


# ---- Debugging with one real batch ----
def debug_with_real_training_batch(dataset, model=None):
    # Take one batch from dataset
    for x_batch, y_batch in dataset.take(1):
        print("Batch shapes:", x_batch.shape, y_batch.shape)
        print("y_batch unique values:", np.unique(y_batch.numpy()))

        # Run model forward pass if provided
        if model is not None:
            y_pred = model(x_batch, training=False)
        else:
            # If no model yet, simulate predictions
            y_pred = tf.random.uniform(y_batch.shape, 0, 1)

        print("y_pred range:", float(tf.reduce_min(y_pred)), float(tf.reduce_max(y_pred)))

        # Compute loss
        loss_value = combined_loss_function(y_batch, y_pred)
        print("Loss value:", float(loss_value.numpy()))

def debug_with_real_validation_batch(dataset, model=None):
    # Take one batch from dataset
    problematic_batch_found = False
    for i, (x_batch, y_batch) in enumerate(dataset):
    # for x_batch, y_batch in dataset.take(1):
        try:
            print("Batch shapes:", x_batch.shape, y_batch.shape)
            print("y_batch unique values:", np.unique(y_batch.numpy()))

            # Run model forward pass if provided
            if model is not None:
                y_pred = model(x_batch, training=False)
            else:
                # If no model yet, simulate predictions
                y_pred = tf.random.uniform(y_batch.shape, 0, 1)

            print("y_pred range:", float(tf.reduce_min(y_pred)), float(tf.reduce_max(y_pred)))

            # Compute loss
            loss_value = combined_loss_function(y_batch, y_pred)
            print(f"Validation loss for single batch: {float(loss_value.numpy())}")
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            problematic_batch_found = True
            break
    if not problematic_batch_found:
        print("No NaN loss found in any batch. The issue might be intermittent or due to the combination of multiple batches.")

# ---- Example usage ----
from models.common_utils.dataset import load_datasets
import tensorflow as tf
import numpy as np
if __name__ == "__main__":
    # Example: if you already have a tf.data.Dataset for training
    config_file = 'config.yaml'
    load_config(config_file)
    train_dataset, validation_dataset = load_datasets(config_file, True)
    model = deeplab_v3_plus(512, 512, 3)

    # debug_with_real_training_batch(train_dataset, model)
    debug_with_real_validation_batch(train_dataset, model)