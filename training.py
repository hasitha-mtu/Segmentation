import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import array_to_img
from datetime import datetime
from model import get_model1
from preprocessing import img_size, get_train_and_validation_data, display_mask


def load_with_trained_model():
    _, (val_input_imgs, _) = get_train_and_validation_data()
    model = keras.models.load_model("oxford_segmentation.keras")
    i = 4
    test_image = val_input_imgs[i]
    plt.axis("off")
    plt.imshow(array_to_img(test_image))
    mask = model.predict(np.expand_dims(test_image, 0))[0]
    display_mask(mask)
    return None

def train_model():
    (train_input_images, train_targets), (val_input_imgs, val_targets) = get_train_and_validation_data()
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="oxford_segmentation.keras",
            save_best_only=True
        ),
        keras.callbacks.TensorBoard(
            log_dir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"),
            histogram_freq=1
        )
    ]
    model = get_model1(img_size=img_size, num_classes=3)
    print(f"Model information: {model.summary()}")
    history = model.fit(
        train_input_images,
        train_targets,
        epochs=50,
        batch_size=16,
        callbacks=callbacks,
        validation_data=(val_input_imgs, val_targets)
    )

    epochs = range(1, len(history.history["loss"])+1)
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()
    return None

if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        train_model()

# if __name__ == "__main__":
#     print(tf.config.list_physical_devices('GPU'))
#     physical_devices = tf.config.experimental.list_physical_devices('GPU')
#     if len(physical_devices) > 0:
#         load_with_trained_model()
