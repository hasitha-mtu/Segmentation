import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from model import get_model
from preprocessing import img_size, get_train_and_validation_data


def train_model():
    (train_input_images, train_targets), (val_input_imgs, val_targets) = get_train_and_validation_data()
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="oxford_segmentation.keras",
            save_best_only=True
        )
    ]
    model = get_model(img_size=img_size, num_classes=3)
    print(f"Model information: {model.summary()}")
    history = model.fit(
        train_input_images,
        train_targets,
        epochs=50,
        batch_size=64,
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
    return None

if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        train_model()