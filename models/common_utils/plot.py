import matplotlib.pyplot as plt
import time
from models.common_utils.images import show_image

def plot_model_history(history):
    print(f'plot_model_history|history:{history.history}')
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(accuracy) + 1)
    plt.figure()
    plt.plot(epochs, accuracy, "bo", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.show(block=False)
    time.sleep(3)
    plt.close()

    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show(block=False)
    time.sleep(3)
    plt.close()

def plot_prediction(i, rgb_image, actual_mask, pred_mask, dir_path):
    plt.figure(figsize=(10, 8))

    plt.subplot(1, 3, 1)
    show_image(dir_path, rgb_image, index=i, title="Original_Image", save=True)
    plt.subplot(1, 3, 2)
    show_image(dir_path, actual_mask.squeeze(), index=i, title="Actual_Mask", save=True)
    plt.subplot(1, 3, 3)
    show_image(dir_path, pred_mask.squeeze(), index=i, title="Predicted_Mask", save=True)

    plt.tight_layout()
    plt.show(block=False)
    time.sleep(5)
    plt.close()

