import time
import os
import matplotlib
matplotlib.use('Agg') # or 'SVG', 'PDF', 'PS' for vector graphics
import matplotlib.pyplot as plt
from models.common_utils.config import ModelConfig

def plot_model_history(history, path):
    print(f'plot_model_history|history:{history.history}')
    print(f'plot_model_history|path:{path}')
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
    plt.savefig(os.path.join(path, 'training_and_validation_accuracy.png'))
    if ModelConfig.PLOT_DISPLAY:
        plt.show(block=False)
        time.sleep(5)
        plt.close()

    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.savefig(os.path.join(path, 'training_and_validation_loss.png'))
    if ModelConfig.PLOT_DISPLAY:
        plt.show(block=False)
        time.sleep(5)
        plt.close()

def plot_prediction(i, rgb_image, actual_mask, pred_mask, dir_path):
    plt.figure(figsize=(10, 8))

    plt.subplot(1, 3, 1)
    show_image(dir_path, rgb_image, index=i, title="Original_Image", save=True)
    plt.subplot(1, 3, 2)
    show_image(dir_path, actual_mask.squeeze(), index=i, title="Actual_Mask", save=True)
    plt.subplot(1, 3, 3)
    show_image(dir_path, pred_mask.squeeze(), index=i, title="Predicted_Mask", save=True)
    if ModelConfig.PLOT_DISPLAY:
        plt.tight_layout()
        plt.show(block=False)
        time.sleep(10)
        plt.close()

def show_image(dir_path, image, index, title=None, save=False):
    if save:
        if title:
            file_name = f"{dir_path}/{title}_{index}.png"
        else:
            file_name = f"{dir_path}/predicted_mask_{index}.png"
        os.makedirs(dir_path, exist_ok=True)
        plt.imsave(file_name, image)
    if ModelConfig.PLOT_DISPLAY:
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
