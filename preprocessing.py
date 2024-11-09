import os
import random

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

input_dir = "input/images/"
target_dir = "input/annotations/trimaps/"
img_size = (200, 200)

def display_target(target_array):
    normalized_array = (target_array.astype("uint8")-1) * 127
    plt.axis("off")
    plt.imshow(normalized_array[:, :, 0])
    plt.show()

def get_input_img_paths():
    input_img_paths = sorted([
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".jpg")
    ])
    random.Random(1337).shuffle(input_img_paths)
    return input_img_paths

def get_target_paths():
    target_paths = sorted([
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ])
    random.Random(1337).shuffle(target_paths)
    return target_paths

def path_to_input_image(path):
    return img_to_array(load_img(path, target_size=img_size))

def path_to_target(path):
    img = img_to_array(load_img(path, target_size=img_size, color_mode="grayscale"))
    img = img.astype("uint8") - 1
    return img

def get_train_and_validation_data():
    input_img_paths = get_input_img_paths()
    target_paths = get_target_paths()
    num_imgs = len(input_img_paths)
    input_imgs = np.zeros((num_imgs,) + img_size + (3,), dtype="float32")
    targets = np.zeros((num_imgs,) + img_size + (1,), dtype="uint8")
    for i in range(num_imgs):
        input_imgs[i] = path_to_input_image(input_img_paths[i])
        targets[i] = path_to_target(target_paths[i])

    num_val_samples = 1000
    train_input_images = input_imgs[:-num_val_samples]
    train_targets = targets[:-num_val_samples]
    val_input_imgs = input_imgs[-num_val_samples:]
    val_targets = targets[-num_val_samples:]

    return (train_input_images, train_targets), (val_input_imgs, val_targets)

if __name__ == "__main__":
    get_train_and_validation_data()



