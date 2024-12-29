import numpy as np
from glob import glob
from keras.utils import load_img, img_to_array
import tensorflow as tf
from tqdm import tqdm
import random
import os

def load_drone_dataset(path, file_extension = "jpg"):
    total_images = len(os.listdir(path))
    print(f'total number of images in path is {total_images}')
    all_image_paths = sorted(glob(path + "/*."+file_extension))
    random.Random(1337).shuffle(all_image_paths)
    print(all_image_paths)
    train_paths = all_image_paths[:300]
    print(f"train image count : {len(train_paths)}")
    x_train, y_train = load_drone_images(train_paths)
    test_paths = all_image_paths[300:]
    print(f"test image count : {len(test_paths)}")
    x_test, y_test = load_drone_images(test_paths)
    return (x_train, y_train),(x_test, y_test)

def load_drone_images(paths):
    images = np.zeros(shape=(len(paths), 256, 256, 3))
    masks = np.zeros(shape=(len(paths), 256, 256, 3))
    for i, path in tqdm(enumerate(paths), total=len(paths), desc="Loading"):
        image = load_image(path)
        images[i] = image
        mask_path = path.replace("images", "annotations")
        mask_path = mask_path.replace("jpg", "png")
        mask = load_image(mask_path)
        masks[i] = mask
    return images, masks

def load_image(path: str):
    img = load_img(path)
    img_array = img_to_array(img)
    normalized_img_array = img_array/255.
    resized_img = tf.image.resize(normalized_img_array, (256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return resized_img

if __name__ == "__main__":
    print('load image for testing')
    (x_train, y_train),(x_test, y_test) = load_drone_dataset("../../input/drone/images")
    print(f"train data shape: {x_train.shape}")
    print(f"test data shape: {x_test.shape}")

