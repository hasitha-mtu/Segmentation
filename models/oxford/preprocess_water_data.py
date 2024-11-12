import os

import numpy as np
from keras.utils import load_img, img_to_array
from tensorflow.image import resize
from glob import glob
from tqdm import tqdm

image_path = "../../input/water_v1/JPEGImages/ADE20K"
img_size = (256, 256)

def load_image(path: str):
    img = load_img(path)
    img_array = img_to_array(img)
    normalized_img_array = img_array/255.
    resized_img = resize(normalized_img_array, (256, 256))
    return resized_img

def load_data():
    total_images = len(os.listdir(image_path))
    print(f'total number of images in path is {total_images}')
    all_image_paths = sorted(glob(image_path+"/*.png"))
    train_paths = all_image_paths[:1000]
    x_train, y_train = load_images(train_paths)
    test_paths = all_image_paths[1000:]
    x_test, y_test = load_images(test_paths)
    print('all images and masks loaded')
    print(f"Train image count : {len(x_train)}")
    print(f"Test image count : {len(x_test)}")
    return (x_train, y_train),(x_test, y_test)

def load_images(paths):
    images = np.zeros(shape=(len(paths), 256, 256, 3))
    masks = np.zeros(shape=(len(paths), 256, 256, 3))
    for i, path in tqdm(enumerate(paths), total=len(paths), desc="Loading"):
        image = load_image(path)
        images[i] = image
        mask_path = path.replace("JPEGImages", "Annotations")
        mask = load_image(mask_path)
        masks[i] = mask
    return images, masks

if __name__ == "__main__":
    print('load image for testing')
    load_data()