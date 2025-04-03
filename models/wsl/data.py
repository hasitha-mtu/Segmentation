import numpy as np
from glob import glob
from keras.utils import load_img, img_to_array
import tensorflow as tf
from tqdm import tqdm
import random
import os

from models.common_utils.images import load_ndwi_edge_map

def load_drone_dataset(path, file_extension = "jpg", num_channels=5):
    total_images = len(os.listdir(path))
    print(f'total number of images in path is {total_images}')
    all_image_paths = sorted(glob(path + "/*."+file_extension))
    random.Random(1337).shuffle(all_image_paths)
    print(all_image_paths)
    train_paths = all_image_paths[:300]
    print(f"train image count : {len(train_paths)}")
    x_train, y_train = load_drone_images(train_paths, channels=num_channels)
    test_paths = all_image_paths[300:]
    print(f"test image count : {len(test_paths)}")
    x_test, y_test = load_drone_images(test_paths, channels=num_channels)
    return (x_train, y_train),(x_test, y_test)

def load_drone_images(paths, channels=3):
    images = np.zeros(shape=(len(paths), 256, 256, channels))
    masks = np.zeros(shape=(len(paths), 256, 256, 3))
    for i, path in tqdm(enumerate(paths), total=len(paths), desc="Loading"):
        image = get_image(path)
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

def format_image(img):
    img_array = img_to_array(img)
    normalized_img_array = img_array/255.
    resized_img = tf.image.resize(normalized_img_array, (256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return resized_img

def get_image(path):
    image_data = load_ndwi_edge_map(path)
    rgb = format_image(image_data['rgb'])
    # print(f'RGB shape: {rgb.shape}')
    ndwi = format_image(image_data['ndwi'])
    # print(f'NDWI shape: {ndwi.shape}')
    edges = format_image(image_data['edge_map'])
    # print(f'EDGE map shape: {edges.shape}')
    stacked_image = np.dstack((rgb, ndwi, edges))
    # print(f'Stacked image shape: {stacked_image.shape}')
    return stacked_image

if __name__ == "__main__":
    sample_image = "../../data/samples/sample1.JPG"
    get_image(sample_image)
    img = load_image(sample_image)
    print(f'Image shape: {img.shape}')

