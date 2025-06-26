import numpy as np
from glob import glob
from keras.utils import load_img, img_to_array
import tensorflow as tf
from tqdm import tqdm
import random
import os
from models.common_utils.images import load_ndwi_edge_map, selected_channels, format_image


def load_drone_images(size, paths, channels):
    (width, height) = size
    images = np.zeros(shape=(len(paths), width, height, len(channels)))
    masks = np.zeros(shape=(len(paths), width, height, 1))
    for i, path in tqdm(enumerate(paths), total=len(paths), desc="Loading"):
        image = get_stacked_image(channels, size, path)
        images[i] = image
        mask_path = path.replace("images", "annotations")
        mask_path = mask_path.replace(".jpg", ".png")
        mask = load_image(size, mask_path, color_mode='grayscale')
        masks[i] = mask
    return images, masks

def load_image(size, path: str, color_mode = "rgb"):
    img = load_img(path, color_mode=color_mode)
    img_array = img_to_array(img)
    normalized_img_array = img_array/255.
    resized_img = tf.image.resize(normalized_img_array, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return resized_img

def get_image(size, path):
    image_data = load_ndwi_edge_map(path, edge_map_type='sobel')
    rgb = format_image(size, image_data['rgb'])
    # print(f'RGB shape: {rgb.shape}')
    ndwi = format_image(size, image_data['ndwi'])
    # print(f'NDWI shape: {ndwi.shape}')
    edges = format_image(size, image_data['edge_map'])
    # print(f'EDGE map shape: {edges.shape}')
    stacked_image = np.dstack((rgb, ndwi, edges))
    # print(f'Stacked image shape: {stacked_image.shape}')
    return stacked_image

def get_stacked_image(channels, size, path):
    stacked_image = selected_channels(channels, size, path)
    return stacked_image

def load_dataset(path, size = (256, 256), file_extension = "JPG",
                 channels=None,
                 image_count=50):
    if channels is None:
        channels = ['RED', 'GREEN', 'BLUE', 'NDWI', 'Canny', 'LBP',
                    'HSV Saturation', 'HSV Value', 'GradMag', 'Shadow Mask']
    total_images = len(os.listdir(path))
    print(f'total number of images in path is {total_images}')
    all_image_paths = sorted(glob(path + "/*."+file_extension))
    random.Random(1337).shuffle(all_image_paths)
    print(all_image_paths)

    if len(all_image_paths) > image_count:
        selected_paths = random.sample(all_image_paths, image_count)
    else:
        selected_paths = all_image_paths
    print(f'Selected number of images in path is {selected_paths}')
    images, masks = load_drone_images(size, selected_paths, channels=channels)
    print(f"load_dataset|images shape : {images.shape}")
    print(f"load_dataset|masks shape : {masks.shape}")
    return (images, masks)

if __name__ == "__main__":
    (images, masks) = load_dataset("../../input/samples/segnet_512/images",
                                                      size=(512, 512),
                                                      file_extension="jpg",
                                                      channels=['RED', 'GREEN', 'BLUE'],
                                                      image_count=10)
