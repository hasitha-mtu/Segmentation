import numpy as np
from glob import glob
from keras.utils import load_img, img_to_array
import tensorflow as tf
from tqdm import tqdm
import random
import os
import matplotlib.pyplot as plt
import sys
from models.common_utils.images import load_ndwi_edge_map, selected_channels, format_image


def load_drone_images(size, paths, channels):
    (width, height) = size
    images = np.zeros(shape=(len(paths), width, height, len(channels)))
    masks = np.zeros(shape=(len(paths), width, height, 1))
    for i, path in tqdm(enumerate(paths), total=len(paths), desc="Loading"):
        image = get_stacked_image(channels, size, path)
        images[i] = image
        mask_path = path.replace("images", "masks")
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
                 percentage=0.7,
                 image_count=50):
    print("Loading data for UNET-ResNet50 Model")
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
    train_size = int(len(selected_paths) * percentage)
    train_paths = selected_paths[:train_size]
    print(f"train image count : {len(train_paths)}")
    x_train, y_train = load_drone_images(size, train_paths, channels=channels)
    print(f"load_dataset|y_train shape : {y_train.shape}")
    test_paths = selected_paths[train_size:]
    print(f"test image count : {len(test_paths)}")
    x_test, y_test = load_drone_images(size, test_paths, channels=channels)
    print(f"load_dataset|y_test shape : {y_test.shape}")
    return (x_train, y_train),(x_test, y_test)

if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    sample_image = "../../input/samples/segnet_512/images/DJI_20250324092908_0001_V.jpg"
    size = (512, 512)
    # get_image(size, sample_image)
    img = load_image(size, sample_image)
    print(f'Image shape: {img.shape}')

    sample_mask = "../../input/samples/segnet_512/masks/DJI_20250324092908_0001_V.png"
    size = (512, 512)
    # get_image(size, sample_image)
    mask = load_image(size, sample_mask, color_mode='grayscale')
    print(f'Mask shape: {mask.shape}')
    print(f'Mask: {mask}')
    np.set_printoptions(threshold=1000)

    plt.subplot(1, 2, 1)
    plt.imshow(img[:, :, :3])  # RGB channels
    plt.title('Input Image')

    plt.subplot(1, 2, 2)
    plt.imshow(mask[:, :, 0], cmap='gray')
    plt.title('Ground Truth Mask')

    plt.show()

