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
        mask_path = path.replace("images", "masks")
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

# if __name__ == "__main__":
#     print('load image for testing')
#     (x_train, y_train),(x_test, y_test) = load_drone_dataset("../../input/drone/images")
#     print(f"train data shape: {x_train.shape}")
#     print(f"test data shape: {x_test.shape}")

# if __name__ == '__main__':
#     path_single = "C:\\Users\\nimbus\PycharmProjects\Segmentation\input\landslide_data\\train\img\image_2000.h5"
#     path_single_mask = "C:\\Users\\nimbus\PycharmProjects\Segmentation\input\landslide_data\\train\mask\mask_2000.h5"
#     f_data = np.zeros((1, 128, 128, 3))
#     with h5py.File(path_single) as hdf:
#         ls = list(hdf.keys())
#         print("ls", ls)
#         data = np.array(hdf.get('img'))
#         print("input data shape:", data.shape)
#         plt.imshow(data[:, :, 3:0:-1])
#
#         data_red = data[:, :, 3]
#         data_green = data[:, :, 2]
#         data_blue = data[:, :, 1]
#         data_nir = data[:, :, 7]
#         data_rgb = data[:, :, 3:0:-1]
#         data_ndvi = np.divide(data_nir - data_red, np.add(data_nir, data_red))
#         f_data[0, :, :, 0] = data_ndvi
#         f_data[0, :, :, 1] = data[:, :, 12]
#         f_data[0, :, :, 2] = data[:, :, 13]
#
#         print("data ndvi shape ", data_ndvi.shape, "f_data shape: ", f_data.shape)
#         plt.imshow(data_ndvi)
#         plt.show()
#     with h5py.File(path_single_mask) as hdf:
#         ls = list(hdf.keys())
#         print("ls", ls)
#         data = np.array(hdf.get('mask'))
#         print("input data shape:", data.shape)
#         plt.imshow(data)
#         plt.show()