import numpy as np
from glob import glob
from keras.utils import load_img, img_to_array
import tensorflow as tf
from tqdm import tqdm
import random
import os
import cv2


from models.common_utils.images import load_ndwi_edge_map, selected_channels, format_image, create_confidence_mask

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

def load_drone_images(size, paths, channels):
    (width, height) = size
    images = np.zeros(shape=(len(paths), width, height, len(channels)))
    masks = np.zeros(shape=(len(paths), width, height, 1))
    for i, path in tqdm(enumerate(paths), total=len(paths), desc="Loading"):
        image = get_stacked_image(channels, size, path)
        images[i] = image
        mask_path = path.replace("images", "masks")
        mask_path = mask_path.replace(".jpg", ".png")
        mask = load_image(size, mask_path, color_mode = "grayscale")
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
                 channels=['RED', 'GREEN', 'BLUE', 'NDWI', 'Canny', 'LBP',
                           'HSV Saturation', 'HSV Value', 'GradMag', 'Shadow Mask'],
                 percentage=0.7):
    total_images = len(os.listdir(path))
    print(f'total number of images in path is {total_images}')
    all_image_paths = sorted(glob(path + "/*."+file_extension))
    random.Random(1337).shuffle(all_image_paths)
    print(all_image_paths)
    train_size = int(len(all_image_paths) * percentage)
    train_paths = all_image_paths[:train_size]
    print(f"train image count : {len(train_paths)}")
    x_train, y_train = load_drone_images(size, train_paths, channels=channels)
    test_paths = all_image_paths[train_size:]
    print(f"test image count : {len(test_paths)}")
    x_test, y_test = load_drone_images(size, test_paths, channels=channels)
    return (x_train, y_train),(x_test, y_test)




if __name__ == "__main__":
    annotation_dir = "../../input/samples/crookstown/annotations"
    formatted_annotation_dir = "../../input/samples/segnet/annotations"
    os.makedirs(formatted_annotation_dir, exist_ok=True)
    for filename in os.listdir(annotation_dir):
        path = os.path.join(annotation_dir, filename)
        print(f"Original mask path: {path}")
        ann = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # shape: (H, W)

        mask = create_confidence_mask(ann)

        # Save mask as 8-bit single-channel image
        updated_mask_path = os.path.join(formatted_annotation_dir, filename)
        print(f"Updated mask path: {updated_mask_path}")
        cv2.imwrite(updated_mask_path, (mask * 255).astype(np.uint8))


# if __name__ == "__main__":
#     sample_image = "../../input/samples1/crookstown/images/DJI_20250324094536_0001_V.jpg"
#     size = (256, 256)
#     # get_image(size, sample_image)
#     img = load_image(size, sample_image)
#     print(f'Image shape: {img.shape}')
#
#     sample_mask = "../../input/samples1/crookstown/masks/DJI_20250324094536_0001_V.png"
#     size = (256, 256)
#     # get_image(size, sample_image)
#     mask = load_image(size, sample_mask, color_mode = "grayscale")
#     print(f'Mask shape: {mask.shape}')
#     np.set_printoptions(threshold=sys.maxsize)
#     print(f'Mask : {mask}')
#     array_2d = np.squeeze(mask)
#     # Save to text file
#     np.savetxt("y_true.txt", array_2d, fmt="%.4f")
#     generated_mask = (array_2d != 0.0)
#     np.savetxt("mask.txt", generated_mask, fmt="%.4f")
#     y_pred_bin = (generated_mask >= 0.5).astype(np.uint8)
#     np.savetxt("bin_mask.txt", y_pred_bin, fmt="%.4f")

# if __name__ == "__main__":
#     sample_image = "../../input/samples1/test/images/0001TP_006690.png"
#     size = (256, 256)
#     # get_image(size, sample_image)
#     img = load_image(size, sample_image)
#     print(f'Image shape: {img.shape}')
#
#     sample_mask = "../../input/samples1/test/masks/0001TP_006690.png"
#     size = (256, 256)
#     # get_image(size, sample_image)
#     mask = load_image(size, sample_mask, color_mode = "grayscale")
#     print(f'Mask shape: {mask.shape}')
#     np.set_printoptions(threshold=sys.maxsize)
#     print(f'Mask : {mask}')
#     array_2d = np.squeeze(mask)
#     # Save to text file
#     np.savetxt("y_true.txt", array_2d, fmt="%.4f")
#     generated_mask = (array_2d != 0.0)
#     np.savetxt("mask.txt", generated_mask, fmt="%.4f")
#     y_pred_bin = (generated_mask >= 0.5).astype(np.uint8)
#     np.savetxt("bin_mask.txt", y_pred_bin, fmt="%.4f")
