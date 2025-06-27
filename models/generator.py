import cv2
import numpy as np
from keras.utils import load_img, img_to_array
import os
from glob import glob
from tqdm import tqdm

def generate_supervised_mask(image_path, mask_path, supervised_mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path)

    # === Step 1: Identify all labeled pixels ===
    # mask > 0 => labeled (water), mask == 0 => non-water
    supervised_base = (mask >= 0).astype(np.uint8)  # initially assume all are labeled

    # === Step 2: Remove heavily occluded/dark regions ===
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = hsv[:, :, 2]
    shadow_mask = (brightness < 50).astype(np.uint8)  # tweak threshold as needed

    # === Step 3: Combine — keep only bright & labeled pixels ===
    final_supervised = supervised_base.copy()
    final_supervised[shadow_mask == 1] = 0  # mask out shadowed areas

    # === Save output ===
    cv2.imwrite(supervised_mask_path, final_supervised * 255)

def create_partial_supervision_masks(mask_path, partial_sup_mask_path):
    gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Mark pixels as supervised if they are either 0 or >0
    # If some pixels are undefined (e.g., value stays at 0 but is meant to be void), add logic here
    # mask = ((gt >= 0) & (gt <= 255)).astype(np.uint8)  # this is equivalent to full supervision unless you mask occlusion
    mask = (gt >0).astype(np.uint8)

    cv2.imwrite(partial_sup_mask_path, mask * 255)


def generator(path):
    total_images = len(os.listdir(path))
    print(f'total number of images in path is {total_images}')
    all_paths = sorted(glob(path + "/*.jpg" ))
    for i, path in tqdm(enumerate(all_paths), total=len(all_paths), desc="Loading"):
        image_path = path
        mask_path = image_path.replace("images", "annotations")
        mask_path = mask_path.replace(".jpg", ".png")
        # supervised_mask_path = mask_path.replace("annotations", "supervised_masks")
        partial_supervised_mask_path = mask_path.replace("annotations", "partial_supervision_masks")
        create_partial_supervision_masks(mask_path, partial_supervised_mask_path)
        # generate_supervised_mask(image_path, mask_path, supervised_mask_path)

def load_images(path):
    total_images = len(os.listdir(path))
    print(f'total number of images in path is {total_images}')
    all_paths = sorted(glob(path + "/*.jpg" ))
    images = []
    masks_true = []
    masks_supervised = []

    for i, path in tqdm(enumerate(all_paths), total=len(all_paths), desc="Loading"):
        image_path = path
        mask_path = image_path.replace("images", "annotations")
        mask_path = mask_path.replace(".jpg", ".png")
        partial_supervised_mask_path = mask_path.replace("annotations", "partial_supervision_masks")
        mask_supervised = cv2.imread(partial_supervised_mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(image_path)
        images.append(image)
        masks_true.append(mask)
        masks_supervised.append(mask_supervised)

    return images, masks_true, masks_supervised

if __name__=="__main__":
    # === Load files ===
    images_path = '../input/samples/segnet_512/images'
    generator(images_path)




