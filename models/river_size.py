import numpy as np
import rasterio
from models.deeplabv3_plus.train_model import load_saved_model
from skimage import measure
from skimage.morphology import convex_hull_image, remove_small_objects
import cv2

# --------------------------
# Parameters
# --------------------------
WINDOW = 512  # size of training patches
STRIDE = 256  # overlap between windows
MIN_PATCH_SIZE = 100  # min connected pixels to keep as water
# --------------------------
# Load trained model
# --------------------------
def load_model(config_file):
    return load_saved_model(config_file, 'Adam', True)

# --------------------------
# Sliding-window prediction
# --------------------------
def sliding_window_predict(img, model, window=512, stride=256):
    H, W, C = img.shape
    output = np.zeros((H, W), dtype=np.float32)  # single-channel mask
    counts = np.zeros((H, W), dtype=np.float32)  # to average overlaps

    for y in range(0, H - window + 1, stride):
        for x in range(0, W - window + 1, stride):
            patch = img[y:y + window, x:x + window, :]
            patch_in = np.expand_dims(patch / 255.0, axis=0)  # normalize
            pred = model.predict(patch_in, verbose=0)[0, :, :, 0]  # assume binary mask

            output[y:y + window, x:x + window] += pred
            counts[y:y + window, x:x + window] += 1

    # Avoid division by zero
    counts[counts == 0] = 1
    output /= counts

    return (output > 0.5).astype(np.uint8)  # threshold → binary mask

# --------------------------
# Remove small speckles
# --------------------------
# def remove_small_objects(mask, min_size=200):
#     labeled = measure.label(mask, connectivity=2)
#     props = measure.regionprops(labeled)
#     cleaned = np.zeros_like(mask)
#     for prop in props:
#         if prop.area >= min_size:
#             cleaned[labeled == prop.label] = 1
#     return cleaned.astype(np.uint8)

def get_valid_footprint(img, nodata=None):
    """
    Build a binary footprint mask of valid pixels.
    img: HxWxC numpy array (RGB)
    nodata: optional nodata value from rasterio (tuple or int)
    """
    if nodata is not None:
        # Mark nodata pixels as invalid
        if isinstance(nodata, (list, tuple)):
            valid_mask = ~np.all(img == nodata, axis=-1)
        else:
            valid_mask = ~(img[:,:,0] == nodata)
    else:
        # fallback: treat pure black as invalid
        valid_mask = ~np.all(img == [0,0,0], axis=-1)

    # Convert to binary (0/1)
    valid_mask = valid_mask.astype(np.uint8)

    # Keep largest connected region
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(valid_mask, connectivity=8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # skip background
        valid_mask = (labels == largest).astype(np.uint8)

    # Optionally smooth via convex hull
    valid_mask = convex_hull_image(valid_mask).astype(np.uint8)

    return valid_mask

def clean_water_mask(water_mask, footprint, min_size=200, keep_largest=True):
    """
    Clean segmentation water mask.
    """
    # Ensure mask is binary
    water_mask = (water_mask > 0).astype(np.uint8)

    # Restrict to footprint
    water_mask[footprint == 0] = 0

    # Remove small speckles (convert to bool first)
    water_mask = remove_small_objects(water_mask.astype(bool), min_size=min_size)

    # Convert back to uint8 (0/1)
    water_mask = water_mask.astype(np.uint8)

    if keep_largest:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(water_mask, connectivity=8)
        if num_labels > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # skip background
            water_mask = (labels == largest).astype(np.uint8)

    return water_mask

def predict_for_the_area_given3(model, input_image, output_image):
    # --------------------------
    # Run on a georeferenced orthophoto
    # --------------------------
    with rasterio.open(input_image) as src:
        img = src.read([1, 2, 3])  # first 3 bands (RGB)
        img = np.transpose(img, (1, 2, 0))  # HWC
        profile = src.profile
        nodata = src.nodata

    print(f"Image shape: {img.shape}")
    mask = sliding_window_predict(img, model, WINDOW, STRIDE)
    print("Mask predicted.")
    # --------------------------
    # Save mask as GeoTIFF
    # --------------------------
    # Build footprint
    footprint = get_valid_footprint(img)

    # Restrict prediction to footprint
    mask[footprint == 0] = 0

    # Postprocess: remove speckles
    mask_clean = clean_water_mask(mask, footprint, min_size=200, keep_largest=True)
    print("Mask cleaned.")

    # Save as GeoTIFF
    profile.update(dtype=rasterio.uint8, count=1, compress="lzw")
    with rasterio.open(output_image, "w", **profile) as dst:
        dst.write(mask_clean, 1)

    print(f"✅ Water mask saved as GeoTIFF: {output_image}")

def predict_for_the_area_given2(model, input_image, output_image):
    # --------------------------
    # Run on a georeferenced orthophoto
    # --------------------------
    with rasterio.open(input_image) as src:
        img = src.read([1, 2, 3])  # first 3 bands (RGB)
        img = np.transpose(img, (1, 2, 0))  # HWC
        profile = src.profile
        nodata = src.nodata

    print(f"Image shape: {img.shape}")
    mask = sliding_window_predict(img, model, WINDOW, STRIDE)
    print("Mask predicted.")
    # --------------------------
    # Save mask as GeoTIFF
    # --------------------------
    # Build footprint
    footprint = get_valid_footprint(img)

    # Restrict prediction to footprint
    mask[footprint == 0] = 0

    # Postprocess: remove speckles
    mask_clean = remove_small_objects(mask, MIN_PATCH_SIZE)
    print("Mask cleaned.")

    # Save as GeoTIFF
    profile.update(dtype=rasterio.uint8, count=1, compress="lzw")
    with rasterio.open(output_image, "w", **profile) as dst:
        dst.write(mask_clean, 1)

    print(f"✅ Water mask saved as GeoTIFF: {output_image}")

def predict_for_the_area_given1(model, input_image, output_image):
    # --------------------------
    # Run on a georeferenced orthophoto
    # --------------------------
    with rasterio.open(input_image) as src:
        img = src.read([1, 2, 3])  # first 3 bands (RGB)
        img = np.transpose(img, (1, 2, 0))  # HWC
        profile = src.profile
        nodata = src.nodata

    print(f"Image shape: {img.shape}")
    mask = sliding_window_predict(img, model, WINDOW, STRIDE)
    print("Mask predicted.")
    # --------------------------
    # Save mask as GeoTIFF
    # --------------------------
    # If nodata is defined, mask it out
    if nodata is not None:
        with rasterio.open("orthophoto.tif") as src:
            mask_array = src.read(1)  # check one band
            nodata_mask = (mask_array == nodata)
        mask[nodata_mask] = 0

    # Postprocess: remove speckles
    mask_clean = remove_small_objects(mask, MIN_PATCH_SIZE)
    print("Mask cleaned.")

    # Save as GeoTIFF
    profile.update(dtype=rasterio.uint8, count=1, compress="lzw")
    with rasterio.open(output_image, "w", **profile) as dst:
        dst.write(mask_clean, 1)

    print(f"✅ Water mask saved as GeoTIFF: {output_image}")


def predict_for_the_area_given(model, input_image, output_image): ## working
    # --------------------------
    # Run on a georeferenced orthophoto
    # --------------------------
    with rasterio.open(input_image) as src:
        img = src.read([1, 2, 3])  # first 3 bands (RGB)
        img = np.transpose(img, (1, 2, 0))  # HWC
        profile = src.profile
        nodata_mask = src.read_masks(1)  # 0 = nodata, 255 = valid

    print(f"Image shape: {img.shape}")
    print(f"No data mask shape: {nodata_mask.shape}")
    print(f"No data mask: {nodata_mask}")
    mask = sliding_window_predict(img, model, WINDOW, STRIDE)
    mask[nodata_mask == 0] = 0  # force nodata areas to non-water
    # --------------------------
    # Save mask as GeoTIFF
    # --------------------------
    profile.update(dtype=rasterio.uint8, count=1, compress="lzw")

    with rasterio.open(output_image, "w", **profile) as dst:
        dst.write(mask, 1)

    print(f"✅ Water mask saved as GeoTIFF: {output_image}")


if __name__=="__main__":
    config_file = '../models/deeplabv3_plus/config.yaml'
    model = load_model(config_file)
    input_image = '../input/webodm/odm_orthophoto.tif'
    output_image3 = '../output/water_mask_updated3.tif'
    predict_for_the_area_given3(model, input_image, output_image3)
