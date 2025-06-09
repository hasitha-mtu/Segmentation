import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def display_mask(pred):
    mask = np.argmax(pred, axis=-1)
    mask *= 127
    plt.axis("off")
    plt.imshow(mask)
    plt.show()

def show_image(dir_path, image, index, title=None, save=False):
    if save:
        if title:
            file_name = f"{dir_path}/{title}_{index}.png"
        else:
            file_name = f"{dir_path}/predicted_mask_{index}.png"
        os.makedirs(dir_path, exist_ok=True)
        plt.imsave(file_name, image)
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')

def overlay_mask_on_image(image, predicted_mask, alpha=0.5):
    """
    Overlays a predicted binary mask on an image.

    Parameters:
    - image: Original image, shape (H, W, 3), dtype uint8
    - predicted_mask: Predicted mask, shape (H, W) or (H, W, 1), dtype float32 or uint8
    - alpha: Transparency for the mask overlay

    Returns:
    - blended image with mask overlay
    """

    # Handle 3D predicted mask
    if predicted_mask.ndim == 3:
        predicted_mask = predicted_mask.squeeze()

    # Normalize and convert mask to uint8
    if predicted_mask.dtype != np.uint8:
        predicted_mask = (predicted_mask * 255).astype(np.uint8)

    # Apply colormap for visualization
    mask_colored = cv2.applyColorMap(predicted_mask, cv2.COLORMAP_JET)

    # Resize mask if needed
    if image.shape[:2] != mask_colored.shape[:2]:
        mask_colored = cv2.resize(mask_colored, (image.shape[1], image.shape[0]))

    # Convert image from float64 to uint8
    if image.dtype == np.float64:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # Convert from RGB to BGR if needed (OpenCV uses BGR)
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Blend images
    blended = cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)

    return blended

def create_blue_mask(predicted_mask):
    """
    Convert binary mask to solid blue RGB overlay.
    """
    blue_mask = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3), dtype=np.uint8)
    blue_mask[:, :, 0] = 255  # Blue channel
    return blue_mask

# def overlay_confident_mask(image, predicted_mask, alpha=0.5, threshold=0.5, color=(255, 0, 0)):
#     """
#     Overlay mask only on confident regions (where mask > threshold).
#
#     Parameters:
#     - image: original image (H, W, 3), float64 or uint8
#     - predicted_mask: soft mask (H, W), float in [0, 1]
#     - alpha: transparency of overlay
#     - threshold: confidence threshold (e.g., 0.5)
#     - color: BGR color to overlay (e.g., (255, 0, 0) for blue)
#
#     Returns:
#     - blended image with overlay only on confident regions
#     """
#     # Normalize image to uint8
#     if image.dtype != np.uint8:
#         if image.max() <= 1.0:
#             image = (image * 255).astype(np.uint8)
#         else:
#             image = image.astype(np.uint8)
#
#     # Convert RGB to BGR for OpenCV (if needed)
#     if image.shape[2] == 3:
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#     # Create a binary mask of confident areas
#     confident_mask = (predicted_mask > threshold).astype(np.uint8)
#
#     # Create a blank color mask
#     color_mask = np.zeros_like(image, dtype=np.uint8)
#     # color_mask[:, :] = color  # BGR
#     color_mask[:] = color  # This will be shape (H, W, 3)
#
#     # Apply confident mask as a 3-channel mask
#     confident_mask_3ch = np.stack([confident_mask]*3, axis=-1)
#
#     # Only overlay color where confident
#     overlay = np.where(confident_mask_3ch, color_mask, 0)
#
#     # Blend only confident areas
#     blended = image.copy()
#     blended = cv2.addWeighted(blended, 1.0, overlay, alpha, 0)
#
#     return blended

def overlay_confident_mask(image, predicted_mask, alpha=0.5, threshold=0.5, color=(255, 0, 0)):
    """
    Overlay a colored mask on confident regions only (mask > threshold).
    """

    # Convert image to uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # Convert RGB to BGR (OpenCV default)
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Binary mask of confident predictions
    confident_mask = (predicted_mask > threshold).astype(np.uint8)

    # # Expand to 3 channels
    # confident_mask_3ch = np.stack([confident_mask]*3, axis=-1)

    # Create color mask (same shape as image)
    # color_mask = np.zeros_like(image, dtype=np.uint8)
    # color_mask[:] = color  # This will be shape (H, W, 3)
    color_mask = np.full_like(image, color, dtype=np.uint8)

    print(f'overlay_mask_on_image|confident_mask shape is {confident_mask.shape}')
    print(f'overlay_mask_on_image|color_mask shape is {color_mask.shape}')
    # print(f'overlay_mask_on_image|confident_mask_3ch shape is {confident_mask_3ch.shape}')

    # Only keep color where confident
    overlay = color_mask * confident_mask

    # Blend
    blended = cv2.addWeighted(image, 1.0, overlay, alpha, 0)

    return blended

def overlay_mask(image, predicted_mask, alpha=0.2):
    """
    Overlays a predicted binary mask on an image.

    Parameters:
    - image: Original image, shape (H, W, 3), dtype uint8
    - predicted_mask: Predicted mask, shape (H, W) or (H, W, 1), dtype float32 or uint8
    - alpha: Transparency for the mask overlay

    Returns:
    - blended image with mask overlay
    """

    # Handle 3D predicted mask
    if predicted_mask.ndim > 2:
        predicted_mask = predicted_mask.squeeze()

    # Normalize and convert mask to uint8
    if predicted_mask.dtype != np.uint8:
        predicted_mask = (predicted_mask * 255).astype(np.uint8)

    # Apply colormap for visualization
    mask_colored = cv2.applyColorMap(predicted_mask, cv2.COLORMAP_JET)

    # Resize mask if needed
    if image.shape[:2] != mask_colored.shape[:2]:
        mask_colored = cv2.resize(mask_colored, (image.shape[1], image.shape[0]))

    # Convert image from float64 to uint8
    if image.dtype == np.float64:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # Convert from RGB to BGR if needed (OpenCV uses BGR)
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    print(f'overlay_mask_on_image|image shape is {image.shape}')
    print(f'overlay_mask_on_image|mask_colored shape is {mask_colored.shape}')

    print(f'overlay_mask_on_image|image data type is {image.dtype}')
    print(f'overlay_mask_on_image|mask_colored data type is {mask_colored.dtype}')

    # Blend images
    blended = cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)

    return blended


if __name__ == "__main__":
    sample1_image = "../../input/samples/sample1.jpg"
    sample1_mask = "../../input/samples/sample1_mask.png"
    image = cv2.imread(sample1_image)
    image_mask = cv2.imread(sample1_mask)
    plt.figure(figsize=(10, 8))

    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    blended1 = overlay_mask(image, image_mask, alpha=0.1)
    plt.imshow(blended1)
    plt.title("blended1")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    blended2 = overlay_mask(image, image_mask, alpha=0.2)
    plt.imshow(blended2)
    plt.title("blended2")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    blended3 = overlay_mask(image, image_mask, alpha=0.5)
    plt.imshow(blended3)
    plt.title("blended3")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

