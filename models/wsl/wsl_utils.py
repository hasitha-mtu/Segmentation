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

    print(f'overlay_mask_on_image|image shape is {image.shape}')
    print(f'overlay_mask_on_image|mask_colored shape is {mask_colored.shape}')

    print(f'overlay_mask_on_image|image data type is {image.dtype}')
    print(f'overlay_mask_on_image|mask_colored data type is {mask_colored.dtype}')

    # Blend images
    blended = cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)

    return blended