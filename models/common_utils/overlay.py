import numpy as np
import cv2
import matplotlib.pyplot as plt

OUTPUT_DIR = "C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\output\\18_07_2025\\0"


def overlay_mask(image, predicted_mask, alpha=0.2, gamma=0.1):
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
    blended = cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, gamma)

    return blended

# --- Example Usage ---
if __name__ == "__main__":
    image_height, image_width = 512, 512
    image_path = f'{OUTPUT_DIR}/image_0.png'
    mask_path = f'{OUTPUT_DIR}/UNET_0.png'

    # You would load your actual image like this:
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) # OpenCV loads as BGR, convert to RGB
    # original_image = img_to_array(load_img(image_path, color_mode='rgb'))

    print(f'original_image shape:{original_image.shape}')
    print(f'original_image type:{type(original_image)}')

    prediction_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    print(f'prediction_mask shape:{prediction_mask.shape}')
    print(f'prediction_mask type:{type(prediction_mask)}')

    overlaid_image = overlay_mask(original_image, prediction_mask, alpha=0.2, gamma=1)

    # --- Display results using Matplotlib ---
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    # For a (512,512,1) mask, imshow can handle it but it's good practice to squeeze
    plt.imshow(prediction_mask.squeeze(), cmap='gray')
    plt.title('Prediction Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(overlaid_image)
    plt.title('Overlay Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # You can also save the overlaid image
    # cv2.imwrite('overlaid_image_green.png', cv2.cvtColor(overlaid_img_green, cv2.COLOR_RGB2BGR)
