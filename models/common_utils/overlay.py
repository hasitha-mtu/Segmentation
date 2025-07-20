import cv2
import numpy as np
import matplotlib.pyplot as plt # For displaying results

def overlay_mask_on_image1(image, mask, alpha = 0.5, gamma = 0):
    image = image.squeeze()
    mask = mask.squeeze()
    print(f'overlay_mask_on_image|image shape:{image.shape}')
    print(f'overlay_mask_on_image|image type:{type(image)}')
    print(f'overlay_mask_on_image|mask shape:{mask.shape}')
    print(f'overlay_mask_on_image|mask type:{type(mask)}')
    beta = 1 - alpha
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    mask_color = [0, 0, 255]  # Red color for the overlay (BGR format)
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    colored_mask[binary_mask == 255] = mask_color

    print(f'overlay_mask_on_image|colored_mask shape:{colored_mask.shape}')
    print(f'overlay_mask_on_image|colored_mask type:{type(colored_mask)}')

    # Overlay the colored mask on the original image
    overlay_alpha = cv2.addWeighted(image, beta, colored_mask, alpha, gamma)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Binary Mask')
    plt.imshow(binary_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Overlay (Alpha Blending)')
    plt.imshow(cv2.cvtColor(overlay_alpha, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


# # --- 1. Load your image and mask (example data) ---
# # Replace with your actual image and mask loading
# # For demonstration, let's create a dummy image and mask
# image = np.zeros((300, 400, 3), dtype=np.uint8) # Black image
# cv2.rectangle(image, (50, 50), (200, 200), (0, 255, 0), -1) # Green rectangle
#
# mask = np.zeros((300, 400), dtype=np.uint8) # Binary mask
# cv2.circle(mask, (250, 150), 80, 255, -1) # White circle for the mask
#
# # Ensure mask is binary (0 or 255) if it's not already
# _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
#
# # --- Method 1.1: Simple Alpha Blending ---
# # This method makes the masked region semi-transparent.
#
# alpha = 0.5 # Transparency factor for the mask (0.0 - 1.0)
# beta = 1 - alpha # Transparency factor for the original image
# gamma = 0 # Scalar added to each sum
#
# # Convert mask to 3 channels to blend with a color if desired
# # Or, if you want to just overlay the mask itself as a grayscale, skip this part
# # and directly use weighted_add below with the binary_mask
# mask_color = [0, 0, 255] # Red color for the overlay (BGR format)
# colored_mask = np.zeros_like(image, dtype=np.uint8)
# colored_mask[binary_mask == 255] = mask_color
#
# # Overlay the colored mask on the original image
# overlay_alpha = cv2.addWeighted(image, beta, colored_mask, alpha, gamma)
#
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 3, 1)
# plt.title('Original Image')
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.axis('off')
#
# plt.subplot(1, 3, 2)
# plt.title('Binary Mask')
# plt.imshow(binary_mask, cmap='gray')
# plt.axis('off')
#
# plt.subplot(1, 3, 3)
# plt.title('Overlay (Alpha Blending)')
# plt.imshow(cv2.cvtColor(overlay_alpha, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()
#
# # --- Method 1.2: Masking with a specific color (solid color overlay) ---
# # This method paints the masked region with a solid color.
#
# overlay_image_solid = image.copy()
# mask_color_solid = [0, 255, 255] # Yellow in BGR
#
# # Apply the color to the masked region
# overlay_image_solid[binary_mask == 255] = mask_color_solid
#
# plt.figure(figsize=(8, 4))
# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.axis('off')
#
# plt.subplot(1, 2, 2)
# plt.title('Overlay (Solid Color)')
# plt.imshow(cv2.cvtColor(overlay_image_solid, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()
#
# # --- Method 1.3: Combined Method (Highlighting and Original) ---
# # This method allows you to show the original image in the unmasked area
# # and a colored/highlighted version in the masked area.
#
# overlay_combined = image.copy()
# mask_color_combined = [0, 0, 255] # Red
#
# # Apply the mask color to the overlay image
# overlay_combined[binary_mask == 255] = mask_color_combined
#
# # Blend the original image with the colored overlay image using the mask as alpha
# # This is a bit more complex, essentially a custom alpha blend where
# # the unmasked part is 100% original, and the masked part is 100% colored.
# # You can achieve a similar effect by using `cv2.addWeighted` with the mask as weights.
#
# # More common way for this kind of "highlight" is to use the alpha blend from 1.1
# # but set the background to the original image and foreground to the colored mask.
# # Let's refine the alpha blend for this:
#
# # Create a background image that is the original image
# background = image.copy()
#
# # Create a foreground image that is just the colored mask
# foreground = np.zeros_like(image, dtype=np.uint8)
# foreground[binary_mask == 255] = mask_color
#
# # Convert mask to 3 channels and normalize to 0-1 for blending
# alpha_channel = binary_mask / 255.0
# alpha_channel_3d = np.stack([alpha_channel, alpha_channel, alpha_channel], axis=-1)
#
# # Blend: result = (foreground * alpha) + (background * (1 - alpha))
# overlay_highlight = (foreground * alpha_channel_3d + background * (1 - alpha_channel_3d)).astype(np.uint8)
#
#
# plt.figure(figsize=(8, 4))
# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.axis('off')
#
# plt.subplot(1, 2, 2)
# plt.title('Overlay (Highlight)')
# plt.imshow(cv2.cvtColor(overlay_highlight, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()

import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.utils import load_img, img_to_array

OUTPUT_DIR = "C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\output\\18_07_2025\\0"

def overlay_mask_on_image(image, mask, color=(0, 255, 0), alpha=0.5):
    """
    Overlays a binary or grayscale segmentation mask on an RGB image.

    Args:
        image (np.ndarray): The original RGB image (H, W, 3).
        mask (np.ndarray): The binary or grayscale mask (H, W, 1) or (H, W).
                           Assumes 1 for foreground, 0 for background.
        color (tuple): The RGB color for the mask overlay (e.g., (0, 255, 0) for green).
        alpha (float): The transparency of the mask overlay (0.0 for fully transparent, 1.0 for fully opaque).

    Returns:
        np.ndarray: The image with the mask overlaid.
    """
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask.squeeze() # Remove the channel dimension if present (512, 512, 1) -> (512, 512)

    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("Image and mask must have the same height and width.")

    # Convert mask to 3 channels for coloring
    # Create an empty 3-channel image for the colored mask
    colored_mask = np.zeros_like(image, dtype=np.uint8)

    # Apply the desired color to the mask regions
    # Where mask is > 0 (assuming foreground is 1), apply the color
    # Ensure mask is boolean for direct indexing or use np.where
    mask_bool = mask > 0

    colored_mask[mask_bool] = color

    # Blend the colored mask with the original image
    # result = alpha * colored_mask + (1 - alpha) * image
    overlaid_image = cv2.addWeighted(colored_mask, alpha, image, 1 - alpha, 0)

    # Alternatively, you could directly manipulate pixels for non-masked regions:
    # result = image.copy()
    # result[mask_bool] = cv2.addWeighted(image[mask_bool], 1 - alpha, colored_mask[mask_bool], alpha, 0)

    return overlaid_image

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

    overlaid_image = overlay_mask(original_image, prediction_mask, alpha=0.2)

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

# # --- Example Usage ---
# if __name__ == "__main__":
#     # 1. Create a dummy original image (e.g., a simple gradient or random colors)
#     # This simulates your (512, 512, 3) image
#     image_height, image_width = 512, 512
#     # original_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
#     #
#     # # Fill with a simple gradient to make it visible
#     # for i in range(image_height):
#     #     original_image[i, :, 0] = int(i / image_height * 255)  # Red gradient
#     #     original_image[i, :, 1] = int((image_height - i) / image_height * 255) # Green gradient
#     # original_image[:, :, 2] = 100 # Blue constant
#
#     image_path = f'{OUTPUT_DIR}/image_0.png'
#     mask_path = f'{OUTPUT_DIR}/UNET_0.png'
#
#     # You would load your actual image like this:
#     # original_image = cv2.imread(image_path)
#     # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) # OpenCV loads as BGR, convert to RGB
#     original_image = img_to_array(load_img(image_path, color_mode='rgb'))
#
#     print(f'original_image shape:{original_image.shape}')
#     print(f'original_image type:{type(original_image)}')
#
#     # prediction_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     prediction_mask = img_to_array(load_img(mask_path, color_mode='grayscale'))
#
#     print(f'prediction_mask shape:{prediction_mask.shape}')
#     print(f'prediction_mask type:{type(prediction_mask)}')
#
#     center_y, center_x = image_height // 2, image_width // 2
#     radius = 100
#     for y in range(image_height):
#         for x in range(image_width):
#             if (x - center_x)**2 + (y - center_y)**2 < radius**2:
#                 prediction_mask[y, x, 0] = 1 # Mark as foreground
#
#     # You would load your actual prediction mask from your model output:
#     # prediction_mask = model.predict(input_image)
#     # Ensure it's binary (0 or 1) and the correct data type (np.uint8 or bool)
#
#     # --- Perform the overlay ---
#     # Example 1: Green overlay with 50% transparency
#     overlaid_img_green = overlay_mask_on_image(original_image, prediction_mask, color=(0, 255, 0), alpha=0.5)
#
#     # Example 2: Blue overlay with 30% transparency
#     overlaid_img_blue = overlay_mask_on_image(original_image, prediction_mask, color=(0, 0, 255), alpha=0.3)
#
#     # Example 3: Red overlay with high transparency (more subtle)
#     overlaid_img_red_subtle = overlay_mask_on_image(original_image, prediction_mask, color=(255, 0, 0), alpha=0.2)
#
#     # --- Display results using Matplotlib ---
#     plt.figure(figsize=(15, 5))
#
#     plt.subplot(1, 4, 1)
#     plt.imshow(original_image)
#     plt.title('Original Image')
#     plt.axis('off')
#
#     plt.subplot(1, 4, 2)
#     # For a (512,512,1) mask, imshow can handle it but it's good practice to squeeze
#     plt.imshow(prediction_mask.squeeze(), cmap='gray')
#     plt.title('Prediction Mask')
#     plt.axis('off')
#
#     plt.subplot(1, 4, 3)
#     plt.imshow(overlaid_img_green)
#     plt.title('Overlay (Green, Alpha 0.5)')
#     plt.axis('off')
#
#     plt.subplot(1, 4, 4)
#     plt.imshow(overlaid_img_blue)
#     plt.title('Overlay (Blue, Alpha 0.3)')
#     plt.axis('off')
#
#     plt.tight_layout()
#     plt.show()
#
#     # You can also save the overlaid image
#     # cv2.imwrite('overlaid_image_green.png', cv2.cvtColor(overlaid_img_green, cv2.COLOR_RGB2BGR))