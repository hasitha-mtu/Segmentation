import cv2
import numpy as np
import matplotlib.pyplot as plt # For displaying results

def overlay_mask_on_image(image, mask, alpha = 0.5, gamma = 0):
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