import numpy as np
import tensorflow as tf
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
import matplotlib.pyplot as plt
import cv2

from models.deeplabv3_plus.train_model import load_saved_model
from common_utils.images import load_image

def score(output):
    return tf.reduce_mean(output[..., 0])  # ✅ target water class

if __name__=="__main__":
    model = load_saved_model()
    print(f"model summary : {model.summary()}")
    target_layer_name = 'conv2d_4' # (None, 32, 32, 256)

    image_path = '../input/samples/segnet_512/images/DJI_20250324092953_0009_V.jpg'
    image_tensor =  load_image(image_path)
    image_tensor =  tf.expand_dims(image_tensor, axis=0)

    # Create GradCAM object
    gradcam = Gradcam(model,
                      model_modifier=ReplaceToLinear(),
                      clone=True)

    # Generate CAM
    cam = gradcam(score,
                  seed_input=image_tensor,
                  penultimate_layer=target_layer_name)

    # Post-process CAM
    heatmap = cam[0]  # Shape: (H, W)
    image_tensor = image_tensor.numpy().squeeze()

    # Resize CAM to image size
    heatmap_resized = cv2.resize(heatmap, (image_tensor.shape[1], image_tensor.shape[0]))

    # Normalize heatmap
    heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())

    # Apply colormap
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = 0.4 * heatmap_color + 0.6 * image_tensor.astype(np.uint8)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image_tensor)

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM Overlay")
    plt.imshow(np.uint8(overlay))
    plt.axis('off')
    plt.show()