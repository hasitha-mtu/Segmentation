import numpy as np
import tensorflow as tf
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
import matplotlib.pyplot as plt
import cv2

from common_utils.images import load_image

from models.train import load_saved_unet_model
from models.train import load_saved_unet_ffc_model
from models.train import load_saved_unet_VGG16_model
from models.train import load_saved_unet_ResNet50_model
from models.train import load_saved_unet_MobileNetV2_model
from models.train import load_saved_unet_plus_plus_model
from models.train import load_saved_segnet_model
from models.train import load_saved_segnet_VGG16_model
from models.train import load_saved_res_unet_plus_plus_model
from models.train import load_saved_deeplabv3_plus_model

def score(output):
    return tf.reduce_mean(output[..., 0])  # as we have one target water class

def gradcam(image, model, target_layer_name = 'conv2d_4'):
    print(f'gradcam|model summary: {model.summary()}')
    image_tensor = tf.expand_dims(image, axis=0)

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
    overlay = np.uint8(overlay)
    return overlay

def gradcam_plus_plus(image, model, target_layer_name = 'conv2d_4'):
    image_tensor = tf.expand_dims(image, axis=0)

    # Create GradCAM++ object
    gradcam_pp = GradcamPlusPlus(model,
                                 model_modifier=ReplaceToLinear(),
                                 clone=True)

    # Generate CAM
    cam = gradcam_pp(score,
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
    overlay = np.uint8(overlay)
    return overlay

def execute_gradcam(image, model, target_layer_name = 'conv2d_4'):
    image_tensor = tf.expand_dims(image, axis=0)

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
    overlay = np.uint8(overlay)
    visualize(overlay, image_tensor)

def execute_gradcam_plus_plus(image, model, target_layer_name = 'conv2d_4'):
    image_tensor = tf.expand_dims(image, axis=0)

    # Create GradCAM++ object
    gradcam_pp = GradcamPlusPlus(model,
                                 model_modifier=ReplaceToLinear(),
                                 clone=True)

    # Generate CAM
    cam = gradcam_pp(score,
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
    overlay = np.uint8(overlay)
    visualize(overlay, image_tensor)

def visualize(heatmap, image):
    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM Overlay")
    plt.imshow(heatmap)
    plt.axis('off')
    plt.show()

if __name__=="__main__":
    model = load_saved_unet_model('Adam', True)
    print(f"model summary : {model.summary()}")

    image_path = '../input/samples/segnet_512/images/DJI_20250324092953_0009_V.jpg'
    image_tensor = load_image(image_path)
    target_layer_name = 'conv2d_25'
    execute_gradcam(image_tensor, model, target_layer_name)
    execute_gradcam_plus_plus(image_tensor, model, target_layer_name)