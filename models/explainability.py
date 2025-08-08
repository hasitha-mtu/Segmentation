import numpy as np
import tensorflow as tf
# from tf_keras_vis.gradcam import Gradcam
# from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
# from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
import matplotlib.pyplot as plt
import cv2

from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

from tf_keras_vis.scorecam import ScoreCAM
from tf_keras_vis.utils.scores import CategoricalScore

from common_utils.images import load_image
from models.common_utils.images import save_image

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

# def score_cam(image, model, target_layer_name = 'conv2d_4'):
#     image_tensor = tf.expand_dims(image, axis=0)
#
#     # Create GradCAM++ object
#     score_cam = Scorecam(model,
#                                  model_modifier=ReplaceToLinear(),
#                                  clone=True)
#
#     # Generate CAM
#     cam = score_cam(score,
#                   seed_input=image_tensor,
#                   penultimate_layer=target_layer_name)
#
#     # Post-process CAM
#     heatmap = cam[0]  # Shape: (H, W)
#
#     image_tensor = image_tensor.numpy().squeeze()
#
#     # Resize CAM to image size
#     heatmap_resized = cv2.resize(heatmap, (image_tensor.shape[1], image_tensor.shape[0]))
#
#     # Normalize heatmap
#     heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())
#
#     # Apply colormap
#     heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
#     overlay = 0.4 * heatmap_color + 0.6 * image_tensor.astype(np.uint8)
#     overlay = np.uint8(overlay)
#     return overlay

def execute_gradcam(image, model, output_path, target_layer_name = 'conv2d_4'):
    image_tensor = tf.expand_dims(image, axis=0)

    # Create GradCAM object
    gradcam = Gradcam(model,
                      model_modifier=ReplaceToLinear(),
                      clone=True)
    # Generate CAM
    cam = gradcam(score,
                  seed_input=image_tensor,
                  penultimate_layer=target_layer_name)

    print(f'execute_gradcam|cam shape: {cam.shape}')

    # Post-process CAM
    heatmap = cam[0]  # Shape: (H, W)
    print(f'execute_gradcam|heatmap shape: {heatmap.shape}')

    image_tensor = image_tensor.numpy().squeeze()

    # Resize CAM to image size
    heatmap_resized = cv2.resize(heatmap, (image_tensor.shape[1], image_tensor.shape[0]))

    # Normalize heatmap
    heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())

    # Apply colormap
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = 0.4 * heatmap_color + 0.6 * image_tensor.astype(np.uint8)
    overlay = np.uint8(overlay)
    save_image(output_path, overlay, 'gradcam')
    return heatmap

def execute_gradcam_plus_plus(image, model, output_path, target_layer_name = 'conv2d_4'):
    image_tensor = tf.expand_dims(image, axis=0)

    # Create GradCAM++ object
    gradcam_pp = GradcamPlusPlus(model,
                                 model_modifier=ReplaceToLinear(),
                                 clone=True)

    # Generate CAM
    cam = gradcam_pp(score,
                  seed_input=image_tensor,
                  penultimate_layer=target_layer_name)

    print(f'execute_gradcam_plus_plus|cam shape: {cam.shape}')

    # Post-process CAM
    heatmap = cam[0]  # Shape: (H, W)
    print(f'execute_gradcam_plus_plus|heatmap shape: {heatmap.shape}')

    image_tensor = image_tensor.numpy().squeeze()

    # Resize CAM to image size
    heatmap_resized = cv2.resize(heatmap, (image_tensor.shape[1], image_tensor.shape[0]))

    # Normalize heatmap
    heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())

    # Apply colormap
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = 0.4 * heatmap_color + 0.6 * image_tensor.astype(np.uint8)
    overlay = np.uint8(overlay)
    save_image(output_path, overlay, 'gradcam_pp')
    return heatmap


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


def segmentation_score1(output, target_class_index):
    """
    Custom score function for segmentation models. It returns the
    average confidence of the pixels classified as the target class.

    Args:
        output (tf.Tensor): The model's segmentation output mask with shape (1, H, W, num_classes).
        target_class_index (int): The index of the class to evaluate.

    Returns:
        tf.Tensor: A tensor with the average confidence score for the target class.
    """
    if len(output.shape) != 4:
        raise ValueError("Output tensor must be 4D (batch, height, width, classes).")

    class_probabilities = output[0, :, :, target_class_index]
    return tf.reduce_mean(class_probabilities)

def segmentation_score(output):
    return tf.reduce_mean(output[..., 0])


def calculate_deletion_metric(model, images, heatmaps, num_steps=100):
    """
    Calculates the deletion metric for a set of heatmaps.

    Args:
        model: The trained Keras model.
        images: A NumPy array of input images.
        heatmaps: A NumPy array of heatmaps, one for each image.
        target_class_index: The index of the class to evaluate.
        num_steps: The number of steps to perform the deletion.

    Returns:
        A list of average prediction scores at each step.
    """
    all_scores = []

    # Squeeze the heatmaps to ensure they are 2D arrays (H, W)
    heatmaps = np.squeeze(heatmaps)

    if heatmaps.ndim == 2:  # Handle a single heatmap
        heatmaps = np.expand_dims(heatmaps, axis=0)

    if heatmaps.ndim > 3:
        raise ValueError("Heatmaps must be a list of 2D arrays, not 4D.")

    for i in range(images.shape[0]):
        original_image = images[i:i + 1]
        heatmap = heatmaps[i]

        flattened_heatmap = heatmap.flatten()
        sorted_indices = np.argsort(flattened_heatmap)[::-1]

        scores_for_image = []
        for j in range(num_steps):
            mask_size = int(len(sorted_indices) * (j / num_steps))
            mask_indices = sorted_indices[:mask_size]

            perturbed_image = np.copy(original_image)

            # Unravel indices based on the 2D shape of the heatmap
            rows, cols = np.unravel_index(mask_indices, heatmap.shape)

            # Set the masked pixels to a neutral value (e.g., mean pixel value)
            perturbed_image[0, rows, cols, :] = np.mean(perturbed_image)

            # Get the model's prediction score for the target class
            prediction_mask = model.predict(perturbed_image, verbose=0)
            # score = segmentation_score(prediction_mask, target_class_index).numpy()
            score = segmentation_score(prediction_mask).numpy()
            scores_for_image.append(score)

        all_scores.append(scores_for_image)

    avg_scores = np.mean(all_scores, axis=0)

    return avg_scores

from sklearn.metrics import auc

if __name__=="__main__":
    model = load_saved_unet_model('Adam', True)
    print(f"model summary : {model.summary()}")

    image_path = '../input/samples/segnet_512/images/DJI_20250324092953_0009_V.jpg'

    image = load_image(image_path)
    print(f'image shape: {image.shape}')
    # print(f'image_tensor shape: {image_tensor.shape}')
    penultimate_layer_name = 'conv2d_23'
    output_path = "../output/gradcam"

    cam_gradcam = execute_gradcam(image, model, output_path, penultimate_layer_name)
    cam_gradcam_plus_plus = execute_gradcam_plus_plus(image, model, output_path, penultimate_layer_name)

    image_tensor = tf.expand_dims(image, axis=0)
    gradcam_scores = calculate_deletion_metric(model, image_tensor, cam_gradcam, num_steps=100)
    gradcam_plus_plus_scores = calculate_deletion_metric(model, image_tensor, cam_gradcam_plus_plus, num_steps=100)

    # 6. Plot the results
    x_axis = np.arange(100) / 100
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, gradcam_scores, label='Grad-CAM')
    plt.plot(x_axis, gradcam_plus_plus_scores, label='Grad-CAM++')
    plt.xlabel('Fraction of Pixels Deleted')
    plt.ylabel('Prediction Score Drop')
    plt.title('Quantitative Evaluation: Deletion Metric for River Water')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 7. Calculate and print a single quantitative value (Area Over the Curve)
    aoc_gradcam = auc(x_axis, gradcam_scores)
    aoc_gradcam_plus_plus = auc(x_axis, gradcam_plus_plus_scores)

    print(f"Area Over the Curve (Grad-CAM): {aoc_gradcam:.4f}")
    print(f"Area Over the Curve (Grad-CAM++): {aoc_gradcam_plus_plus:.4f}")
