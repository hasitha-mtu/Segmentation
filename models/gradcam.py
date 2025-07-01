import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from models.unet_MobileNetV2.train_model import load_saved_model
from common_utils.images import load_image

def compute_gradcam(image_tensor, gradcam_model, class_index=1):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = gradcam_model(image_tensor)
        # Assume binary segmentation: get gradients w.r.t. water class
        loss = predictions[..., class_index]  # shape: (B, H, W)

    grads = tape.gradient(loss, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(1, 2))  # GAP
    cam = tf.reduce_sum(tf.multiply(conv_outputs, weights[:, tf.newaxis, tf.newaxis, :]), axis=-1)

    cam = tf.nn.relu(cam)
    cam = tf.image.resize(cam[..., tf.newaxis], (image_tensor.shape[1], image_tensor.shape[2]))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()[0, ..., 0]  # 2D heatmap

def overlay_gradcam(image, cam):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = heatmap * 0.4 + image
    return np.uint8(overlay)

def grad_cam_segmentation(model, image, target_layer_name, target_class_index=1):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(target_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = tf.reduce_mean(predictions[..., target_class_index])  # target class only

    grads = tape.gradient(loss, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(1, 2))  # Global average pooling
    cam = tf.reduce_sum(tf.multiply(weights[:, tf.newaxis, tf.newaxis, :], conv_outputs), axis=-1)

    cam = tf.nn.relu(cam)
    cam = cam[0].numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize 0-1

    return cam

# if __name__=="__main__":
#     cam_map = grad_cam_segmentation(model, image, target_layer_name="conv_last", target_class_index=1)
#
#     # Resize CAM to match input
#     cam_resized = cv2.resize(cam_map, (image.shape[2], image.shape[1]))
#
#     # Overlay on original image
#     heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
#     overlay = 0.4 * heatmap + 0.6 * image[0].numpy()
#
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.title("Original Image")
#     plt.imshow(image[0].numpy().astype(np.uint8))
#
#     plt.subplot(1, 2, 2)
#     plt.title("Grad-CAM Overlay")
#     plt.imshow(np.uint8(overlay))
#     plt.axis('off')
#     plt.show()

# if __name__=="__main__":
#     input_image = original_image_tensor[0].numpy()
#     cam = compute_gradcam(input_tensor, gradcam_model)
#     overlay = overlay_gradcam(input_image, cam)
#
#     plt.imshow(overlay)
#     plt.title("Grad-CAM Overlay")
#     plt.axis(False)
#     plt.show()

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
import matplotlib.pyplot as plt
from tf_keras_vis.utils.scores import CategoricalScore

def water_score_function(output):
    # Assume binary segmentation → focus on class 1 (water)
    return tf.reduce_mean(output[..., 1])  # shape: (B, H, W, C)

if __name__=="__main__":
    model = load_saved_model()
    gradcam = Gradcam(model,
                      model_modifier=ReplaceToLinear(),  # important for softmax/sigmoid
                      clone=True)

    image_path = '../input/samples/segnet_512/images/DJI_20250324093158_0041_V.jpg'
    image_tensor = load_image(image_path)
    print(f'image_tensor shape: {image_tensor.shape}')
    # Assume input shape (1, H, W, 3), scaled [0, 1]
    gradcam = Gradcam(model=model,model_modifier=ReplaceToLinear(),clone=True, score=water_score_function())
    cam = gradcam(score_function=water_score_function,
                  seed_input=image_tensor,  # shape = (1, H, W, 3)
                  penultimate_layer='block_13_depthwise_relu')  # e.g., 'conv5_block3_out'

    heatmap = cam[0]
    plt.imshow(image_tensor[0])
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title("Grad-CAM for Water Class")
    plt.axis(False)
    plt.show()

