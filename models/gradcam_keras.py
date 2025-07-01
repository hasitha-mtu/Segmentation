import numpy as np
import tensorflow as tf
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from models.deeplabv3_plus.train_model import load_saved_model
from common_utils.images import load_image

def score_function(output):
    # output shape: (1, H, W, num_classes)
    class_index = 1  # water class (change as needed)
    return tf.reduce_mean(output[..., class_index])

if __name__=="__main__":
    model = load_saved_model()

    # Choose a layer like 'aspp_conv_3x3' or 'decoder_conv3_bn' (depends on your model)
    target_layer_name = 'conv2d_4'  # example

    image_path = '../input/samples/segnet_512/images/DJI_20250324093158_0041_V.jpg'
    image_tensor =  load_image(image_path)
    image_tensor =  tf.expand_dims(image_tensor, axis=0)
    print(f'image_tensor shape: {image_tensor.shape}')

    model(tf.keras.Input(shape=(512, 512, 3)))  # Build model
    print(model.summary())
    print(type(model))
    print(f'model output shape: {print(model.output.shape)}')

    # Create a new input layer
    input_layer = Input(shape=(512, 512, 3))
    output_layer = model(input_layer)  # Connect to existing model

    # Wrap into a new functional model
    model1 = Model(inputs=input_layer, outputs=output_layer)
    print(model1.output.shape)

    print(f'model1 output shape: {print(model1.output.shape)}')

    # Create GradCAM object
    gradcam = Gradcam(model,
                      model_modifier=ReplaceToLinear(),
                      clone=True)

    # Generate CAM
    cam = gradcam(score_function,
                  seed_input=image_tensor,
                  penultimate_layer=target_layer_name)

    # Post-process CAM
    heatmap = cam[0]  # Shape: (H, W)

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
    plt.imshow(image_tensor.astype(np.uint8))

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM Overlay")
    plt.imshow(np.uint8(overlay))
    plt.axis('off')
    plt.show()