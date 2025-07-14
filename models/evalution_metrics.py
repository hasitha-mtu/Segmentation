import tensorflow as tf
import numpy as np
import time
import cv2
from scipy.spatial.distance import directed_hausdorff
from keras.utils import load_img, img_to_array
import os

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

from models.common_utils.images import save_image
from models.common_utils.data import load_dataset

from train import train_all_models

from gradcam_keras import gradcam,gradcam_plus_plus

# OUTPUT_DIR = "C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\output"
OUTPUT_DIR = "C:\\Users\AdikariAdikari\OneDrive - Munster Technological University\ModelResults\Segmentation\output4"

# Dice Coefficient
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)
    print(f'dice_coefficient|y_true shape:{y_true.shape}')
    print(f'dice_coefficient|y_pred shape:{y_pred.shape}')
    print(f'dice_coefficient|mask shape:{mask.shape}')
    y_true = y_true * mask
    y_pred = y_pred * mask
    # Ensure rank-4 tensors
    y_true = tf.expand_dims(y_true, axis=0) if len(tf.shape(y_true)) == 3 else y_true
    y_true = tf.cast(y_true, tf.float32)

    y_pred = tf.cast(y_pred, tf.float32)

    # If y_pred has 3 channels, convert to single channel via argmax or mean
    if tf.shape(y_pred)[-1] == 3:
        y_pred = tf.reduce_mean(y_pred, axis=-1, keepdims=True)

    # Reshape to flat vectors
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])

    # Dice calculation
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# IoU Score
def iou_score(y_true, y_pred, smooth=1e-6):
    mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)
    print(f'iou_score|y_true shape:{y_true.shape}')
    print(f'iou_score|y_pred shape:{y_pred.shape}')
    print(f'iou_score|mask shape:{mask.shape}')
    y_true = y_true * mask
    y_pred = y_pred * mask
    # Ensure rank-4 tensors
    y_true = tf.expand_dims(y_true, axis=0) if len(tf.shape(y_true)) == 3 else y_true
    y_true = tf.cast(y_true, tf.float32)

    y_pred = tf.cast(y_pred, tf.float32)

    # If y_pred has 3 channels, convert to single channel via argmax or mean
    if tf.shape(y_pred)[-1] == 3:
        y_pred = tf.reduce_mean(y_pred, axis=-1, keepdims=True)

    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


# Pixel Accuracy
def pixel_accuracy(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)
    print(f'pixel_accuracy|y_true shape:{y_true.shape}')
    print(f'pixel_accuracy|y_pred shape:{y_pred.shape}')
    print(f'pixel_accuracy|mask shape:{mask.shape}')
    y_true = y_true * mask
    y_pred = y_pred * mask
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(tf.math.greater_equal(y_pred, 0.5), tf.bool)
    correct = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.float32))
    total = tf.size(y_true, out_type=tf.float32)
    return correct / total


# Precision and Recall
def precision_recall(y_true, y_pred, smooth=1e-6):
    mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)
    print(f'precision_recall|y_true shape:{y_true.shape}')
    print(f'precision_recall|y_pred shape:{y_pred.shape}')
    print(f'precision_recall|mask shape:{mask.shape}')
    y_true = y_true * mask
    y_pred = y_pred * mask
    # Ensure rank-4 tensors
    y_true = tf.expand_dims(y_true, axis=0) if len(tf.shape(y_true)) == 3 else y_true
    y_true = tf.cast(y_true, tf.float32)

    y_pred = tf.cast(y_pred, tf.float32)

    # If y_pred has 3 channels, convert to single channel via argmax or mean
    if tf.shape(y_pred)[-1] == 3:
        y_pred = tf.reduce_mean(y_pred, axis=-1, keepdims=True)

    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])

    true_positives = tf.reduce_sum(y_true_f * y_pred_f)
    predicted_positives = tf.reduce_sum(y_pred_f)
    actual_positives = tf.reduce_sum(y_true_f)

    precision = (true_positives + smooth) / (predicted_positives + smooth)
    recall = (true_positives + smooth) / (actual_positives + smooth)

    return precision, recall


# Measure Inference Time (optional)
def measure_inference_time(model, input_sample):
    start_time = time.time()
    _ = model.predict(input_sample)
    end_time = time.time()
    return (end_time - start_time) * 1000  # ms


# Combine All Metrics
def evaluate_segmentation(y_true, y_pred, model=None, sample=None):
    metrics = {
        'Dice Coefficient': float(dice_coefficient(y_true, y_pred).numpy()),
        'IoU Score': float(iou_score(y_true, y_pred).numpy()),
        'Pixel Accuracy': float(pixel_accuracy(y_true, y_pred).numpy())
    }

    precision, recall = precision_recall(y_true, y_pred)
    metrics['Precision'] = float(precision.numpy())
    metrics['Recall'] = float(recall.numpy())

    if model and sample is not None:
        metrics['Inference Time (ms)'] = float(measure_inference_time(model, sample))

    b_iou = boundary_iou(y_true, y_pred)
    metrics['Boundary IoU'] = b_iou
    hd = hausdorff_distance(y_true, y_pred)
    metrics['Hausdorff Distance'] = hd

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    return metrics


def get_boundary(mask, dilation_ratio=0.02):
    """
    Extract the boundary pixels from a binary mask using dilation and erosion.
    dilation_ratio: proportion of image diagonal to determine boundary width
    """
    print(f'get_boundary|mask shape:{mask.shape}')
    mask = np.squeeze(mask)
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = max(1, int(round(dilation_ratio * img_diag)))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=dilation)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=dilation)
    boundary = dilated - eroded
    return boundary


def boundary_iou(y_true, y_pred, dilation_ratio=0.02):
    mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)
    print(f'boundary_iou|y_true shape:{y_true.shape}')
    print(f'boundary_iou|y_pred shape:{y_pred.shape}')
    print(f'boundary_iou|mask shape:{mask.shape}')

    """
    Calculate Boundary IoU score for binary masks.
    """
    y_true = y_true.astype(np.bool_)
    y_pred = y_pred.astype(np.bool_)

    y_true = y_true * mask
    y_pred = y_pred * mask

    # Ensure rank-4 tensors
    y_true = tf.expand_dims(y_true, axis=0) if len(tf.shape(y_true)) == 3 else y_true
    y_true = tf.cast(y_true, tf.float32)

    y_pred = tf.cast(y_pred, tf.float32)

    # If y_pred has 3 channels, convert to single channel via argmax or mean
    if tf.shape(y_pred)[-1] == 3:
        y_pred = tf.reduce_mean(y_pred, axis=-1, keepdims=True)

    true_boundary = get_boundary(y_true, dilation_ratio)
    pred_boundary = get_boundary(y_pred, dilation_ratio)

    intersection = np.logical_and(true_boundary, pred_boundary).sum()
    union = np.logical_or(true_boundary, pred_boundary).sum()

    if union == 0:
        return 1.0  # Both boundaries empty means perfect match

    return intersection / union


def hausdorff_distance(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)
    y_true = y_true * mask
    y_pred = y_pred * mask

    """
    Compute the Hausdorff Distance between two binary masks.
    Returns the symmetric Hausdorff distance.
    """
    # Ensure rank-4 tensors
    y_true = tf.expand_dims(y_true, axis=0) if len(tf.shape(y_true)) == 3 else y_true
    y_true = tf.cast(y_true, tf.float32)

    y_pred = tf.cast(y_pred, tf.float32)

    # If y_pred has 3 channels, convert to single channel via argmax or mean
    if tf.shape(y_pred)[-1] == 3:
        y_pred = tf.reduce_mean(y_pred, axis=-1, keepdims=True)

    y_pred = np.squeeze(y_pred)
    y_true = np.squeeze(y_true)

    y_true = y_true.astype(np.bool_)
    y_pred = y_pred.astype(np.bool_)

    true_points = np.argwhere(y_true)
    pred_points = np.argwhere(y_pred)

    if len(true_points) == 0 or len(pred_points) == 0:
        return np.nan  # undefined if no points
    forward_hd = directed_hausdorff(true_points, pred_points)[0]
    backward_hd = directed_hausdorff(pred_points, true_points)[0]

    return max(forward_hd, backward_hd)


def make_prediction(image, mask, index=0):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = f'{OUTPUT_DIR}/{index}'
    os.makedirs(output_path, exist_ok=True)
    save_image(output_path, image, f'image_{index}')
    save_image(output_path, mask.squeeze(), f'actual_mask_{index}')
    result_metrics = {}
    unet = load_saved_unet_model()
    print('UNET')
    result_metrics['UNET'] = evaluate_model('UNET', unet, image, mask, index, output_path)
    unet_gradcam = gradcam(image, unet, 'conv2d_23')
    save_image(output_path, unet_gradcam, f'UNET_gradcam_{index}')
    unet_gradcam_plus_plus = gradcam_plus_plus(image, unet, 'conv2d_23')
    save_image(output_path, unet_gradcam_plus_plus, f'UNET_gradcam_plus_plus_{index}')

    print('==================================================================================================')
    unet_fcc = load_saved_unet_ffc_model()
    print('UNET-FFC')
    result_metrics['UNET-FFC'] = evaluate_model('UNET-FFC', unet_fcc, image, mask, index, output_path)
    unet_fcc_gradcam = gradcam(image, unet_fcc, 'conv2d_15')
    save_image(output_path, unet_fcc_gradcam, f'UNET-FFC_gradcam_{index}')
    unet_fcc_gradcam_plus_plus = gradcam_plus_plus(image, unet_fcc, 'conv2d_15')
    save_image(output_path, unet_fcc_gradcam_plus_plus, f'UNET-FFC_gradcam_plus_plus_{index}')

    print('==================================================================================================')
    unet_vgg16 = load_saved_unet_VGG16_model()
    print('UNET-VGG16')
    result_metrics['UNET-VGG16'] = evaluate_model('UNET-VGG16', unet_vgg16, image, mask, index, output_path)
    unet_vgg16_gradcam = gradcam(image, unet_vgg16, 'conv2d_33')
    save_image(output_path, unet_vgg16_gradcam, f'UNET-VGG16_gradcam_{index}')
    unet_vgg16_gradcam_plus_plus = gradcam_plus_plus(image, unet_vgg16, 'conv2d_33')
    save_image(output_path, unet_vgg16_gradcam_plus_plus, f'UNET-VGG16_gradcam_plus_plus_{index}')

    print('==================================================================================================')
    unet_resnet50 = load_saved_unet_ResNet50_model()
    print('UNET-ResNet50')
    result_metrics['UNET-ResNet50'] = evaluate_model('UNET-ResNet50', unet_resnet50, image, mask, index, output_path)
    unet_resnet50_gradcam = gradcam(image, unet_resnet50, 'conv2d_42')
    save_image(output_path, unet_resnet50_gradcam, f'UNET-ResNet50_gradcam_{index}')
    unet_resnet50_gradcam_plus_plus = gradcam_plus_plus(image, unet_resnet50, 'conv2d_42')
    save_image(output_path, unet_resnet50_gradcam_plus_plus, f'UNET-ResNet50_gradcam_plus_plus_{index}')

    print('==================================================================================================')
    unet_mobilenetv2 = load_saved_unet_MobileNetV2_model()
    print('UNET-MobileNetV2')
    result_metrics['UNET-MobileNetV2'] = evaluate_model('UNET-MobileNetV2', unet_mobilenetv2, image, mask, index, output_path)
    unet_mobilenetv2_gradcam = gradcam(image, unet_mobilenetv2, 'conv2d_10')
    save_image(output_path, unet_mobilenetv2_gradcam, f'UNET-MobileNetV2_gradcam_{index}')
    unet_mobilenetv2_gradcam_plus_plus = gradcam_plus_plus(image, unet_mobilenetv2, 'conv2d_10')
    save_image(output_path, unet_mobilenetv2_gradcam_plus_plus, f'UNET-MobileNetV2_gradcam_plus_plus_{index}')

    print('==================================================================================================')
    unet_plus_plus = load_saved_unet_plus_plus_model()
    print('UNET++')
    result_metrics['UNET++'] = evaluate_model('UNET++', unet_plus_plus, image, mask, index, output_path)
    unet_plus_plus_gradcam = gradcam(image, unet_plus_plus, 'conv2d_29')
    save_image(output_path, unet_plus_plus_gradcam, f'UNET++_gradcam_{index}')
    unet_plus_plus_gradcam_plus_plus = gradcam_plus_plus(image, unet_plus_plus, 'conv2d_29')
    save_image(output_path, unet_plus_plus_gradcam_plus_plus, f'UNET++_gradcam_plus_plus_{index}')

    print('==================================================================================================')
    segnet = load_saved_segnet_model()
    print('SegNet')
    result_metrics['SegNet'] = evaluate_model('SegNet', segnet, image, mask, index, output_path)
    segnet_gradcam = gradcam(image, segnet, 'conv2d_14')
    save_image(output_path, segnet_gradcam, f'SegNet_gradcam_{index}')
    segnet_gradcam_plus_plus = gradcam_plus_plus(image, segnet, 'conv2d_14')
    save_image(output_path, segnet_gradcam_plus_plus, f'SegNet_gradcam_plus_plus_{index}')

    print('==================================================================================================')
    segnet_vgg16 = load_saved_segnet_VGG16_model()
    print('SegNet-Vgg16')
    result_metrics['SegNet-Vgg16'] = evaluate_model('SegNet-Vgg16', segnet_vgg16, image, mask, index, output_path)
    segnet_vgg16_gradcam = gradcam(image, segnet_vgg16, 'conv2d_9')
    save_image(output_path, segnet_vgg16_gradcam, f'SegNet-Vgg16_gradcam_{index}')
    segnet_vgg16_gradcam_plus_plus = gradcam_plus_plus(image, segnet_vgg16, 'conv2d_9')
    save_image(output_path, segnet_vgg16_gradcam_plus_plus, f'SegNet-Vgg16_gradcam_plus_plus_{index}')

    print('==================================================================================================')
    res_unet_plus_plus = load_saved_res_unet_plus_plus_model()
    print('ResUNET++')
    result_metrics['ResUNET++'] = evaluate_model('ResUNET++', res_unet_plus_plus, image, mask, index, output_path)
    res_unet_plus_plus_gradcam = gradcam(image, res_unet_plus_plus, 'conv2d_40')
    save_image(output_path,res_unet_plus_plus_gradcam, f'ResUNET++_gradcam_{index}')
    res_unet_plus_plus_gradcam_plus_plus = gradcam_plus_plus(image, res_unet_plus_plus, 'conv2d_40')
    save_image(output_path, res_unet_plus_plus_gradcam_plus_plus, f'ResUNET++_gradcam_plus_plus_{index}')

    print('==================================================================================================')
    deeplabv3_plus = load_saved_deeplabv3_plus_model()
    print('DeepLabV3+')
    result_metrics['DeepLabV3+'] = evaluate_model('DeepLabV3+', deeplabv3_plus, image, mask, index, output_path)
    deeplabv3_plus_gradcam = gradcam(image, deeplabv3_plus, 'conv2d_49')
    save_image(output_path, deeplabv3_plus_gradcam, f'DeepLabV3+_gradcam_{index}')
    deeplabv3_plus_gradcam_plus_plus = gradcam_plus_plus(image, deeplabv3_plus, 'conv2d_49')
    save_image(output_path, deeplabv3_plus_gradcam_plus_plus, f'DeepLabV3+_gradcam_plus_plus_{index}')

    result_file = f'{output_path}/result_metrics_{index}.csv'
    with open(result_file, 'w') as file:
        file.write('Model Name,Dice Coefficient,IoU Score,Pixel Accuracy,Precision,Recall,Boundary IoU,'
                   'Hausdorff Distance,Inference Time (ms)\n')
        for model_name, metrics in result_metrics.items():
            file.write(f'{model_name},'
                       f'{metrics["Dice Coefficient"]},'
                       f'{metrics["IoU Score"]},'
                       f'{metrics["Pixel Accuracy"]},'
                       f'{metrics["Precision"]},'
                       f'{metrics["Recall"]},'
                       f'{metrics["Boundary IoU"]},'
                       f'{metrics["Hausdorff Distance"]},'
                       f'{metrics["Inference Time (ms)"]}\n')
    return result_metrics

def evaluate_model(model_name, model, image, mask, index, output_path):
    image = np.expand_dims(image, 0)
    y_true = mask
    y_pred = model.predict(image)
    save_image(output_path, y_pred.squeeze(), f'{model_name}_{index}')

    return evaluate_segmentation(y_true, y_pred[0], model=model, sample=image)

# Both require binary masks (0 or 1) as NumPy arrays.
# Typically, you'd threshold your model output predictions (e.g., pred_mask = (pred_prob > 0.5)).
# Hausdorff distance can be sensitive to noise/outliers;
# consider using a percentile Hausdorff (e.g., 95th percentile) if needed.

def load_image(path: str, size=(512, 512),  color_mode = "rgb"):
    img = load_img(path, color_mode=color_mode)
    img_array = img_to_array(img)
    normalized_img_array = img_array/255.
    formatted_img = tf.image.resize(normalized_img_array, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return formatted_img

# if __name__=="__main__":
#     image_path = "../input/samples/segnet_512/images/DJI_20250324092908_0001_V.jpg"
#     mask_path = "../input/samples/segnet_512/masks/DJI_20250324092908_0001_V.jpg"
#     image = load_image(image_path)
#     mask = load_image(image_path, color_mode="grayscale")
#     print(f'image shape:{image.shape}')
#     print(f'mask shape:{mask.shape}')
#
#     make_prediction(image, mask)

if __name__=="__main__":
    path = "../input/updated_samples/segnet_512/images"
    image_count = 20
    (images, masks) = load_dataset(path,
                                   size=(512, 512),
                                   file_extension="jpg",
                                   channels=['RED', 'GREEN', 'BLUE'],
                                   image_count=image_count)
    print(f'images shape:{images.shape}')
    print(f'masks shape:{masks.shape}')
    for i in range(image_count):
        make_prediction(images[i], masks[i], i)

    # for image in images:
    #     print(f'image shape:{image.shape}')
    #     print(f'mask shape:{mask.shape}')
    #     make_prediction(image, mask)

# if __name__=="__main__":
#     image_path = "../input/samples/segnet_512/images/DJI_20250324092908_0001_V.jpg"
#     mask_path = "../input/samples/segnet_512/masks/DJI_20250324092908_0001_V.jpg"
#     image = load_image(image_path)
#     mask = load_image(image_path, color_mode="grayscale")
#     print(f'image shape:{image.shape}')
#     print(f'mask shape:{mask.shape}')
#
#     image = np.expand_dims(image, 0)
#
#     model = load_saved_unet_model()
#
#     y_true = mask.numpy()
#     y_pred = model.predict(image)  # shape: (H, W)
#     results = evaluate_segmentation(y_true, y_pred, model=model, sample=image)
#     print(results)
#
#     # y_true and y_pred are binary masks (H, W)
#     b_iou = boundary_iou(y_true, y_pred)
#     hd = hausdorff_distance(y_true, y_pred)
#
#     print(f"Boundary IoU: {b_iou:.4f}")
#     print(f"Hausdorff Distance: {hd:.4f} pixels")