import tensorflow as tf
import numpy as np
import time
import cv2
from scipy.spatial.distance import directed_hausdorff
from skimage import measure
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
from models.common_utils.overlay import overlay_mask

from gradcam_keras import gradcam,gradcam_plus_plus

from models.train import  train_all_models

# OUTPUT_DIR = "C:\\Users\AdikariAdikari\OneDrive - Munster Technological University\ModelResults\Segmentation\\05_08_2025"
OUTPUT_DIR = "C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\output\\07_08_2025"


# Measure Inference Time (optional)
def measure_inference_time(model, input_sample):
    start_time = time.time()
    _ = model.predict(input_sample)
    end_time = time.time()
    return (end_time - start_time) * 1000  # ms


# Combine All Metrics
def evaluate_segmentation(y_true, y_pred, model=None, sample=None):

    precision, recall, dice, iou, pixel_accuracy = calculate_matrices(y_true, y_pred)

    metrics = {
        'Dice Coefficient': float(dice),
        'IoU Score': float(iou),
        'Pixel Accuracy': float(pixel_accuracy),
        'Precision': float(precision),
        'Recall': float(recall)
    }

    if model and sample is not None:
        metrics['Inference Time (ms)'] = float(measure_inference_time(model, sample))

    hd = calculate_symmetric_hausdorff_distance(y_true, y_pred)
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



def make_prediction(image, mask, index=0):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = f'{OUTPUT_DIR}/{index}'
    os.makedirs(output_path, exist_ok=True)
    save_image(output_path, image, f'image_{index}')
    save_image(output_path, mask.squeeze(), f'actual_mask_{index}')

    print('UNET')
    unet_metrics = {}
    unet_output_path = f'{output_path}/unet'
    unet_metrics[f'UNET_Adam_{True}'] = make_prediction_model('UNET', 'Adam', True,
                                                              unet_output_path, 'conv2d_23', image,
                                                              mask, index)
    unet_metrics[f'UNET_Adam_{False}'] = make_prediction_model('UNET', 'Adam', False,
                                                               unet_output_path, 'conv2d_23', image,
                                                               mask, index)
    unet_metrics[f'UNET_AdamW_{True}'] = make_prediction_model('UNET', 'AdamW', True,
                                                               unet_output_path, 'conv2d_23', image,
                                                               mask, index)
    unet_metrics[f'UNET_AdamW_{False}'] = make_prediction_model('UNET', 'AdamW', False,
                                                                unet_output_path, 'conv2d_23', image,
                                                                mask, index)
    save_matrics(unet_output_path, unet_metrics, index)

    print('==================================================================================================')
    print('UNET-FFC')
    unet_ffc_metrics = {}
    unet_ffc_output_path = f'{output_path}/unet_ffc'
    unet_ffc_metrics[f'UNET-FFC_Adam_{True}'] = make_prediction_model('UNET-FFC', 'Adam', True,
                                                              unet_ffc_output_path, 'conv2d_15', image,
                                                                      mask, index)
    unet_ffc_metrics[f'UNET-FFC_Adam_{False}'] = make_prediction_model('UNET-FFC', 'Adam', False,
                                                               unet_ffc_output_path, 'conv2d_15', image,
                                                                       mask, index)
    unet_ffc_metrics[f'UNET-FFC_AdamW_{True}'] = make_prediction_model('UNET-FFC', 'AdamW', True,
                                                               unet_ffc_output_path, 'conv2d_15', image,
                                                                       mask, index)
    unet_ffc_metrics[f'UNET-FFC_AdamW_{False}'] = make_prediction_model('UNET-FFC', 'AdamW', False,
                                                                unet_ffc_output_path, 'conv2d_15', image,
                                                                        mask, index)
    save_matrics(unet_ffc_output_path, unet_ffc_metrics, index)

    print('==================================================================================================')
    print('UNET-VGG16')
    unet_vgg16_metrics = {}
    unet_vgg16_output_path = f'{output_path}/unet_vgg16'
    unet_vgg16_metrics[f'UNET-VGG16_Adam_{True}'] = make_prediction_model('UNET-VGG16', 'Adam', True,
                                                                      unet_vgg16_output_path, 'conv2d_33',
                                                                      image, mask, index)
    unet_vgg16_metrics[f'UNET-VGG16_Adam_{False}'] = make_prediction_model('UNET-VGG16', 'Adam', False,
                                                                       unet_vgg16_output_path, 'conv2d_33',
                                                                       image, mask, index)
    unet_vgg16_metrics[f'UNET-VGG16_AdamW_{True}'] = make_prediction_model('UNET-VGG16', 'AdamW', True,
                                                                       unet_vgg16_output_path, 'conv2d_33',
                                                                       image, mask, index)
    unet_vgg16_metrics[f'UNET-VGG16_AdamW_{False}'] = make_prediction_model('UNET-VGG16', 'AdamW', False,
                                                                        unet_vgg16_output_path, 'conv2d_33',
                                                                        image, mask, index)
    save_matrics(unet_vgg16_output_path, unet_vgg16_metrics, index)

    print('==================================================================================================')
    print('UNET-ResNet50')
    unet_resnet50_metrics = {}
    unet_resnet50_output_path = f'{output_path}/unet_resnet50'
    unet_resnet50_metrics[f'UNET-ResNet50_Adam_{True}'] = make_prediction_model('UNET-ResNet50', 'Adam', True,
                                                                          unet_resnet50_output_path, 'conv2d_42',
                                                                          image, mask, index)
    unet_resnet50_metrics[f'UNET-ResNet50_Adam_{False}'] = make_prediction_model('UNET-ResNet50', 'Adam', False,
                                                                           unet_resnet50_output_path, 'conv2d_42',
                                                                           image, mask, index)
    unet_resnet50_metrics[f'UNET-ResNet50_AdamW_{True}'] = make_prediction_model('UNET-ResNet50', 'AdamW', True,
                                                                           unet_resnet50_output_path, 'conv2d_42',
                                                                           image, mask, index)
    unet_resnet50_metrics[f'UNET-ResNet50_AdamW_{False}'] = make_prediction_model('UNET-ResNet50', 'AdamW', False,
                                                                            unet_resnet50_output_path, 'conv2d_42',
                                                                            image, mask, index)
    save_matrics(unet_resnet50_output_path, unet_resnet50_metrics, index)

    print('==================================================================================================')
    print('UNET-MobileNetV2')
    unet_mobilenetv2_metrics = {}
    unet_mobilenetv2_output_path = f'{output_path}/unet_mobilenetv2'
    unet_mobilenetv2_metrics[f'UNET-MobileNetV2_Adam_{True}'] = make_prediction_model('UNET-MobileNetV2', 'Adam', True,
                                                                             unet_mobilenetv2_output_path, 'conv2d_10',
                                                                             image, mask, index)
    unet_mobilenetv2_metrics[f'UNET-MobileNetV2_Adam_{False}'] = make_prediction_model('UNET-MobileNetV2', 'Adam', False,
                                                                              unet_mobilenetv2_output_path, 'conv2d_10',
                                                                              image, mask, index)
    unet_mobilenetv2_metrics[f'UNET-MobileNetV2_AdamW_{True}'] = make_prediction_model('UNET-MobileNetV2', 'AdamW', True,
                                                                              unet_mobilenetv2_output_path, 'conv2d_10',
                                                                              image, mask, index)
    unet_mobilenetv2_metrics[f'UNET-MobileNetV2_AdamW_{False}'] = make_prediction_model('UNET-MobileNetV2', 'AdamW', False,
                                                                               unet_mobilenetv2_output_path, 'conv2d_10',
                                                                               image, mask, index)
    save_matrics(unet_mobilenetv2_output_path, unet_mobilenetv2_metrics, index)

    print('==================================================================================================')
    print('UNET++')
    unet_plus_plus_metrics = {}
    unet_plus_plus_output_path = f'{output_path}/unet_plus_plus'
    unet_plus_plus_metrics[f'UNET++_Adam_{True}'] = make_prediction_model('UNET++', 'Adam', True,
                                                                                unet_plus_plus_output_path,
                                                                                'conv2d_29',
                                                                                image, mask, index)
    unet_plus_plus_metrics[f'UNET++_Adam_{False}'] = make_prediction_model('UNET++', 'Adam', False,
                                                                                 unet_plus_plus_output_path,
                                                                                 'conv2d_29',
                                                                                 image, mask, index)
    unet_plus_plus_metrics[f'UNET++_AdamW_{True}'] = make_prediction_model('UNET++', 'AdamW', True,
                                                                                 unet_plus_plus_output_path,
                                                                                 'conv2d_29',
                                                                                 image, mask, index)
    unet_plus_plus_metrics[f'UNET++_AdamW_{False}'] = make_prediction_model('UNET++', 'AdamW', False,
                                                                                  unet_plus_plus_output_path,
                                                                                  'conv2d_29',
                                                                                  image, mask, index)
    save_matrics(unet_plus_plus_output_path, unet_plus_plus_metrics, index)

    print('==================================================================================================')
    print('SegNet')
    segnet_metrics = {}
    segnet_output_path = f'{output_path}/segnet'
    segnet_metrics[f'SegNet_Adam_{True}'] = make_prediction_model('SegNet', 'Adam', True,
                                                                                      segnet_output_path,
                                                                                      'conv2d_14',
                                                                                      image, mask, index)
    segnet_metrics[f'SegNet_Adam_{False}'] = make_prediction_model('SegNet', 'Adam',
                                                                                       False,
                                                                                       segnet_output_path,
                                                                                       'conv2d_14',
                                                                                       image, mask, index)
    segnet_metrics[f'SegNet_AdamW_{True}'] = make_prediction_model('SegNet', 'AdamW',
                                                                                       True,
                                                                                       segnet_output_path,
                                                                                       'conv2d_14',
                                                                                       image, mask, index)
    segnet_metrics[f'SegNet_AdamW_{False}'] = make_prediction_model('SegNet', 'AdamW',
                                                                                        False,
                                                                                        segnet_output_path,
                                                                                        'conv2d_14',
                                                                                        image, mask, index)
    save_matrics(segnet_output_path, segnet_metrics, index)

    print('==================================================================================================')
    print('SegNet-Vgg16')
    segnet_vgg16_metrics = {}
    segnet_vgg16_output_path = f'{output_path}/segnet_vgg16'
    segnet_vgg16_metrics[f'SegNet-Vgg16_Adam_{True}'] = make_prediction_model('SegNet-Vgg16', 'Adam',
                                                                  True,
                                                                  segnet_vgg16_output_path,
                                                                  'conv2d_9',
                                                                  image, mask, index)
    segnet_vgg16_metrics[f'SegNet-Vgg16_Adam_{False}'] = make_prediction_model('SegNet-Vgg16', 'Adam',
                                                                   False,
                                                                   segnet_vgg16_output_path,
                                                                   'conv2d_9',
                                                                   image, mask, index)
    segnet_vgg16_metrics[f'SegNet-Vgg16_AdamW_{True}'] = make_prediction_model('SegNet-Vgg16', 'AdamW',
                                                                   True,
                                                                   segnet_vgg16_output_path,
                                                                   'conv2d_9',
                                                                   image, mask, index)
    segnet_vgg16_metrics[f'SegNet-Vgg16_AdamW_{False}'] = make_prediction_model('SegNet-Vgg16', 'AdamW',
                                                                    False,
                                                                    segnet_vgg16_output_path,
                                                                    'conv2d_9',
                                                                    image, mask, index)
    save_matrics(segnet_vgg16_output_path, segnet_vgg16_metrics, index)

    print('==================================================================================================')
    print('ResUNET++')
    res_unet_plus_plus_metrics = {}
    res_unet_plus_plus_output_path = f'{output_path}/res_unet_plus_plus'
    res_unet_plus_plus_metrics[f'ResUNET++_Adam_{True}'] = make_prediction_model('ResUNET++', 'Adam',
                                                                  True,
                                                                  res_unet_plus_plus_output_path,
                                                                  'conv2d_40',
                                                                  image, mask, index)
    res_unet_plus_plus_metrics[f'ResUNET++_Adam_{False}'] = make_prediction_model('ResUNET++', 'Adam',
                                                                   False,
                                                                   res_unet_plus_plus_output_path,
                                                                   'conv2d_40',
                                                                   image, mask, index)
    res_unet_plus_plus_metrics[f'ResUNET++_AdamW_{True}'] = make_prediction_model('ResUNET++', 'AdamW',
                                                                   True,
                                                                   res_unet_plus_plus_output_path,
                                                                   'conv2d_40',
                                                                   image, mask, index)
    res_unet_plus_plus_metrics[f'ResUNET++_AdamW_{False}'] = make_prediction_model('ResUNET++', 'AdamW',
                                                                    False,
                                                                    res_unet_plus_plus_output_path,
                                                                    'conv2d_40',
                                                                    image, mask, index)
    save_matrics(res_unet_plus_plus_output_path, res_unet_plus_plus_metrics, index)

    print('==================================================================================================')
    print('DeepLabV3+')
    deeplabv3_plus_metrics = {}
    deeplabv3_plus_output_path = f'{output_path}/deeplabv3_plus'
    deeplabv3_plus_metrics[f'DeepLabV3+_Adam_{True}'] = make_prediction_model('DeepLabV3+', 'Adam',
                                                                  True,
                                                                  deeplabv3_plus_output_path,
                                                                  'conv2d_49',
                                                                  image, mask, index)
    deeplabv3_plus_metrics[f'DeepLabV3+_Adam_{False}'] = make_prediction_model('DeepLabV3+', 'Adam',
                                                                   False,
                                                                   deeplabv3_plus_output_path,
                                                                   'conv2d_49',
                                                                   image, mask, index)
    deeplabv3_plus_metrics[f'DeepLabV3+_AdamW_{True}'] = make_prediction_model('DeepLabV3+', 'AdamW',
                                                                   True,
                                                                   deeplabv3_plus_output_path,
                                                                   'conv2d_49',
                                                                   image, mask, index)
    deeplabv3_plus_metrics[f'DeepLabV3+_AdamW_{False}'] = make_prediction_model('DeepLabV3+', 'AdamW',
                                                                    False,
                                                                    deeplabv3_plus_output_path,
                                                                    'conv2d_49',
                                                                    image, mask, index)
    save_matrics(deeplabv3_plus_output_path, deeplabv3_plus_metrics, index)

def make_prediction_model(name, optimizer, enable_clrs, output_path, layer_name, image, mask, index):
    print(f'make_prediction_model|name: {name}')
    print(f'make_prediction_model|optimizer: {optimizer}')
    print(f'make_prediction_model|enable_clrs: {enable_clrs}')
    print(f'make_prediction_model|output_path: {output_path}')
    print(f'make_prediction_model|layer_name: {layer_name}')
    model_name = f'{name}_{optimizer}_CLRS_{enable_clrs}'
    model = load_saved_unet_model(optimizer, enable_clrs)
    metrics = evaluate_model(model_name, model, image, mask, index, output_path)
    model_gradcam = gradcam(image, model, layer_name)
    save_image(output_path, model_gradcam, f'{model_name}_gradcam_{index}')
    model_gradcam_plus_plus = gradcam_plus_plus(image, model, layer_name)
    save_image(output_path, model_gradcam_plus_plus, f'{model_name}_gradcam_plus_plus_{index}')
    return metrics

def save_matrics(output_path, result_metrics, index):
    result_file = f'{output_path}/result_metrics_{index}.csv'
    with open(result_file, 'w') as file:
        file.write('Model Name,Dice Coefficient,IoU Score,Pixel Accuracy,Precision,Recall,'
                   'Hausdorff Distance,Inference Time (ms)\n')
        for model_name, metrics in result_metrics.items():
            file.write(f'{model_name},'
                       f'{metrics["Dice Coefficient"]},'
                       f'{metrics["IoU Score"]},'
                       f'{metrics["Pixel Accuracy"]},'
                       f'{metrics["Precision"]},'
                       f'{metrics["Recall"]},'
                       f'{metrics["Hausdorff Distance"]},'
                       f'{metrics["Inference Time (ms)"]}\n')

def evaluate_model(model_name, model, image, mask, index, output_path):
    image = np.expand_dims(image, 0)
    y_true = mask
    y_pred = model.predict(image)

    print(f'images shape:{images.shape}')
    print(f'masks shape:{masks.shape}')
    print(f'y_pred shape:{y_pred.shape}')

    save_image(output_path, y_pred.squeeze(), f'{model_name}_{index}')

    overlaid_image = overlay_mask(image[0], y_pred[0])
    save_image(output_path, overlaid_image, f'overlaid_{model_name}_{index}')

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


def calculate_matrices(y_true, y_pred):
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    mask = (y_true != 0.0).astype(np.float32)

    predicted_binary_mask = (y_pred >= 0.5).astype(bool)

    y_true_bool = y_true.astype(bool)
    predicted_bool = predicted_binary_mask.astype(bool)
    eval_mask_bool = mask.astype(bool)

    # True Positives (TP): Pixels correctly predicted as positive AND within the evaluation mask
    true_positives = np.sum(np.logical_and(predicted_bool, y_true_bool, eval_mask_bool))

    # False Positives (FP): Pixels incorrectly predicted as positive AND within the evaluation mask
    false_positives = np.sum(np.logical_and(predicted_bool, np.logical_not(y_true_bool), eval_mask_bool))

    # False Negatives (FN): Pixels where ground truth is 1 BUT prediction is 0 (missed positives)
    false_negatives = np.sum(np.logical_and(eval_mask_bool == 1, predicted_bool == 0))

    if (true_positives + false_positives) == 0:
        precision = 0.0
    else:
        precision = true_positives / (true_positives + false_positives)

    if (true_positives + false_negatives) == 0:
        recall = 0.0  # If there are no actual positives, recall is 0.0 (or sometimes undefined/1.0 depending on convention)
    else:
        recall = true_positives / (true_positives + false_negatives)

    if ((2 * true_positives) + false_positives + false_negatives) == 0:
        # If both masks are entirely empty (no positive pixels in either GT or prediction),
        # Dice is usually considered 1.0 (perfect agreement on "nothing").
        dice = 1.0
    else:
        dice = (2.0 * true_positives) / ((2 * true_positives) + false_positives + false_negatives)

    # Calculate True Positives (TP) - Intersection
    intersection = np.sum(np.logical_and(eval_mask_bool, predicted_bool))
    # Calculate the Union
    # Union = TP + FP + FN
    # This is equivalent to np.sum(np.logical_or(eval_mask_bool, predicted_bool))
    union = intersection + false_positives + false_negatives
    if union == 0:
        # If both masks are entirely empty (no positive pixels in either GT or prediction),
        # IoU is usually considered 1.0 (perfect agreement on "nothing").
        iou = 1.0
    else:
        iou = float(intersection) / float(union)

    total_pixels = y_true.size
    true_negatives = np.sum(np.logical_and(eval_mask_bool == 0, predicted_bool == 0))

    pixel_accuracy = (true_positives + true_negatives)/total_pixels

    print(f"  True Positives (TP): {true_positives}")
    print(f"  False Negatives (FN): {false_negatives}")
    print(f"  False Positives (TP): {false_positives}")
    print(f"  True Negatives (FN): {true_negatives}")
    print(f"  Total Pixels: {total_pixels}")

    print(f'precision_recall_updated|precision:{precision}')
    print(f'precision_recall_updated|recall:{recall}')
    print(f'precision_recall_updated|dice:{dice}')
    print(f'precision_recall_updated|iou:{iou}')
    print(f'precision_recall_updated|pixel_accuracy:{pixel_accuracy}')

    return precision, recall, dice, iou, pixel_accuracy


def calculate_symmetric_hausdorff_distance(y_true, y_pred):
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    mask = (y_true != 0.0).astype(np.float32)

    y_true = y_true * mask
    y_pred = y_pred * mask

    """
    Calculates the symmetric Hausdorff Distance between two binary masks.
    This function assumes the inputs are binary (0/1) masks.
    """
    # Ensure inputs are boolean for find_contours
    mask1_bool = y_true > 0
    mask2_bool = y_pred > 0

    # Find contours
    # find_contours returns (row, col) coordinates. We want (x, y) for distance
    # For image coordinates, row is y, col is x.
    contours1 = measure.find_contours(mask1_bool, 0.5)  # 0.5 is the level to find contours at
    contours2 = measure.find_contours(mask2_bool, 0.5)

    if not contours1 or not contours2:
        # Handle cases where one or both masks are empty
        if not contours1 and not contours2:
            return 0.0  # Both empty, distance is 0
        else:
            return float('inf')  # One is empty, infinite distance

    # Concatenate all contour segments into single arrays of points
    points1 = np.concatenate(contours1, axis=0)
    points2 = np.concatenate(contours2, axis=0)

    # Convert (row, col) to (x, y) if necessary for consistency, though
    # for Euclidean distance it usually doesn't matter for 2D.
    # SciPy's directed_hausdorff expects (M, N) arrays where M is number of points, N is dimensions
    # Our contours are already (num_points, 2)

    # Calculate directed Hausdorff distances
    # directed_hausdorff returns (distance, (idx_u, idx_v))
    d_ab = directed_hausdorff(points1, points2)[0]
    print(f'calculate_symmetric_hausdorff_distance|d_ab:{d_ab}')
    d_ba = directed_hausdorff(points2, points1)[0]
    print(f'calculate_symmetric_hausdorff_distance|d_ba:{d_ba}')

    # Symmetric Hausdorff Distance is the maximum of the two directed distances
    symmetric_hd = max(d_ab, d_ba)
    print(f'calculate_symmetric_hausdorff_distance|symmetric_hd:{symmetric_hd}')
    return symmetric_hd


if __name__=="__main__":
    path = "../input/updated_samples/segnet_512/images"
    image_count = 10
    (images, masks) = load_dataset(path,
                                   size=(512, 512),
                                   file_extension="png",
                                   channels=['RED', 'GREEN', 'BLUE'],
                                   image_count=image_count)
    print(f'images shape:{images.shape}')
    print(f'masks shape:{masks.shape}')

    for i in range(image_count):
        image = images[i]
        mask = masks[i]
        make_prediction(images[i], masks[i], i)

