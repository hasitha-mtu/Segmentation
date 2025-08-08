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

OUTPUT_DIR = "C:\\Users\AdikariAdikari\OneDrive - Munster Technological University\ModelResults\Segmentation\\05_08_2025"


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
    return result_metrics

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
    train_all_models()
    path = "../input/dataset/validation/images"
    image_count = 25
    (images, masks) = load_dataset(path,
                                   size=(512, 512),
                                   file_extension="jpg",
                                   channels=['RED', 'GREEN', 'BLUE'],
                                   image_count=image_count)
    print(f'images shape:{images.shape}')
    print(f'masks shape:{masks.shape}')

    for i in range(image_count):
        image = images[i]
        mask = masks[i]
        make_prediction(images[i], masks[i], i)


import os
import pandas as pd
import numpy as np

if __name__=="__main__":
    root_path = "../output/17_07_2025"
    paths = os.listdir(root_path)
    print(f'paths : {paths}')
    results = []
    for dir_name in paths:
        result_path = f"{root_path}/{dir_name}/result_metrics_{dir_name}.csv"
        result_df = pd.read_csv(result_path)
        results.append(result_df)
    numeric_dfs = [df.drop(columns=['Model Name']) for df in results]
    mean_array = np.mean([df.values for df in numeric_dfs], axis=0)
    mean_df = pd.DataFrame(mean_array, columns=numeric_dfs[0].columns)
    mean_df.insert(0, 'Model Name', results[0]['Model Name'].values)
    mean_df.to_csv(f"{root_path}/model_results.csv", index=False)
    print(mean_df)