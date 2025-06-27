import os
import numpy as np
import tensorflow as tf
import cv2
import time
from scipy.ndimage import binary_erosion

from unet_wsl.train_model import load_saved_model as load_saved_unet_model
from unet_ffc.train_model import load_saved_model as load_saved_unet_ffc_model
from unet_VGG16.train_model import load_saved_model as load_saved_unet_VGG16_model
from unet_ResNet50.train_model import load_saved_model as load_saved_unet_ResNet50_model
from unet_MobileNetV2.train_model import load_saved_model as load_saved_unet_MobileNetV2_model
from unet_plus_plus.train_model import load_saved_model as load_saved_unet_plus_plus_model
from segnet.train_model import load_saved_model as load_saved_segnet_model
from segnet_VGG16.train_model import load_saved_model as load_saved_segnet_VGG16_model
from res_unet_plus_plus.train_model import load_saved_model as load_saved_res_unet_plus_plus_model
from deeplabv3_plus.train_model import load_saved_model as load_saved_deeplabv3_plus_model

from models.common_utils.images import save_image
from models.generator import load_images

OUTPUT_DIR = "C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\output"

# === Metric Functions with Supervision Mask ===
def masked_dice(y_true, y_pred, mask, smooth=1e-6):
    y_pred = np.argmax(y_pred, axis=-1)[0]
    print(f'masked_dice|y_true shape:{y_true.shape}')
    print(f'masked_dice|y_pred shape:{y_pred.shape}')
    print(f'masked_dice|mask shape:{mask.shape}')
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    mask = tf.cast(mask, tf.float32)
    y_true = y_true * mask
    y_pred = y_pred * mask
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def masked_iou(y_true, y_pred, mask, smooth=1e-6):
    y_pred = np.argmax(y_pred, axis=-1)[0]
    print(f'masked_iou|y_true shape:{y_true.shape}')
    print(f'masked_iou|y_pred shape:{y_pred.shape}')
    print(f'masked_iou|mask shape:{mask.shape}')
    y_true = tf.cast(y_true, tf.float32) * mask
    y_pred = tf.cast(y_pred > 0.5, tf.float32) * mask
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

def masked_precision(y_true, y_pred, mask, smooth=1e-6):
    y_pred = np.argmax(y_pred, axis=-1)[0]
    print(f'masked_precision|y_true shape:{y_true.shape}')
    print(f'masked_precision|y_pred shape:{y_pred.shape}')
    print(f'masked_precision|mask shape:{mask.shape}')
    y_true = tf.cast(y_true, tf.float32) * mask
    y_pred = tf.cast(y_pred > 0.5, tf.float32) * mask
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    return (tp + smooth) / (tp + fp + smooth)

def masked_recall(y_true, y_pred, mask, smooth=1e-6):
    y_pred = np.argmax(y_pred, axis=-1)[0]
    print(f'masked_recall|y_true shape:{y_true.shape}')
    print(f'masked_recall|y_pred shape:{y_pred.shape}')
    print(f'masked_recall|mask shape:{mask.shape}')
    y_true = tf.cast(y_true, tf.float32) * mask
    y_pred = tf.cast(y_pred > 0.5, tf.float32) * mask
    tp = tf.reduce_sum(y_true * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    return (tp + smooth) / (tp + fn + smooth)

def masked_boundary_iou(y_true, y_pred, mask, dilation_ratio=0.02):
    y_pred = np.argmax(y_pred, axis=-1)[0]
    print(f'masked_boundary_iou|y_true shape:{y_true.shape}')
    print(f'masked_boundary_iou|y_pred shape:{y_pred.shape}')
    print(f'masked_boundary_iou|mask shape:{mask.shape}')
    def erode_boundary(mask_np):
        h, w = mask_np.shape
        diag_len = np.sqrt(h ** 2 + w ** 2)
        erosion_radius = max(1, int(dilation_ratio * diag_len))
        structure = np.ones((erosion_radius, erosion_radius), dtype=bool)
        return binary_erosion(mask_np, structure=structure)

    y_true_np = np.squeeze(y_true).astype(np.uint8)
    y_pred_np = (np.squeeze(y_pred) > 0.5).astype(np.uint8)
    mask_np = np.squeeze(mask).astype(np.uint8)

    y_true_b = erode_boundary(y_true_np * mask_np)
    y_pred_b = erode_boundary(y_pred_np * mask_np)
    intersection = np.sum(y_true_b & y_pred_b)
    union = np.sum((y_true_b | y_pred_b))
    return (intersection + 1e-6) / (union + 1e-6)

def masked_hausdorff_distance(y_true, y_pred, mask):
    y_pred = np.argmax(y_pred, axis=-1)[0]
    y_true = (tf.cast(y_true > 0.0, tf.float32) * mask)[0, ..., 0]
    y_pred = (tf.cast(y_pred > 0.0, tf.float32) * mask)[0, ..., 0]

    print(f'masked_hausdorff_distance|y_true shape:{y_true.shape}')
    print(f'masked_hausdorff_distance|y_pred shape:{y_pred.shape}')
    print(f'masked_hausdorff_distance|mask shape:{mask.shape}')

    coords_true = tf.where(y_true > 0)
    coords_pred = tf.where(y_pred > 0)
    coords_true = tf.cast(coords_true, tf.float32)
    coords_pred = tf.cast(coords_pred, tf.float32)

    if tf.shape(coords_true)[0] == 0 or tf.shape(coords_pred)[0] == 0:
        return tf.constant(0.0)

    dists = tf.norm(tf.expand_dims(coords_true, 1) - tf.expand_dims(coords_pred, 0), axis=-1)
    hd = tf.maximum(tf.reduce_max(tf.reduce_min(dists, axis=1)),
                    tf.reduce_max(tf.reduce_min(dists, axis=0)))
    return hd

# === Utility Function to Load and Process Mask ===
def load_mask(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (512, 512))
    img = (img > 127).astype(np.float32)
    return img[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)

# === Main Evaluation Loop ===
def evaluate_all(y_true, y_pred, mask_sup, model, sample):
    dsc = masked_dice(y_true, y_pred, mask_sup).numpy()
    print(f'evaluate_all|dsc:{dsc}')
    iou = masked_iou(y_true, y_pred, mask_sup).numpy()
    print(f'evaluate_all|iou:{iou}')
    prec = masked_precision(y_true, y_pred, mask_sup).numpy()
    print(f'evaluate_all|prec:{prec}')
    rec = masked_recall(y_true, y_pred, mask_sup).numpy()
    print(f'evaluate_all|rec:{rec}')
    b_iou = masked_boundary_iou(y_true, y_pred, mask_sup)
    print(f'evaluate_all|b_iou:{b_iou}')
    h_dist = masked_hausdorff_distance(y_true, y_pred, mask_sup).numpy()
    print(f'evaluate_all|h_dist:{h_dist}')
    inference_time = float(measure_inference_time(model, sample))
    print(f'evaluate_all|inference_time:{inference_time}')
    metrics ={
        "Dice": dsc,
        "IoU": iou,
        "Precision": prec,
        "Recall": rec,
        "Boundary IoU": b_iou,
        "Hausdorff": h_dist,
        "Inference Time (ms)": inference_time,
    }

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    return metrics


# Measure Inference Time (optional)
def measure_inference_time(model, input_sample):
    start_time = time.time()
    _ = model.predict(input_sample)
    end_time = time.time()
    return (end_time - start_time) * 1000  # ms

def make_prediction(image, mask, supervised_mask, index=0):
    print(f'make_prediction|image shape:{image.shape}')
    print(f'make_prediction|mask shape:{mask.shape}')
    print(f'make_prediction|supervised_mask shape:{supervised_mask.shape}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = f'{OUTPUT_DIR}/{index}'
    os.makedirs(output_path, exist_ok=True)
    save_image(output_path, image, f'image_{index}')
    save_image(output_path, mask.squeeze(), f'actual_mask_{index}')
    result_metrics = {}
    unet = load_saved_unet_model()
    print('UNET')
    result_metrics['UNET'] = evaluate_model('UNET', unet, image, mask, supervised_mask, index, output_path)
    print('==================================================================================================')
    unet_fcc = load_saved_unet_ffc_model()
    print('UNET-FFC')
    result_metrics['UNET-FFC'] = evaluate_model('UNET-FFC', unet_fcc, image, mask, supervised_mask, index, output_path)
    print('==================================================================================================')
    unet_vgg16 = load_saved_unet_VGG16_model()
    print('UNET-VGG16')
    result_metrics['UNET-VGG16'] = evaluate_model('UNET-VGG16', unet_vgg16, image, mask, supervised_mask, index, output_path)
    print('==================================================================================================')
    unet_resnet50 = load_saved_unet_ResNet50_model()
    print('UNET-ResNet50')
    result_metrics['UNET-ResNet50'] = evaluate_model('UNET-ResNet50', unet_resnet50, image, mask, supervised_mask, index, output_path)
    print('==================================================================================================')
    unet_mobilenetv2 = load_saved_unet_MobileNetV2_model()
    print('UNET-MobileNetV2')
    result_metrics['UNET-MobileNetV2'] = evaluate_model('UNET-MobileNetV2', unet_mobilenetv2, image, mask, supervised_mask, index, output_path)
    print('==================================================================================================')
    unet_plus_plus = load_saved_unet_plus_plus_model()
    print('UNET++')
    result_metrics['UNET++'] = evaluate_model('UNET++', unet_plus_plus, image, mask, supervised_mask, index, output_path)
    print('==================================================================================================')
    segnet = load_saved_segnet_model()
    print('SegNet')
    result_metrics['SegNet'] = evaluate_model('SegNet', segnet, image, mask, supervised_mask, index, output_path)
    print('==================================================================================================')
    segnet_vgg16 = load_saved_segnet_VGG16_model()
    print('SegNet-Vgg16')
    result_metrics['SegNet-Vgg16'] = evaluate_model('SegNet-Vgg16', segnet_vgg16, image, mask, supervised_mask, index, output_path)
    print('==================================================================================================')
    res_unet_plus_plus = load_saved_res_unet_plus_plus_model()
    print('ResUNET++')
    result_metrics['ResUNET++'] = evaluate_model('ResUNET++', res_unet_plus_plus, image, mask, supervised_mask, index, output_path)
    print('==================================================================================================')
    deeplabv3_plus = load_saved_deeplabv3_plus_model()
    print('DeepLabV3+')
    result_metrics['DeepLabV3+'] = evaluate_model('DeepLabV3+', deeplabv3_plus, image, mask, supervised_mask, index, output_path)

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

def evaluate_model(model_name, model, image, mask, supervised_mask, index, output_path):
    image = np.expand_dims(image, 0)
    y_true = mask
    y_pred = model.predict(image)
    save_image(output_path, y_pred.squeeze(), f'{model_name}_{index}')
    return evaluate_all(y_true, y_pred, supervised_mask, model=model, sample=image)

if __name__=="__main__":
    path = "../input/samples/segnet_512/images"
    image_count = 1
    (images, masks, supervised_masks) = load_images(path)
    for i in range(image_count):
        np.savetxt('mask.txt', masks[i], fmt='%.4f', delimiter=' ')
        np.savetxt('supervised_mask.txt', supervised_masks[i], fmt='%.4f', delimiter=' ')
        # make_prediction(images[i], masks[i], supervised_masks[i], i)
