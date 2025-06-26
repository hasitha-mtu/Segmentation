import tensorflow as tf
import time
import os
import numpy as np
import tensorflow_addons as tfa

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
from models.common_utils.data import load_dataset

OUTPUT_DIR = "G:\Other computers\My Laptop\GoogleDrive\ModelResults"


def preprocess_y(y_true, y_pred):
    # Handle case where y_pred has 3 channels (e.g., softmax output)
    if tf.shape(y_pred)[-1] == 3:
        y_pred = y_pred[..., 1]  # Assuming channel 1 is water class
    elif tf.shape(y_pred)[-1] == 1:
        y_pred = tf.squeeze(y_pred, axis=-1)

    y_true = tf.squeeze(y_true, axis=-1)

    # Create mask: labeled pixels only
    mask = tf.cast(y_true >= 0.0, tf.float32)

    # Convert y_true from {-1, 0, 1} to {0, 1}
    y_true = tf.where(y_true < 0.0, 0.0, y_true)

    return y_true, y_pred, mask


def masked_dice(y_true, y_pred, smooth=1e-6):
    y_true, y_pred, mask = preprocess_y(y_true, y_pred)

    y_true = y_true * mask
    y_pred = y_pred * mask

    intersection = tf.reduce_sum(y_true * y_pred)
    total = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

    return (2. * intersection + smooth) / (total + smooth)


def masked_iou(y_true, y_pred, smooth=1e-6):
    y_true, y_pred, mask = preprocess_y(y_true, y_pred)

    y_true = y_true * mask
    y_pred = y_pred * mask

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection

    return (intersection + smooth) / (union + smooth)


def masked_precision(y_true, y_pred, smooth=1e-6):
    y_true, y_pred, mask = preprocess_y(y_true, y_pred)

    y_true = y_true * mask
    y_pred = y_pred * mask

    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred)

    return (tp + smooth) / (tp + fp + smooth)


def masked_recall(y_true, y_pred, smooth=1e-6):
    y_true, y_pred, mask = preprocess_y(y_true, y_pred)

    y_true = y_true * mask
    y_pred = y_pred * mask

    tp = tf.reduce_sum(y_true * y_pred)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred))

    return (tp + smooth) / (tp + fn + smooth)


def sobel_edges(img):
    # img: (H, W), assumed to be 0-1 float mask
    img = tf.expand_dims(img, axis=-1)  # (H, W, 1)
    sobel = tfa.image.sobel_edges(img)  # (H, W, 1, 2)
    dx, dy = sobel[..., 0], sobel[..., 1]
    edge = tf.sqrt(tf.square(dx) + tf.square(dy))  # gradient magnitude
    edge = tf.squeeze(edge, axis=-1)  # (H, W)
    return tf.cast(edge > 0.1, tf.float32)  # binary edge map


def boundary_f1_score(y_true, y_pred, tolerance=3):
    # Handles (H, W, 1) or (H, W, 3)
    if tf.shape(y_pred)[-1] == 3:
        y_pred = y_pred[..., 1]  # water channel
    y_pred = tf.squeeze(y_pred, axis=-1)
    y_true = tf.squeeze(y_true, axis=-1)

    # Handle mask for weak supervision: assume -1 is unlabeled
    mask = tf.cast(y_true >= 0, tf.float32)
    y_true = tf.where(y_true < 0, 0.0, y_true)

    # Extract edges
    true_edge = sobel_edges(y_true) * mask
    pred_edge = sobel_edges(y_pred) * mask

    # Dilate true/pred edges
    kernel = tf.ones((2 * tolerance + 1, 2 * tolerance + 1, 1), tf.float32)
    true_dil = tf.nn.dilation2d(tf.expand_dims(true_edge, axis=-1),
                                filters=kernel,
                                strides=[1, 1, 1, 1],
                                padding='SAME',
                                dilations=[1, 1, 1, 1])

    pred_dil = tf.nn.dilation2d(tf.expand_dims(pred_edge, axis=-1),
                                filters=kernel,
                                strides=[1, 1, 1, 1],
                                padding='SAME',
                                dilations=[1, 1, 1, 1])

    # Remove squeeze and compute TP
    true_match = tf.squeeze(true_dil, axis=-1) * pred_edge
    pred_match = tf.squeeze(pred_dil, axis=-1) * true_edge

    tp = tf.reduce_sum(true_match)
    fp = tf.reduce_sum(pred_edge) - tp
    fn = tf.reduce_sum(true_edge) - tp

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    bf1 = (2 * precision * recall) / (precision + recall + 1e-6)

    return bf1

# Measure Inference Time (optional)
def measure_inference_time(model, input_sample):
    start_time = time.time()
    _ = model.predict(input_sample)
    end_time = time.time()
    return (end_time - start_time) * 1000  # ms

# Combine All Metrics
def evaluate_segmentation(y_true, y_pred, model=None, sample=None):
    metrics = {
        'Dice Coefficient': float(masked_dice(y_true, y_pred).numpy()),
        'IoU Score': float(masked_iou(y_true, y_pred).numpy()),
        'Precision': float(masked_precision(y_true, y_pred).numpy()),
        'Recall': float(masked_recall(y_true, y_pred).numpy())
    }

    if model and sample is not None:
        metrics['Inference Time (ms)'] = float(measure_inference_time(model, sample))

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    return metrics

def evaluate_model(model_name, model, image, mask, index, output_path):
    image = np.expand_dims(image, 0)
    y_true = mask
    y_pred = model.predict(image)
    save_image(output_path, y_pred.squeeze(), f'{model_name}_{index}')

    return evaluate_segmentation(y_true, y_pred, model=model, sample=image)

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
    print('==================================================================================================')
    unet_fcc = load_saved_unet_ffc_model()
    print('UNET-FFC')
    result_metrics['UNET-FFC'] = evaluate_model('UNET-FFC', unet_fcc, image, mask, index, output_path)
    print('==================================================================================================')
    unet_vgg16 = load_saved_unet_VGG16_model()
    print('UNET-VGG16')
    result_metrics['UNET-VGG16'] = evaluate_model('UNET-VGG16', unet_vgg16, image, mask, index, output_path)
    print('==================================================================================================')
    unet_resnet50 = load_saved_unet_ResNet50_model()
    print('UNET-ResNet50')
    result_metrics['UNET-ResNet50'] = evaluate_model('UNET-ResNet50', unet_resnet50, image, mask, index, output_path)
    print('==================================================================================================')
    unet_mobilenetv2 = load_saved_unet_MobileNetV2_model()
    print('UNET-MobileNetV2')
    result_metrics['UNET-MobileNetV2'] = evaluate_model('UNET-MobileNetV2', unet_mobilenetv2, image, mask, index, output_path)
    print('==================================================================================================')
    unet_plus_plus = load_saved_unet_plus_plus_model()
    print('UNET++')
    result_metrics['UNET++'] = evaluate_model('UNET++', unet_plus_plus, image, mask, index, output_path)
    print('==================================================================================================')
    segnet = load_saved_segnet_model()
    print('SegNet')
    result_metrics['SegNet'] = evaluate_model('SegNet', segnet, image, mask, index, output_path)
    print('==================================================================================================')
    segnet_vgg16 = load_saved_segnet_VGG16_model()
    print('SegNet-Vgg16')
    result_metrics['SegNet-Vgg16'] = evaluate_model('SegNet-Vgg16', segnet_vgg16, image, mask, index, output_path)
    print('==================================================================================================')
    res_unet_plus_plus = load_saved_res_unet_plus_plus_model()
    print('ResUNET++')
    result_metrics['ResUNET++'] = evaluate_model('ResUNET++', res_unet_plus_plus, image, mask, index, output_path)
    print('==================================================================================================')
    deeplabv3_plus = load_saved_deeplabv3_plus_model()
    print('DeepLabV3+')
    result_metrics['DeepLabV3+'] = evaluate_model('DeepLabV3+', deeplabv3_plus, image, mask, index, output_path)

    result_file = f'{output_path}/result_metrics_{index}.csv'
    with open(result_file, 'w') as file:
        file.write('Model Name,Dice Coefficient,IoU Score,Precision,Recall,Inference Time (ms)\n')
        for model_name, metrics in result_metrics.items():
            file.write(f'{model_name},'
                       f'{metrics["Dice Coefficient"]},'
                       f'{metrics["IoU Score"]},'
                       f'{metrics["Precision"]},'
                       f'{metrics["Recall"]},'
                       f'{metrics["Inference Time (ms)"]}\n')
    return result_metrics

if __name__=="__main__":
    path = "../input/samples/segnet_512/images"
    image_count = 50
    (images, masks) = load_dataset(path,
                                   size=(512, 512),
                                   file_extension="jpg",
                                   channels=['RED', 'GREEN', 'BLUE'],
                                   image_count=image_count)
    print(f'images shape:{images.shape}')
    print(f'masks shape:{masks.shape}')
    for i in range(image_count):
        make_prediction(images[i], masks[i], i)