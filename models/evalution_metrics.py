import tensorflow as tf
import numpy as np
import time
import cv2
from scipy.spatial.distance import directed_hausdorff

# Dice Coefficient
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


# IoU Score
def iou_score(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


# Pixel Accuracy
def pixel_accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(tf.math.greater_equal(y_pred, 0.5), tf.bool)
    correct = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.float32))
    total = tf.size(y_true, out_type=tf.float32)
    return correct / total


# Precision and Recall
def precision_recall(y_true, y_pred, smooth=1e-6):
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
    input_sample = tf.convert_to_tensor(input_sample)
    input_sample = tf.expand_dims(input_sample, axis=0)  # add batch dimension
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

    return metrics


def get_boundary(mask, dilation_ratio=0.02):
    """
    Extract the boundary pixels from a binary mask using dilation and erosion.
    dilation_ratio: proportion of image diagonal to determine boundary width
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = max(1, int(round(dilation_ratio * img_diag)))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=dilation)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=dilation)
    boundary = dilated - eroded
    return boundary


def boundary_iou(y_true, y_pred, dilation_ratio=0.02):
    """
    Calculate Boundary IoU score for binary masks.
    """
    y_true = y_true.astype(np.bool_)
    y_pred = y_pred.astype(np.bool_)

    true_boundary = get_boundary(y_true, dilation_ratio)
    pred_boundary = get_boundary(y_pred, dilation_ratio)

    intersection = np.logical_and(true_boundary, pred_boundary).sum()
    union = np.logical_or(true_boundary, pred_boundary).sum()

    if union == 0:
        return 1.0  # Both boundaries empty means perfect match

    return intersection / union


def hausdorff_distance(y_true, y_pred):
    """
    Compute the Hausdorff Distance between two binary masks.
    Returns the symmetric Hausdorff distance.
    """
    y_true = y_true.astype(np.bool_)
    y_pred = y_pred.astype(np.bool_)

    true_points = np.argwhere(y_true)
    pred_points = np.argwhere(y_pred)

    if len(true_points) == 0 or len(pred_points) == 0:
        return np.nan  # undefined if no points

    forward_hd = directed_hausdorff(true_points, pred_points)[0]
    backward_hd = directed_hausdorff(pred_points, true_points)[0]

    return max(forward_hd, backward_hd)

# Both require binary masks (0 or 1) as NumPy arrays.
# Typically, you'd threshold your model output predictions (e.g., pred_mask = (pred_prob > 0.5)).
# Hausdorff distance can be sensitive to noise/outliers;
# consider using a percentile Hausdorff (e.g., 95th percentile) if needed.

if __name__=="__manin__":
    y_true = ground_truth_mask  # shape: (H, W)
    y_pred = model.predict(image)  # shape: (H, W)
    results = evaluate_segmentation(y_true, y_pred, model=model, sample=image)
    print(results)

    # y_true and y_pred are binary masks (H, W)
    b_iou = boundary_iou(y_true, y_pred)
    hd = hausdorff_distance(y_true, y_pred)

    print(f"Boundary IoU: {b_iou:.4f}")
    print(f"Hausdorff Distance: {hd:.4f} pixels")