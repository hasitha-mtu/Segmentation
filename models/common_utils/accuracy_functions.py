import sys

from sklearn.metrics import jaccard_score, f1_score, accuracy_score
import numpy as np
import tensorflow as tf

def calculate_jaccard_score(y_true, y_pred):
    valid_mask = (y_true != 255)
    y_pred_classes = np.argmax(y_pred, axis=-1).squeeze()
    iou = jaccard_score(y_true[valid_mask].flatten(), y_pred_classes[valid_mask].flatten(), average='macro')
    print("IoU:", iou)

def calculate_pixel(y_true, y_pred):
    valid_mask = (y_true != 255)
    y_pred_classes = np.argmax(y_pred, axis=-1).squeeze()
    accuracy = np.mean((y_pred_classes[valid_mask] == y_true[valid_mask]))
    print("Pixel Accuracy:", accuracy)

def calculate_dice_score(y_true, y_pred):
    valid_mask = (y_true != 255)
    y_pred_classes = np.argmax(y_pred, axis=-1).squeeze()
    dice = f1_score(y_true[valid_mask].flatten(), y_pred_classes[valid_mask].flatten(), average='macro')
    print("Dice Score:", dice)

def masked_accuracy(y_true, y_pred, ignore_value=0):
    y_pred_bin = tf.round(y_pred)
    mask = tf.cast(tf.not_equal(y_true, ignore_value), tf.float32)

    correct = tf.cast(tf.equal(y_true, y_pred_bin), tf.float32)
    masked_correct = correct * mask
    masked = tf.reduce_sum(masked_correct) / (tf.reduce_sum(mask) + 1e-7)
    return masked

def generate_mask(y_true, ignore_value=0.0):
    """
    Binary mask: 1 where y_true != ignore_value, 0 where it should be ignored
    """
    return (y_true != ignore_value).astype(np.float32)

def evaluate_prediction(y_true, y_pred, threshold=0.5, ignore_value=0.0):
    # Remove singleton dimensions if needed
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    # Sanity check: shapes must match
    assert y_true.shape == y_pred.shape, f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"

    # Generate binary mask where label is valid (not ignore_value)
    mask = (y_true != ignore_value)

    # Binarize predictions and labels
    y_pred_bin = (y_pred >= threshold).astype(np.uint8)
    y_true_bin = (y_true >= threshold).astype(np.uint8)

    # Masked flattening
    y_pred_masked = y_pred_bin[mask]
    y_true_masked = y_true_bin[mask]

    acc = accuracy_score(y_true_masked, y_pred_masked)
    f1 = f1_score(y_true_masked, y_pred_masked)

    print("F1 Score:", f1)
    print("Accuracy:", acc)

def calculate_accuracy(y_true, y_pred, threshold=0.5, ignore_value=0.0):
    # Remove singleton dimensions if needed
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    # Sanity check: shapes must match
    assert y_true.shape == y_pred.shape, f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"
    # Generate binary mask where label is valid (not ignore_value)
    mask = (y_true != ignore_value)
    # Binarize predictions and labels
    y_pred_bin = (y_pred >= threshold).astype(np.uint8)
    y_true_bin = (y_true >= threshold).astype(np.uint8)
    # Masked flattening
    y_pred_masked = y_pred_bin[mask]
    y_true_masked = y_true_bin[mask]

    iou = jaccard_score(y_true_masked.flatten(), y_pred_masked.flatten(), average='macro')
    dice = f1_score(y_true_masked.flatten(), y_pred_masked.flatten(), average='macro')
    accuracy = np.mean((y_pred_masked == y_true_masked))
    print("IoU:", iou)
    print("Dice Score:", dice)
    print("Pixel Accuracy:", accuracy)
