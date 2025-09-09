from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np


# recall
# def recall_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall
#
# # precision
# def precision_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives/ (predicted_positives + K.epsilon())
#     return precision
#
# # f1_score
# def f1_score(y_true, y_pred):
#     recall = recall_m(y_true, y_pred)
#     precision = precision_m(y_true, y_pred)
#     f1_score = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
#     return f1_score

def recall_m(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If no positives in ground truth → return 1.0 (perfect recall by convention)
    return tf.where(
        tf.greater(possible_positives, 0),
        true_positives / (possible_positives + K.epsilon()),
        1.0
    )

def precision_m(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    # If model predicts nothing → precision = 1.0 (no false positives)
    return tf.where(
        tf.greater(predicted_positives, 0),
        true_positives / (predicted_positives + K.epsilon()),
        1.0
    )

def f1_score(y_true, y_pred):
    p = precision_m(y_true, y_pred)
    r = recall_m(y_true, y_pred)

    return tf.where(
        tf.greater(p + r, 0),
        2 * ((p * r) / (p + r + K.epsilon())),
        0.0
    )

def dice_loss(y_true, y_pred):
    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return 1 - (2. * intersection + smooth) / (union + smooth)

def masked_loss(y_true, y_pred, mask):
    """Compute loss only for labeled pixels."""
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    loss = loss_fn(y_true * mask, y_pred * mask)
    return loss

# Train with Partial Labels Using Masked Loss
def masked_dice_loss2(y_true, y_pred):
    """
    Compute Dice Loss but only for labeled pixels (mask > 0).
    y_true: Ground truth segmentation (partial labels)
    y_pred: Model prediction
    mask: Binary mask (1 = labeled pixel, 0 = unlabeled)
    """
    smooth = 1e-7
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    mask = tf.where(tf.math.is_nan(y_true), 0.0, 1.0)
    # mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred * mask)
    union = tf.reduce_sum(y_true * mask) + tf.reduce_sum(y_pred * mask)

    return 1 - (2.0 * intersection + smooth) / (union + smooth)

def partial_crossentropy(y_true, y_pred):
    """
    y_true: [batch, h, w, 1] - with labels: 1 (water), 0 (non-water), -1 (ignore)
    y_pred: [batch, h, w, 1] - predicted probability map
    """
    print(f"partial_crossentropy|y_true shape: {y_true.shape}")
    print(f"partial_crossentropy|y_pred shape: {y_pred.shape}")
    mask = tf.not_equal(y_true, 0.0)
    print(f"partial_crossentropy|mask shape: {mask.shape}")
    y_true_clipped = tf.where(mask, y_true, tf.zeros_like(y_true))
    print(f"partial_crossentropy|y_true_clipped shape: {y_true_clipped.shape}")
    loss = tf.keras.losses.binary_crossentropy(y_true_clipped, y_pred)
    print(f"partial_crossentropy|loss shape: {loss.shape}")
    loss = tf.stack([loss, loss, loss], axis=-1)
    loss = tf.where(mask, loss, tf.zeros_like(loss))
    return tf.reduce_mean(loss)

def wsl_masked_dice_loss(y_true, y_pred, mask=None, smooth=1e-6):
    if mask is None:
        mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)

    y_true = y_true * mask
    y_pred = y_pred * mask

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

def masked_dice_loss1(y_true, y_pred, mask=None, smooth=1e-6):
    if mask is None:
        mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)

    y_true = y_true * mask
    y_pred = y_pred * mask

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    dice = (2. * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(1 - dice)

def masked_focal_loss(y_true, y_pred, mask=None, alpha=None, gamma=2.0):
    if mask is None:
        mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)

    epsilon = K.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    if alpha is None:
        pos = tf.reduce_sum(y_true * mask)
        neg = tf.reduce_sum((1 - y_true) * mask)
        alpha = pos / (pos + neg + epsilon)

    cross_entropy = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    focal = alpha * tf.pow(1. - pt, gamma) * cross_entropy
    masked_focal = focal * mask
    return tf.reduce_sum(masked_focal) / (tf.reduce_sum(mask) + epsilon)

def edge_penalty_loss(y_true, y_pred, mask=None, weight=0.1):
    if mask is None:
        mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)
    edge_true = tf.image.sobel_edges(y_true)  # Shape: [B, H, W, 1, 2]
    edge_pred = tf.image.sobel_edges(y_pred)
    edge_diff = tf.reduce_mean(tf.abs(edge_true - edge_pred), axis=-1)

    edge_masked = edge_diff * mask
    return weight * tf.reduce_mean(edge_masked)

def combined_masked_dice_focal_loss(y_true, y_pred, dice_weight=0.25, focal_weight=0.75):
    mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)

    dice = masked_dice_loss1(y_true, y_pred, mask)
    focal = masked_focal_loss(y_true, y_pred, mask)

    return dice_weight * dice + focal_weight * focal

def combined_loss_with_edge(y_true, y_pred, dice_weight=0.3, focal_weight=0.4, edge_weight=0.3):
    mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)
    dice = masked_dice_loss1(y_true, y_pred, mask)
    focal = masked_focal_loss(y_true, y_pred, mask)
    edge = edge_penalty_loss(y_true, y_pred, mask)
    return dice_weight * dice + focal_weight * focal + edge_weight * edge


def unet_resnet50_loss_function(y_true, y_pred):
    return bce_dice_loss(y_true, y_pred)

def dice_loss1(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    d_loss = dice_loss1(y_true, y_pred)
    return bce + d_loss

# def masked_dice_loss(y_true, y_pred, mask = None):
#     epsilon = 1e-6
#     # Clip the predicted values to a safe range to prevent log(0)
#     y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
#
#     smooth = 1e-7
#     y_true = tf.cast(y_true, tf.float32)
#     y_pred = tf.cast(y_pred, tf.float32)
#
#     if mask is None:
#         mask = tf.where(tf.math.is_nan(y_true), 0.0, 1.0)
#
#     # Check if the mask is completely empty (no valid pixels)
#     if tf.reduce_sum(mask) == 0:
#         return 0.0  # Return a loss of 0 if there are no valid pixels to segment
#
#     intersection = tf.reduce_sum(y_true * y_pred * mask)
#     union = tf.reduce_sum(y_true * mask) + tf.reduce_sum(y_pred * mask)
#
#     return 1 - (2.0 * intersection + smooth) / (union + smooth)
#
# def combined_masked_dice_bce_loss(y_true, y_pred, mask=None):
#     tf.debugging.check_numerics(y_true, "y_true contains NaN/Inf")
#     tf.debugging.check_numerics(y_pred, "y_pred contains NaN/Inf")
#     # Set a robust epsilon value
#     epsilon = 1e-6
#
#     # Clip the predicted values to a safe range to prevent log(0)
#     y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
#
#     if mask is None:
#         mask = tf.where(tf.math.is_nan(y_true), 0.0, 1.0)
#
#     # Check if the mask is completely empty (no valid pixels)
#     if tf.reduce_sum(mask) == 0:
#         return 0.0
#
#     dice = masked_dice_loss(y_true, y_pred, mask)
#
#     bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
#     bce = tf.repeat(tf.expand_dims(bce, axis=-1), repeats=3, axis=-1)
#
#     masked_bce = tf.reduce_sum(bce * mask) / (tf.reduce_sum(mask) + epsilon)
#     return 0.5 * dice + 0.5 * masked_bce

def masked_dice_loss3(y_true, y_pred, mask=None, smooth=1e-7):
    epsilon = 1e-6
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Clip preds to avoid log(0) and extreme values
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

    # Default mask: all ones (use all pixels)
    if mask is None:
        mask = tf.ones_like(y_true, dtype=tf.float32)

    # Compute intersection and union only over masked region
    intersection = tf.reduce_sum(y_true * y_pred * mask)
    union = tf.reduce_sum(y_true * mask) + tf.reduce_sum(y_pred * mask)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice


def debug_combined_masked_dice_bce_loss(y_true, y_pred, mask=None, smooth=1e-7, epsilon=1e-6, batch_index=None):
    """
    A safe Dice + BCE loss with built-in debug logging.
    If NaNs or Infs appear, it prints the batch index and returns a safe loss.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Clip predictions to safe range
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

    if mask is None:
        mask = tf.ones_like(y_true, dtype=tf.float32)

    # ---- Dice part ----
    intersection = tf.reduce_sum(y_true * y_pred * mask)
    union = tf.reduce_sum(y_true * mask) + tf.reduce_sum(y_pred * mask)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice

    # ---- BCE part ----
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
    if len(bce.shape) < len(mask.shape):
        bce = tf.expand_dims(bce, axis=-1)
    masked_bce = tf.reduce_sum(bce * mask) / (tf.reduce_sum(mask) + epsilon)

    # ---- Combine ----
    loss = 0.5 * dice_loss + 0.5 * masked_bce

    # ---- Debug checks ----
    nan_in_loss = tf.reduce_any(tf.math.is_nan(loss))
    inf_in_loss = tf.reduce_any(tf.math.is_inf(loss))

    def _debug_print():
        tf.print("⚠️ NaN/Inf detected in loss",
                 "Batch:", batch_index if batch_index is not None else -1,
                 "dice_loss:", dice_loss,
                 "masked_bce:", masked_bce,
                 "loss:", loss,
                 summarize=-1)
        return tf.constant(True)

    tf.cond(nan_in_loss | inf_in_loss, _debug_print, lambda: tf.constant(False))

    # Replace NaN/Inf with a safe fallback value
    loss = tf.where(tf.math.is_finite(loss), loss, tf.constant(1.0, dtype=loss.dtype))
    return loss

def combined_loss_function(y_true, y_pred):
    return combined_masked_dice_bce_loss(y_true, y_pred)

def safe_binary_crossentropy(y_true, y_pred, epsilon=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    return - (y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))

def masked_dice_loss(y_true, y_pred, mask=None, smooth=1e-7):
    if mask is None:
        mask = tf.ones_like(y_true, dtype=tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred * mask)
    union = tf.reduce_sum(y_true * mask) + tf.reduce_sum(y_pred * mask)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    dice = tf.where(tf.math.is_finite(dice), dice, 0.0)  # <-- safety
    return 1.0 - dice

def combined_masked_dice_bce_loss(y_true, y_pred, mask=None, smooth=1e-7, epsilon=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

    if mask is None:
        mask = tf.ones_like(y_true, dtype=tf.float32)

    dice = masked_dice_loss(y_true, y_pred, mask, smooth)

    # Safe BCE
    bce = safe_binary_crossentropy(y_true, y_pred, epsilon)
    if len(bce.shape) < len(mask.shape):
        bce = tf.expand_dims(bce, axis=-1)
    masked_bce = tf.reduce_sum(bce * mask) / (tf.reduce_sum(mask) + epsilon)

    # Replace NaN with safe value
    masked_bce = tf.where(tf.math.is_finite(masked_bce), masked_bce, 0.0)

    return 0.5 * dice + 0.5 * masked_bce