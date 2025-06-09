from tensorflow.keras import backend as K
import tensorflow as tf


# recall
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# precision
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives/ (predicted_positives + K.epsilon())
    return precision

# f1_score
def f1_score(y_true, y_pred):
    recall = recall_m(y_true, y_pred)
    precision = precision_m(y_true, y_pred)
    f1_score = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return f1_score

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

# # Train with Partial Labels Using Masked Loss
# def masked_dice_loss(y_true, y_pred):
#     """
#     Compute Dice Loss but only for labeled pixels (mask > 0).
#     y_true: Ground truth segmentation (partial labels)
#     y_pred: Model prediction
#     mask: Binary mask (1 = labeled pixel, 0 = unlabeled)
#     """
#     smooth = 1e-7
#     y_true = tf.cast(y_true, tf.float32)
#     y_pred = tf.cast(y_pred, tf.float32)
#
#     # mask = tf.where(tf.math.is_nan(y_true), 0.0, 1.0)
#     mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)
#
#     intersection = tf.reduce_sum(y_true * y_pred * mask)
#     union = tf.reduce_sum(y_true * mask) + tf.reduce_sum(y_pred * mask)
#
#     return 1 - (2.0 * intersection + smooth) / (union + smooth)

def partial_crossentropy(y_true, y_pred):
    """
    y_true: [batch, h, w, 1] - with labels: 1 (water), 0 (non-water), -1 (ignore)
    y_pred: [batch, h, w, 1] - predicted probability map
    """
    mask = tf.not_equal(y_true, -1)
    y_true_clipped = tf.where(mask, y_true, tf.zeros_like(y_true))
    loss = tf.keras.losses.binary_crossentropy(y_true_clipped, y_pred)
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

def masked_dice_loss(y_true, y_pred, mask=None, smooth=1e-6):
    if mask is None:
        mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)

    print(f"masked_dice_loss|y_true:{y_true.shape}")
    print(f"masked_dice_loss|y_pred:{y_pred.shape}")
    print(f"masked_dice_loss|mask:{mask.shape}")

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

    dice = masked_dice_loss(y_true, y_pred, mask)
    focal = masked_focal_loss(y_true, y_pred, mask)

    return dice_weight * dice + focal_weight * focal

def combined_loss_with_edge(y_true, y_pred, dice_weight=0.3, focal_weight=0.6, edge_weight=0.1):
    mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)
    dice = masked_dice_loss(y_true, y_pred, mask)
    focal = masked_focal_loss(y_true, y_pred, mask)
    edge = edge_penalty_loss(y_true, y_pred, mask)
    return dice_weight * dice + focal_weight * focal + edge_weight * edge
