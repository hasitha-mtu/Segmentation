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

# Train with Partial Labels Using Masked Loss
def masked_dice_loss(y_true, y_pred):
    """
    Compute Dice Loss but only for labeled pixels (mask > 0).
    y_true: Ground truth segmentation (partial labels)
    y_pred: Model prediction
    mask: Binary mask (1 = labeled pixel, 0 = unlabeled)
    """
    smooth = 1e-6
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    mask = tf.where(tf.math.is_nan(y_true), 0.0, 1.0)

    intersection = tf.reduce_sum(y_true * y_pred * mask)
    union = tf.reduce_sum(y_true * mask) + tf.reduce_sum(y_pred * mask)

    return 1 - (2.0 * intersection + smooth) / (union + smooth)

