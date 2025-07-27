import tensorflow as tf

def masked_dice_loss(y_true, y_pred, mask = None):
    smooth = 1e-7
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    if mask is None:
        mask = tf.where(tf.math.is_nan(y_true), 0.0, 1.0)

    intersection = tf.reduce_sum(y_true * y_pred * mask)
    union = tf.reduce_sum(y_true * mask) + tf.reduce_sum(y_pred * mask)

    return 1 - (2.0 * intersection + smooth) / (union + smooth)

def combined_masked_dice_bce_loss(y_true, y_pred, mask=None):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    if mask is None:
        mask = tf.where(tf.math.is_nan(y_true), 0.0, 1.0)

    dice = masked_dice_loss(y_true, y_pred, mask)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    mask_squeezed = tf.squeeze(mask, axis=-1)
    sum1 = tf.reduce_sum(bce * mask_squeezed)
    sum2 = (tf.reduce_sum(mask_squeezed) + 1e-6)
    masked_bce = sum1 / sum2
    return 0.5 * dice + 0.5 * masked_bce