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
    print(f"combined_masked_dice_bce_loss|y_true shape:{y_true.shape}")
    print(f"combined_masked_dice_bce_loss|y_pred shape:{y_pred.shape}")
    if mask is None:
        mask = tf.where(tf.math.is_nan(y_true), 0.0, 1.0)

    print(f"combined_masked_dice_bce_loss|mask shape:{mask.shape}")
    dice = masked_dice_loss(y_true, y_pred, mask)
    print(f"combined_masked_dice_bce_loss|dice:{dice}")
    print(f"combined_masked_dice_bce_loss|dice shape:{dice.shape}")
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    print(f"combined_masked_dice_bce_loss|bce shape:{bce.shape}")
    bce = tf.repeat(tf.expand_dims(bce, axis=-1), repeats=3, axis=-1)
    print(f"combined_masked_dice_bce_loss|bce shape:{bce.shape}")
    masked_bce = tf.reduce_sum(bce * mask) / (tf.reduce_sum(mask) + 1e-6)
    return 0.5 * dice + 0.5 * masked_bce