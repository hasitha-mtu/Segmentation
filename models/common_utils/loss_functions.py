from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np


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

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    d_loss = dice_loss(y_true, y_pred)
    return bce + d_loss

def masked_dice_loss(y_true, y_pred, mask = None):
    epsilon = 1e-6
    # Clip the predicted values to a safe range to prevent log(0)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

    smooth = 1e-7
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    if mask is None:
        mask = tf.where(tf.math.is_nan(y_true), 0.0, 1.0)

    # Check if the mask is completely empty (no valid pixels)
    if tf.reduce_sum(mask) == 0:
        return 0.0  # Return a loss of 0 if there are no valid pixels to segment

    intersection = tf.reduce_sum(y_true * y_pred * mask)
    union = tf.reduce_sum(y_true * mask) + tf.reduce_sum(y_pred * mask)

    return 1 - (2.0 * intersection + smooth) / (union + smooth)

def combined_masked_dice_bce_loss(y_true, y_pred, mask=None):
    tf.debugging.check_numerics(y_true, "y_true contains NaN/Inf")
    tf.debugging.check_numerics(y_pred, "y_pred contains NaN/Inf")
    # Set a robust epsilon value
    epsilon = 1e-6

    # Clip the predicted values to a safe range to prevent log(0)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

    if mask is None:
        mask = tf.where(tf.math.is_nan(y_true), 0.0, 1.0)

    # Check if the mask is completely empty (no valid pixels)
    if tf.reduce_sum(mask) == 0:
        return 0.0

    dice = masked_dice_loss(y_true, y_pred, mask)

    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
    bce = tf.repeat(tf.expand_dims(bce, axis=-1), repeats=3, axis=-1)

    masked_bce = tf.reduce_sum(bce * mask) / (tf.reduce_sum(mask) + epsilon)
    return 0.5 * dice + 0.5 * masked_bce

def combined_loss_function(y_true, y_pred):
    print(f"combined_loss_function|y_true shape:{y_true.shape}")
    print(f"combined_loss_function|y_pred shape:{y_pred.shape}")
    return combined_masked_dice_bce_loss(y_true, y_pred)

def safe_dice_loss(y_true, y_pred, smooth=1e-7):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1. - dice

def safe_combined_dice_bce(y_true, y_pred, smooth=1e-7):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, smooth, 1.0 - smooth)  # keep probs safe

    # Dice
    dice_loss = safe_dice_loss(y_true, y_pred, smooth)

    # BCE (direct mean over all pixels)
    bce_loss = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(y_true, y_pred)
    )

    return 0.5 * dice_loss + 0.5 * bce_loss

def combined_masked_dice_bce_loss1(y_true, y_pred):
    # This is a key step to prevent NaN values from the start
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

    # Use a tensor-based boolean check
    mask = tf.where(tf.math.is_nan(y_true), 0.0, 1.0)
    mask_sum = tf.reduce_sum(mask)

    # We use tf.cond to create a TensorFlow graph conditional
    loss = tf.cond(tf.equal(mask_sum, 0),
                   lambda: 0.0,
                   lambda: _calculate_masked_loss(y_true, y_pred, mask))
    return loss


def _calculate_masked_loss(y_true, y_pred, mask):
    # Dice Loss
    smooth = 1e-7

    # We add a small constant to prevent division by zero in the union
    y_true_masked = y_true * mask
    y_pred_masked = y_pred * mask

    intersection = tf.reduce_sum(y_true_masked * y_pred_masked)
    union = tf.reduce_sum(y_true_masked) + tf.reduce_sum(y_pred_masked)
    dice_loss = 1 - (2.0 * intersection + smooth) / (union + smooth)

    # BCE Loss
    bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    # Ensure shapes are compatible before multiplication
    if bce_loss.shape.rank == 4:
        bce_loss = tf.squeeze(bce_loss, axis=-1)
    if mask.shape.rank == 4:
        mask = tf.squeeze(mask, axis=-1)

    # Apply the mask
    masked_bce_loss = tf.reduce_sum(bce_loss * mask) / (tf.reduce_sum(mask) + 1e-7)

    return 0.5 * dice_loss + 0.5 * masked_bce_loss

def safe_dice_loss(y_true, y_pred, smooth=1e-7):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1. - dice


def safe_combined_dice_bce(y_true, y_pred, smooth=1e-7):
    # Ensure numerical stability
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), smooth, 1.0 - smooth)

    # Dice
    dice_loss = safe_dice_loss(y_true, y_pred, smooth)

    # BCE
    bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce_loss = tf.reduce_mean(bce_loss)

    # Combine
    total_loss = 0.5 * dice_loss + 0.5 * bce_loss
    return total_loss

# def masked_dice_loss(y_true, y_pred, mask=None):
#     smooth = 1e-6
#     y_true = tf.cast(y_true, tf.float32)
#     y_pred = tf.cast(y_pred, tf.float32)
#
#     if mask is None:
#         mask = tf.ones_like(y_true)  # assume no NaNs in y_true
#
#     intersection = tf.reduce_sum(y_true * y_pred * mask)
#     union = tf.reduce_sum(y_true * mask) + tf.reduce_sum(y_pred * mask)
#     dice = (2.0 * intersection + smooth) / (union + smooth)
#     return 1.0 - dice
#
#
# def combined_masked_dice_bce_loss(y_true, y_pred, mask=None):
#     y_true = tf.cast(y_true, tf.float32)
#     y_pred = tf.cast(y_pred, tf.float32)
#
#     if mask is None:
#         mask = tf.ones_like(y_true)
#
#     # Dice loss
#     dice = masked_dice_loss(y_true, y_pred, mask)
#
#     # Binary cross-entropy (handles (batch, h, w, 1) automatically)
#     bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
#     bce = tf.reduce_mean(bce * tf.squeeze(mask, axis=-1))
#
#     return 0.5 * dice + 0.5 * bce
#
# # ---- Debug Harness ----
# def test_loss_function():
#     # Create dummy batch: 2 images of size 128x128 with 1 channel
#     y_true = np.random.randint(0, 2, size=(2, 128, 128, 1)).astype("float32")
#     y_pred = np.random.rand(2, 128, 128, 1).astype("float32")  # model outputs in [0,1]
#
#     # Run the loss
#     loss_value = combined_masked_dice_bce_loss(y_true, y_pred)
#
#     print("Unique values in y_true:", np.unique(y_true))
#     print("y_pred range:", y_pred.min(), y_pred.max())
#     print("Loss value:", loss_value.numpy())
#
#
#
#
# if __name__ == "__main__":
#     test_loss_function()


# # ---- Debugging with one real batch ----
# def debug_with_real_batch(dataset, model=None):
#     # Take one batch from dataset
#     for x_batch, y_batch in dataset.take(1):
#         print("Batch shapes:", x_batch.shape, y_batch.shape)
#         print("y_batch unique values:", np.unique(y_batch.numpy()))
#
#         # Run model forward pass if provided
#         if model is not None:
#             y_pred = model(x_batch, training=False)
#         else:
#             # If no model yet, simulate predictions
#             y_pred = tf.random.uniform(y_batch.shape, 0, 1)
#
#         print("y_pred range:", float(tf.reduce_min(y_pred)), float(tf.reduce_max(y_pred)))
#
#         # Compute loss
#         loss_value = combined_masked_dice_bce_loss(y_batch, y_pred)
#         print("Loss value:", float(loss_value.numpy()))
#
#
# # ---- Example usage ----
# from models.common_utils.dataset import load_datasets
# from models.deeplabv3_plus.model import deeplab_v3_plus
# if __name__ == "__main__":
#     # Example: if you already have a tf.data.Dataset for training
#     config_file = '../deeplabv3_plus/config.yaml'
#     train_dataset, validation_dataset = load_datasets(config_file, True)
#     model = deeplab_v3_plus(512, 512, 3)
#
#     debug_with_real_batch(train_dataset, model)

