import tensorflow as tf
from models.common_utils.loss_functions import combined_masked_dice_bce_loss

def combined_loss_function(y_true, y_pred):
    combined_masked_dice_bce_loss(y_true, y_pred)

def masked_dice_loss(y_true, y_pred, mask = None):
    smooth = 1e-7
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    if mask is None:
        mask = tf.where(tf.math.is_nan(y_true), 0.0, 1.0)

    intersection = tf.reduce_sum(y_true * y_pred * mask)
    union = tf.reduce_sum(y_true * mask) + tf.reduce_sum(y_pred * mask)

    return 1 - (2.0 * intersection + smooth) / (union + smooth)

def combined_masked_dice_bce_loss1(y_true, y_pred, mask=None):
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

def combined_masked_dice_bce_loss2(y_true, y_pred, mask=None):
    return combined_loss_masked(y_true, y_pred)


# Define a weighted BCE + Dice Loss
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def combined_loss(y_true, y_pred):
    # Ensure y_true has the same dtype as y_pred
    y_true = tf.cast(y_true, y_pred.dtype)

    # 1. Define the BCE loss object with reduction set to NONE
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

    # 2. Calculate the un-reduced BCE loss
    bce_unreduced = bce(y_true, y_pred)

    # 3. Define your class weights
    # Note: For water segmentation, this is crucial for imbalance.
    water_weight = 10.0  # Example value, tune this based on your dataset imbalance
    background_weight = 1.0

    # 4. Create a mask of weights based on the true labels (y_true)
    # The mask will have water_weight where y_true is 1 and background_weight elsewhere
    weights_mask = tf.where(tf.equal(y_true, 1), water_weight, background_weight)

    print(f"combined_loss|bce_unreduced shape:{bce_unreduced.shape}")
    print(f"combined_loss|weights_mask shape:{weights_mask.shape}")
    # 5. Apply the weights to the un-reduced BCE loss

    bce_unreduced = tf.expand_dims(bce_unreduced, axis=-1)
    print(f"combined_loss|bce_unreduced shape:{bce_unreduced.shape}")

    weighted_bce_loss = bce_unreduced * weights_mask

    # 6. Manually reduce the weighted loss (e.g., by taking the mean)
    bce_loss_mean = tf.reduce_mean(weighted_bce_loss)

    # 7. Calculate Dice Loss (assuming you have a function for this)
    # Make sure your dice loss is also a mean over the batch
    dice_l = dice_loss(y_true, y_pred)

    # 8. Combine the two losses
    return bce_loss_mean + dice_l


def dice_loss1(y_true, y_pred, smooth=1e-6):
    # Flatten the tensors to 1D for easier calculation
    y_true_f = tf.cast(tf.keras.layers.Flatten()(y_true), tf.float32)
    y_pred_f = tf.cast(tf.keras.layers.Flatten()(y_pred), tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def combined_loss_masked(y_true, y_pred):
    # Ensure dtypes are consistent
    y_true = tf.cast(y_true, y_pred.dtype)

    # Use a BCE loss with no reduction
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

    # Calculate the un-reduced BCE loss. The shape will be (None, 512, 512)
    bce_unreduced = bce(y_true, y_pred)

    # Define class weights. Adjust these values based on your data imbalance.
    water_weight = 10.0
    background_weight = 1.0

    # Create the weights mask. Shape will be (None, 512, 512, 1)
    weights_mask = tf.where(tf.equal(y_true, 1), water_weight, background_weight)

    # Add a channel dimension to the BCE loss tensor to match the mask
    bce_unreduced = tf.expand_dims(bce_unreduced, axis=-1)

    # Apply the weights to the un-reduced BCE loss
    weighted_bce_loss = bce_unreduced * weights_mask

    # --- MASKING LOGIC STARTS HERE ---
    # Create a mask to only consider annotated pixels.
    # We assume y_true is 1 for water and 0 for non-water.
    # The mask is non-zero (1) for annotated pixels and zero for unannotated.
    # In this case, y_true itself can serve as the mask for the BCE loss,
    # but we need to create a dedicated mask for the unannotated areas if they exist.
    # A cleaner approach is to use a dedicated mask. Let's assume you have one.

    # Let's assume you have a 4th channel in your ground truth that is a mask
    # Or, if your y_true is a binary mask, you can infer it.
    # Here, we'll just filter out the zero-loss areas.

    # Let's say your `y_true` is a binary mask (1 for water, 0 for land).
    # You want the background_weight to be 1 for land.
    # If a pixel is unannotated, it's typically assigned a special value, but let's assume
    # for simplicity that your `y_true` is 1 for water and 0 for background.
    # This means your original `tf.where` already implicitly acts as a mask.

    # Let's redefine the scenario: your `y_true` has a special value for "unannotated".
    # For example, `y_true` shape is (None, 512, 512, 1) where
    # 1 = water, 0 = background, and -1 = unannotated.

    # Assuming your dataset gives you a separate mask, let's call it `annotated_mask`.
    # Let's assume it's part of the y_true tensor for simplicity, e.g., the last channel.
    # If not, you need to create it.

    # Example: assume your y_true has shape (None, 512, 512, 2),
    # where the first channel is the water/background labels and the second is the mask.
    # This is a common pattern. Let's work with the simpler case first.

    # Let's assume your y_true is 1 for water and 0 for everything else,
    # and you want to ignore the pixels that are NOT 1.

    # A better way to define the weights for masked areas:
    water_weight_mask = tf.where(tf.equal(y_true, 1), water_weight, background_weight)

    # Let's assume a third class, "unannotated," which should have a weight of 0.
    # This is the key change.

    # This requires your `y_true` to have a way to identify unannotated areas.
    # A common way is to have a third label, e.g., 2.
    # Let's modify the `tf.where` to handle this.

    # --- REVISED LOGIC TO HANDLE UNANNOTATED AREAS ---
    # Assuming `y_true` is 1 for water, 0 for land, and a third value for unannotated.
    # Let's stick with the simple binary case for now, where your `y_true` itself is the mask.
    # The previous code already does this by assigning `background_weight` to the `0` pixels.
    # Let's adjust the `tf.reduce_mean` to be more accurate for sparse masks.

    # Sum the weighted losses
    weighted_bce_loss_sum = tf.reduce_sum(weighted_bce_loss)

    # Get the number of annotated pixels (the sum of the weights mask)
    # The sum of a mask that is 10 for water and 1 for land is not accurate.
    # A better way is to use a separate binary mask

    # Let's create a binary mask of the annotated areas
    annotated_mask = tf.cast(tf.not_equal(y_true, 0), y_pred.dtype)

    # Multiply the weighted loss by this mask to zero out unannotated areas
    weighted_bce_loss = weighted_bce_loss * annotated_mask

    # Sum the loss for only the annotated pixels
    weighted_bce_loss_sum = tf.reduce_sum(weighted_bce_loss)

    # Get the number of annotated pixels to get the true mean loss
    num_annotated_pixels = tf.reduce_sum(annotated_mask) + 1e-6  # Add a small epsilon to avoid division by zero

    bce_loss_mean = weighted_bce_loss_sum / num_annotated_pixels

    # The Dice loss also needs to be masked
    # This can be done by zeroing out the predictions outside the annotated area
    y_pred_masked = y_pred * annotated_mask

    dice_l = dice_loss1(y_true, y_pred_masked)

    return bce_loss_mean + dice_l