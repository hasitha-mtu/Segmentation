import tensorflow as tf
from tensorflow.keras import backend as K

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    d_loss = dice_loss(y_true, y_pred)
    return bce + d_loss

def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    TP = K.sum(y_true_f * y_pred_f)
    FP = K.sum((1 - y_true_f) * y_pred_f)
    FN = K.sum(y_true_f * (1 - y_pred_f))
    return 1 - (TP + 1e-6) / (TP + alpha * FP + beta * FN + 1e-6)

def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=0.75):
    t_loss = tversky_loss(y_true, y_pred, alpha, beta)
    return K.pow(t_loss, gamma)

# # Custom BCE + Dice Loss for tf.distribute.Strategy compatibility
# class BCEDiceLoss(tf.keras.losses.Loss):
#     def __init__(self, global_batch_size, name='bce_dice_loss'):
#         super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name)
#         self.global_batch_size = global_batch_size
#         self.bce = tf.keras.losses.BinaryCrossentropy(
#             reduction=tf.keras.losses.Reduction.NONE
#         )
#
#     def dice_loss(self, y_true, y_pred, smooth=1e-6):
#         y_true_f = tf.reshape(y_true, [self.global_batch_size, -1])
#         y_pred_f = tf.reshape(y_pred, [self.global_batch_size, -1])
#         intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
#         dice = (2. * intersection + smooth) / (
#             tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1) + smooth
#         )
#         return 1 - dice
#
#     def call(self, y_true, y_pred):
#         bce_loss = self.bce(y_true, y_pred)
#         bce_loss = tf.reduce_sum(bce_loss) * (1. / self.global_batch_size)
#
#         d_loss = self.dice_loss(y_true, y_pred)
#         d_loss = tf.reduce_sum(d_loss) * (1. / self.global_batch_size)
#
#         return bce_loss + d_loss

class BCEDiceLoss(tf.keras.losses.Loss):
    def __init__(self, global_batch_size=None, reduction=tf.keras.losses.Reduction.AUTO, name="BCEDiceLoss"):
        super().__init__(reduction=reduction, name=name)
        self.global_batch_size = global_batch_size
        self.bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    def dice_loss(self, y_true, y_pred):
        smooth = 1e-6
        y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
        union = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice

    def call(self, y_true, y_pred):
        bce_loss = self.bce(y_true, y_pred)
        # Default to mean reduction if global_batch_size isn't set
        if self.global_batch_size:
            bce_loss = tf.reduce_sum(bce_loss) / self.global_batch_size
        else:
            bce_loss = tf.reduce_mean(bce_loss)

        dice_loss = tf.reduce_mean(self.dice_loss(y_true, y_pred))
        return bce_loss + dice_loss