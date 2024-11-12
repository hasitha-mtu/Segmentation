import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.ops.metrics_impl import true_positives


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

