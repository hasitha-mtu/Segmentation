import tensorflow_addons as tfa
import tensorflow as tf

def get_optimizer(config_val):
    initial_learning_rate = 1e-4  # A smaller initial learning rate
    weight_decay = 1e-5  # A small, but effective weight decay
    if config_val == 'AdamW':
        return tfa.optimizers.AdamW(
            weight_decay=weight_decay,
            learning_rate=initial_learning_rate
        )
    else:
        return tf.optimizers.Adam()

