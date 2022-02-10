# implementation from DeepLTL https://github.com/reactive-systems/deepltl

import tensorflow as tf


class TransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning Rate Schedule proposed by Vaswani et al. (2017) that corresponds to a linear increase
    during the warmup phase followed by a decrease proportional to the inverse of the square root of
    the step number"""

    def __init__(self, d_embedding, warmup_steps=4000):
        super(TransformerSchedule, self).__init__()

        self.d_embedding = tf.cast(d_embedding, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        increasing_lr = step * (self.warmup_steps ** -1.5)
        decreasing_lr = tf.math.rsqrt(step)
        return tf.math.rsqrt(self.d_embedding) * tf.math.minimum(increasing_lr, decreasing_lr)
