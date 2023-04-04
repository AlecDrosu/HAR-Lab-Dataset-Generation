from keras.layers import Layer
from keras import backend as K
import tensorflow as tf

class CustomPenaltyLayer(Layer):
    def __init__(self, scaler, input_dim, timesteps, **kwargs):
        super(CustomPenaltyLayer, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.timesteps = timesteps
        self.scale_ = K.constant(scaler.scale_, dtype=tf.float32)
        self.min_ = K.constant(scaler.min_, dtype=tf.float32)

    def call(self, inputs):
        x = inputs
        x_reshaped = K.reshape(x, (-1, self.input_dim))
        x_inv = (x_reshaped - self.min_) / self.scale_
        x_inv = K.reshape(x_inv, (-1, self.timesteps, self.input_dim))
        penalty = self.compute_custom_penalty(x_inv)
        return penalty

    def compute_custom_penalty(self, y_pred):
        # Extract the columns from the input tensor
        device_id = y_pred[:, 2]
        status = y_pred[:, 3]
        activity = y_pred[:, 4]
        activity_status = y_pred[:, 5]

        # Device IDs starting with 'M'
        m_device_condition = tf.logical_and(3 <= device_id, device_id <= 33)
        m_device_penalty = tf.reduce_sum(tf.cast(tf.logical_and(status != 55, status != 54), tf.float32))

        # Device IDs starting with 'T'
        t_device_condition = tf.logical_and(34 <= device_id, device_id <= 38)
        t_device_penalty = tf.reduce_sum(tf.cast(tf.logical_not(tf.logical_and(16 <= status, status <= 43)), tf.float32))

        # Device IDs starting with 'D'
        d_device_condition = tf.logical_and(0 <= device_id, device_id <= 2)
        d_device_penalty = tf.reduce_sum(tf.cast(tf.logical_and(status != 56, status != 53), tf.float32))

        # Enforce activity constraints
        activity_condition = tf.logical_and(0 <= activity, activity <= 10)
        activity_penalty = tf.reduce_sum(tf.cast(tf.logical_and(activity_status != 0, activity_status != 1), tf.float32))

        penalty = m_device_penalty + t_device_penalty + d_device_penalty + activity_penalty
        return penalty