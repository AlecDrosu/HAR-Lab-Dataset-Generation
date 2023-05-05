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

    @tf.function
    def compute_custom_penalty(self, y_pred):
        # Extract the columns from the input tensor
        device_and_status = y_pred[:, 2]
        activity_and_status = y_pred[:, 3]

        # Define the valid ranges
        valid_device_and_status_range = (0, 252)

        # Calculate penalties for device_and_status
        device_and_status_penalty = tf.reduce_sum(
            tf.where(
                tf.logical_and(
                    device_and_status >= valid_device_and_status_range[0],
                    device_and_status <= valid_device_and_status_range[1]
                ),
                0.0,
                1.0
            )
        )

        # Calculate penalties for activity_and_status
        is_out_of_range = tf.logical_or(
            activity_and_status < 0,
            activity_and_status > 22
        )
        activity_and_status_penalty = tf.reduce_sum(tf.cast(is_out_of_range, tf.float32))

        def loop_body(i, invalid_transition_penalty):
            prev_value = activity_and_status[i - 1]
            current_value = activity_and_status[i]

            condition = tf.logical_and(prev_value % 2 == 0, prev_value < 20)
            if tf.reduce_all(condition):
                invalid_transition = tf.logical_and(
                    current_value != prev_value + 1,
                    current_value != 22
                )
                invalid_transition_penalty += tf.cast(invalid_transition, tf.float32)
            else:
                invalid_transition_penalty = invalid_transition_penalty

            return i + 1, invalid_transition_penalty

        def loop_cond(i, _):
            return i < tf.shape(activity_and_status)[0]

        _, invalid_transition_penalty = tf.while_loop(
            loop_cond,
            loop_body,
            (tf.constant(1, dtype=tf.int32), tf.constant(0.0, dtype=tf.float32)),
            shape_invariants=(
                tf.TensorShape([]),
                tf.TensorShape([])
            )
        )

        # Check for the average number of activities in the window
        num_activities = tf.reduce_sum(tf.cast(activity_and_status != 22, tf.float32))
        activity_count_penalty = tf.abs(num_activities - 58.0)  # Adjust the target number of activities as needed

        # Calculate the total penalty
        total_penalty = device_and_status_penalty + activity_and_status_penalty + invalid_transition_penalty + activity_count_penalty

        return total_penalty