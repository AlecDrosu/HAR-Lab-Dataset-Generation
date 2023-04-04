# import numpy as np
# import tensorflow as tf
# from keras import backend as K

def custom_penalty(y_pred, normalized_min_max_values, input_dim):
    # Device ID constraints
    min_device_id_m = normalized_min_max_values[0, 1]
    max_device_id_m = normalized_min_max_values[1, 1]
    min_device_id_t = normalized_min_max_values[2, 1]
    max_device_id_t = normalized_min_max_values[3, 1]
    min_device_id_d = normalized_min_max_values[0, 1]
    max_device_id_d = normalized_min_max_values[1, 1]

    # Status constraints
    min_status_m = normalized_min_max_values[0, 2]
    max_status_m = normalized_min_max_values[1, 2]
    min_status_t = normalized_min_max_values[2, 2]
    max_status_t = normalized_min_max_values[3, 2]
    min_status_d = normalized_min_max_values[0, 2]
    max_status_d = normalized_min_max_values[1, 2]

    # Reshape the flattened y_pred to have the correct shape
    y_pred = K.reshape(y_pred, (-1, input_dim))

    # Extract the 'Device ID' and 'Status' columns from the predicted values
    y_pred_device_id = y_pred[:, 1]
    y_pred_status = y_pred[:, 2]

    # Calculate the penalty for values outside the desired range
    m_device_penalty = K.sum(K.relu(y_pred_device_id - max_device_id_m) + K.relu(min_device_id_m - y_pred_device_id))
    t_device_penalty = K.sum(K.relu(y_pred_device_id - max_device_id_t) + K.relu(min_device_id_t - y_pred_device_id))
    d_device_penalty = K.sum(K.relu(y_pred_device_id - max_device_id_d) + K.relu(min_device_id_d - y_pred_device_id))

    m_status_penalty = K.sum(K.relu(y_pred_status - max_status_m) + K.relu(min_status_m - y_pred_status))
    t_status_penalty = K.sum(K.relu(y_pred_status - max_status_t) + K.relu(min_status_t - y_pred_status))
    d_status_penalty = K.sum(K.relu(y_pred_status - max_status_d) + K.relu(min_status_d - y_pred_status))

    penalty = m_device_penalty + t_device_penalty + d_device_penalty + m_status_penalty + t_status_penalty + d_status_penalty

    return penalty

import numpy as np
import tensorflow as tf
from keras import backend as K

def custom_penalty(y_pred, normalized_min_max_values, input_dim):
    y_pred = K.reshape(y_pred, (-1, input_dim))

    y_pred_device_id = y_pred[:, 1]
    y_pred_status = y_pred[:, 2]

    penalties = []
    for i in range(3):
        min_device_id = K.reshape(normalized_min_max_values[i * 2, 1], K.shape(y_pred_device_id))
        max_device_id = K.reshape(normalized_min_max_values[i * 2 + 1, 1], K.shape(y_pred_device_id))
        min_status = K.reshape(normalized_min_max_values[i * 2, 2], K.shape(y_pred_status))
        max_status = K.reshape(normalized_min_max_values[i * 2 + 1, 2], K.shape(y_pred_status))

        device_penalty = K.sum(K.relu(y_pred_device_id - max_device_id) + K.relu(min_device_id - y_pred_device_id))
        status_penalty = K.sum(K.relu(y_pred_status - max_status) + K.relu(min_status - y_pred_status))

        penalties.append(device_penalty + status_penalty)

    penalty = K.mean(K.stack(penalties, axis=0))

    return penalty

def custom_penalty(y_pred, normalized_min_max_values, input_dim, loss):
    # Device ID constraints
    min_device_id_m = normalized_min_max_values[0, 1]
    max_device_id_m = normalized_min_max_values[1, 1]
    min_device_id_t = normalized_min_max_values[2, 1]
    max_device_id_t = normalized_min_max_values[3, 1]
    min_device_id_d = normalized_min_max_values[0, 1]
    max_device_id_d = normalized_min_max_values[1, 1]

    # Status constraints
    min_status_m = normalized_min_max_values[0, 2]
    max_status_m = normalized_min_max_values[1, 2]
    min_status_t = normalized_min_max_values[2, 2]
    max_status_t = normalized_min_max_values[3, 2]
    min_status_d = normalized_min_max_values[0, 2]
    max_status_d = normalized_min_max_values[1, 2]

    # Reshape the flattened y_pred to have the correct shape
    y_pred = K.reshape(y_pred, (-1, input_dim))

    # Extract the 'Device ID' and 'Status' columns from the predicted values
    y_pred_device_id = y_pred[:, 1]
    y_pred_status = y_pred[:, 2]

    # Calculate the penalty for values outside the desired range
    m_device_penalty = K.sum(K.relu(y_pred_device_id - max_device_id_m) + K.relu(min_device_id_m - y_pred_device_id))
    t_device_penalty = K.sum(K.relu(y_pred_device_id - max_device_id_t) + K.relu(min_device_id_t - y_pred_device_id))
    d_device_penalty = K.sum(K.relu(y_pred_device_id - max_device_id_d) + K.relu(min_device_id_d - y_pred_device_id))

    m_status_penalty = K.sum(K.relu(y_pred_status - max_status_m) + K.relu(min_status_m - y_pred_status))
    t_status_penalty = K.sum(K.relu(y_pred_status - max_status_t) + K.relu(min_status_t - y_pred_status))
    d_status_penalty = K.sum(K.relu(y_pred_status - max_status_d) + K.relu(min_status_d - y_pred_status))

    penalty = m_device_penalty + t_device_penalty + d_device_penalty + m_status_penalty + t_status_penalty + d_status_penalty
    penalty_per_sample = penalty / K.cast(K.shape(y_pred)[0], dtype='float32')

    return K.reshape(penalty_per_sample, K.shape(loss))

# LOSS FUNCTION FOR CUSTOM PENALTY ---------------------------------------------------------------------------------------------#
# sample_size = K.prod(K.shape(inputs)[:-1])
# reconstruction_loss = binary_crossentropy(K.reshape(inputs, (sample_size, input_dim)), K.reshape(outputs, (sample_size, input_dim)))
# reconstruction_loss *= input_dim
# kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
# kl_loss = K.sum(kl_loss, axis=-1)
# kl_loss *= -0.5

# # Add the custom penalty to the loss function
# penalty_weight = 10.0  # Adjust the weight of the penalty term as needed
# # penalty = custom_penalty(y_pred=outputs)
# penalty = custom_penalty(outputs, normalized_min_max_values, input_dim, reconstruction_loss)
# penalty *= penalty_weight

# vae_loss = K.mean(reconstruction_loss + kl_loss + penalty)
# vae.add_loss(vae_loss/4096)
# vae.compile(optimizer='adam')
# ------------------------------------------------------------------------------------------------------------------------------#