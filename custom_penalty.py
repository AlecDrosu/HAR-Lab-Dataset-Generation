def custom_penalty(y_pred, scaler):
    # Inverse transform the predicted values
    y_pred_inv = scaler.inverse_transform(y_pred)

    # Initialize penalties
    m_device_penalty = 0
    t_device_penalty = 0
    d_device_penalty = 0

    # Iterate through the rows and enforce constraints
    for row in y_pred_inv:
        device_id = row[2]
        status = row[3]
        activity = row[4]
        activity_status = row[5]

        # Device IDs starting with 'M'
        if 3 <= device_id <= 33:
            if status != 55 and status != 54:
                m_device_penalty += 1

        # Device IDs starting with 'T'
        elif 34 <= device_id <= 38:
            if not (16 <= status <= 43):  # Assuming the temperature range is from 0 to 43
                t_device_penalty += 1

        # Device IDs starting with 'D'
        elif 0 <= device_id <= 2:
            if status != 56 and status != 53:
                d_device_penalty += 1

        # Enforce activity constraints
        if 0 <= activity <= 10:
            if activity_status != 0 and activity_status != 1:
                d_device_penalty += 1

    penalty = m_device_penalty + t_device_penalty + d_device_penalty
    return penalty

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