# Failed attempt 1

pd.options.display.max_rows = None
pd.options.display.max_columns = None

# Load the original dataset
processed_data = pd.read_csv('processed_data_converted.csv')

# Extract the relevant columns from the dataset
timestamp = processed_data['Timestamp'].values
device_id = processed_data['Device ID'].values
status = processed_data['Status'].values
activity = processed_data['Activity'].values
activity_status = processed_data['Activity Status'].values

# Normalize the data for training
scaler = MinMaxScaler()
timestamp = scaler.fit_transform(timestamp.reshape(-1, 1))
status = scaler.fit_transform(status.reshape(-1, 1))
activity = scaler.fit_transform(activity.reshape(-1, 1))
activity_status = scaler.fit_transform(activity_status.reshape(-1, 1))

# Build the encoder model
encoder_inputs = keras.Input(shape=(4,))
x = layers.Dense(64, activation='relu')(encoder_inputs)
x = layers.Dense(32, activation='relu')(x)
z_mean = layers.Dense(16, activation='linear')(x)
z_log_var = layers.Dense(16, activation='linear')(x)

# Create a custom layer for the sampling function
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Use the custom layer to sample from the distribution
sampler = Sampling()
z = sampler([z_mean, z_log_var])

# Build the decoder model
decoder_inputs = layers.Input(shape=(16,))
x = layers.Dense(32, activation='relu')(decoder_inputs)
x = layers.Dense(64, activation='relu')(x)
decoder_outputs = layers.Dense(4, activation='linear')(x)

# Combine the encoder and decoder into an autoencoder
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
decoder = keras.Model(decoder_inputs, decoder_outputs, name='decoder')
autoencoder_inputs = encoder_inputs
x = encoder(autoencoder_inputs)
autoencoder_outputs = decoder(x[2])

autoencoder = keras.Model(autoencoder_inputs, autoencoder_outputs, name='autoencoder')


def custom_loss(y_true, y_pred):
    reconstruction_loss = keras.losses.mean_squared_error(y_true, y_pred)
    kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    return reconstruction_loss + kl_loss

inputs = Input(shape=(timestamp.shape[1],))
encoded = Dense(64, activation='relu')(inputs)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(timestamp.shape[1], activation='sigmoid')(decoded)

# Wrap the custom loss in a Lambda layer
loss_layer = Lambda(lambda x: custom_loss(*x))

autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss=loss_layer)

# Train the model
autoencoder.fit(np.column_stack([timestamp, status, activity, activity_status]),
                np.column_stack([timestamp, status, activity, activity_status]),
                epochs=100,
                batch_size=128,
                validation_split=0.2)

processed_data_encoded = autoencoder.predict(np.column_stack([timestamp, status, activity, activity_status]))

# Decode the predictions and save them back to the original dataframe
processed_data_decoded = autoencoder.predict(processed_data_encoded)
processed_data['Timestamp'] = processed_data_decoded[:,0]
processed_data['Status'] = processed_data_decoded[:,1]
processed_data['Activity'] = processed_data_decoded[:,2]
processed_data['Activity Status'] = processed_data_decoded[:,3]

# Save the predictions to a CSV file
processed_data.to_csv('processed_data_predictions.csv', index=False)

# failed attempt 2


pd.options.display.max_rows = None
pd.options.display.max_columns = None

# Load the original dataset
processed_data = pd.read_csv('processed_data_converted.csv')

# Extract the relevant columns from the dataset
timestamp = processed_data['Timestamp'].values
device_id = processed_data['Device ID'].values
status = processed_data['Status'].values
activity = processed_data['Activity'].values
activity_status = processed_data['Activity Status'].values

# Normalize the data for training
scaler = MinMaxScaler()
timestamp = scaler.fit_transform(timestamp.reshape(-1, 1))
status = scaler.fit_transform(status.reshape(-1, 1))
activity = scaler.fit_transform(activity.reshape(-1, 1))
activity_status = scaler.fit_transform(activity_status.reshape(-1, 1))

# Build the encoder model
encoder_inputs = keras.Input(shape=(4,))
x = layers.Dense(64, activation='relu')(encoder_inputs)
x = layers.Dense(32, activation='relu')(x)
z_mean = layers.Dense(16, activation='linear')(x)
z_log_var = layers.Dense(16, activation='linear')(x)

# Create a custom layer for the sampling function
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Use the custom layer to sample from the distribution
sampler = Sampling()
z = sampler([z_mean, z_log_var])

# Build the decoder model
decoder_inputs = layers.Input(shape=(16,))
x = layers.Dense(32, activation='relu')(decoder_inputs)
x = layers.Dense(64, activation='relu')(x)
decoder_outputs = layers.Dense(4, activation='linear')(x)

# Combine the encoder and decoder into an autoencoder
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
decoder = keras.Model(decoder_inputs, decoder_outputs, name='decoder')
autoencoder = keras.Model(encoder_inputs, decoder(encoder(encoder_inputs)[2]), name='autoencoder')

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(np.column_stack([timestamp, device_id, status, activity]),
np.column_stack([timestamp, device_id, status, activity_status]),
epochs=50,
batch_size=128,
validation_split=0.1)

# Use the encoder to generate the latent representation of the data
latent_representation = encoder.predict(np.column_stack([timestamp, device_id, status, activity]))[0]

# Generate a new dataset based on the latent representation
generated_data = pd.DataFrame(latent_representation, columns=['Timestamp', 'Device ID', 'Status', 'Activity'])

# Save the generated dataset to a CSV file
generated_data.to_csv('generated_data.csv', index=False)

print('Generated dataset saved to generated_data.csv')


# failed attempt 3

pd.options.display.max_rows = None
pd.options.display.max_columns = None

# Load the original dataset
processed_data = pd.read_csv('processed_data_converted.csv')

# Extract the relevant columns from the dataset
timestamp = processed_data['Timestamp'].values
device_id = processed_data['Device ID'].values
status = processed_data['Status'].values
activity = processed_data['Activity'].values
activity_status = processed_data['Activity Status'].values

# Normalize the data for training
scaler = MinMaxScaler()
timestamp = scaler.fit_transform(timestamp.reshape(-1, 1))
status = scaler.fit_transform(status.reshape(-1, 1))
activity = scaler.fit_transform(activity.reshape(-1, 1))
activity_status = scaler.fit_transform(activity_status.reshape(-1, 1))

# Build the encoder model
encoder_inputs = keras.Input(shape=(4,))
x = layers.Dense(64, activation='relu')(encoder_inputs)
x = layers.Dense(32, activation='relu')(x)
z_mean = layers.Dense(16, activation='linear')(x)
z_log_var = layers.Dense(16, activation='linear')(x)

# Create a custom layer for the sampling function
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Use the custom layer to sample from the distribution
sampler = Sampling()
z = sampler([z_mean, z_log_var])

# Build the decoder model
decoder_inputs = layers.Input(shape=(16,))
x = layers.Dense(32, activation='relu')(decoder_inputs)
x = layers.Dense(64, activation='relu')(x)
decoder_outputs = layers.Dense(4, activation='linear')(x)

# Combine the encoder and decoder into an autoencoder
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
decoder = keras.Model(decoder_inputs, decoder_outputs, name='decoder')
autoencoder_inputs = encoder_inputs
x = encoder(autoencoder_inputs)
autoencoder_outputs = decoder(x[2])

autoencoder = keras.Model(autoencoder_inputs, autoencoder_outputs, name='autoencoder')


def custom_loss(y_true, y_pred):
    reconstruction_loss = keras.losses.mean_squared_error(y_true, y_pred)
    kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    return reconstruction_loss + kl_loss

inputs = Input(shape=(timestamp.shape[1],))
encoded = Dense(64, activation='relu')(inputs)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(timestamp.shape[1], activation='sigmoid')(decoded)

# Wrap the custom loss in a Lambda layer
loss_layer = Lambda(lambda x: custom_loss(*x))

autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss=loss_layer)

# Train the model
autoencoder.fit(np.column_stack([timestamp, status, activity, activity_status]),
                np.column_stack([timestamp, status, activity, activity_status]),
                epochs=100,
                batch_size=128,
                validation_split=0.2)

processed_data_encoded = autoencoder.predict(np.column_stack([timestamp, status, activity, activity_status]))

# Decode the predictions and save them back to the original dataframe
processed_data_decoded = autoencoder.predict(processed_data_encoded)
processed_data['Timestamp'] = processed_data_decoded[:,0]
processed_data['Status'] = processed_data_decoded[:,1]
processed_data['Activity'] = processed_data_decoded[:,2]
processed_data['Activity Status'] = processed_data_decoded[:,3]

# Save the predictions to a CSV file
processed_data.to_csv('processed_data_predictions.csv', index=False)