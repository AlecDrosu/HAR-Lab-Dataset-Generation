import numpy as np
import pandas as pd

class DataPredictor:
    def __init__(self, vae_model, encoder_model, scaler, batch_size):
        self.vae_model = vae_model
        self.encoder_model = encoder_model
        self.scaler = scaler
        self.batch_size = batch_size
    
    def predict_data(self, processed_data, timesteps, input_dim):
        # Generate a fake dataset using the VAE model
        n_samples = len(processed_data)

        noise = np.random.normal(size=(n_samples, timesteps, input_dim))
        predicted_values = self.vae_model.predict(noise, batch_size=self.batch_size)
        # reshape predicted values to have the correct shape
        predicted_values = np.reshape(predicted_values, (n_samples, timesteps, input_dim))

        # undo the normalization
        predicted_values = np.reshape(predicted_values, (-1, input_dim))
        predicted_values = self.scaler.inverse_transform(predicted_values)

        # Round each of the values in the array to the nearest integer
        predicted_values = np.rint(predicted_values)

        # Reshape predicted_values to match the input shape of encoder_model
        predicted_values = np.reshape(predicted_values, (n_samples, timesteps, input_dim))

        # # Assign cluster labels to each of the predicted values
        # y_pred = classifier.predict(self.encoder_model.predict(predicted_values, batch_size=self.batch_size))
        # # Print all information of the y_pred line above

        # # Reshape y_pred to match the shape of predicted_values
        # y_pred = np.reshape(y_pred, (n_samples, timesteps))

        # Save the prediction data to a new file 'predicted_Data.csv'
        predicted_data = pd.DataFrame(predicted_values.reshape((-1, input_dim)), columns=['Timestamp', 'Device ID', 'Status', 'Activity', 'Activity Status'])
        # predicted_data['Cluster'] = y_pred.reshape(-1)
        predicted_data.to_csv('Predictions/Aruba_17_prediction.csv', index=False)