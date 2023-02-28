import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.spatial.distance import cosine, pdist
import pandas as pd
from scipy.stats import entropy

original_dataset = pd.read_csv('Processed Data/Aruba_17/processed_data.csv')
predicted_dataset = pd.read_csv('Predictions/Aruba_17_prediction.txt', delimiter=',')
original_dataset = original_dataset.head(1000)
predicted_dataset = predicted_dataset.head(1000)


# Calculate MSE and MAE
mse = mean_squared_error(original_dataset, predicted_dataset)
mae = mean_absolute_error(original_dataset, predicted_dataset)

print("MSE:", mse)
print("MAE:", mae)

# Calculate KLD
kld = entropy(original_dataset.values.flatten(), predicted_dataset.values.flatten())

print("KLD:", kld)