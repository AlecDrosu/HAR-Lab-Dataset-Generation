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
# Create RMSE
rmse = np.sqrt(mse)

print("MSE:", mse)
print("MAE:", mae)
print("RMSE:", rmse)

# Calculate the KL Divergence. 
# KL divergence is typically applied to each feature independently; 
# it is not designed as a covariant feature measure but rather a metric that shows how each feature has diverged independently from the baseline values.
predicted_dataset.columns = ['Timestamp', 'Device ID', 'Status', 'Activity', 'Activity Status']
# Calculate the KL Divergence of each feature independently
kl_divergence = []
for column in original_dataset.columns:
    original_column_values = original_dataset[column]
    predicted_column_values = predicted_dataset[column]
    kl_divergence.append(entropy(original_column_values, predicted_column_values))

# Format the output as a table
kl_divergence_table = pd.DataFrame({'Feature': original_dataset.columns, 'KL Divergence': kl_divergence})
print(kl_divergence_table)

# Plot the KL Divergence for each feature
import matplotlib.pyplot as plt
plt.bar(original_dataset.columns, kl_divergence)
plt.title('KL Divergence for Each Feature')
plt.xlabel('Feature')
plt.ylabel('KL Divergence')
plt.show()


