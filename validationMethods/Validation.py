import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

FILE_PATH = 'Processed Data/Aruba_17/pre_processed_data.csv'

# Read data
data_df = pd.read_csv(FILE_PATH, names=['Date', 'Time', 'Device ID', 'Status', 'Activity', 'Activity Status'])

# Forward-fill the 'Activity' column so that each sensor event has an associated activity
data_df['Activity'] = data_df['Activity'].fillna(method='ffill')

# Filter the data by sensor type
door_sensors = data_df[data_df['Device ID'].str.startswith('D')]
door_sensors = door_sensors[~door_sensors['Device ID'].str.contains('Device ID')]
motion_sensors = data_df[data_df['Device ID'].str.startswith('M')]
temp_sensors = data_df[data_df['Device ID'].str.startswith('T')]

sensor_types = [door_sensors, motion_sensors, temp_sensors]
titles = ['Door Sensors', 'Motion Sensors', 'Temperature Sensors']

# Create separate heatmaps for each sensor type
# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))
# Change the size to make the plots square
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(4, 8))
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
for i, (sensor_data, title) in enumerate(zip(sensor_types, titles)):
    # Create a pivot table to count the frequency of each sensor event for each activity
    pivot_table = sensor_data.pivot_table(index='Activity', columns='Device ID', values='Status', aggfunc='count', fill_value=0)
    # Normalize the pivot table by row to show the proportion of sensor events for each activity
    normalized_pivot_table = pivot_table.div(pivot_table.sum(axis=1), axis=0)
    
    # Create a heatmap
    sns.heatmap(normalized_pivot_table, cmap='plasma', annot=False, fmt='.5f', linewidths=.5, vmin=0, vmax=0.5, robust=True, ax=axes[i], cbar=i == 0, cbar_ax=None if i else cbar_ax)
    
    # Add labels
    axes[i].set_xlabel('Sensor ID')
    axes[i].set_ylabel('Activity' if i == 0 else '')
    axes[i].set_title(title)

    # Adjust x-axis labels
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=30, ha='right')
        

# Show the plot
plt.tight_layout()
plt.show()

# Create a combined pivot table with all sensor types
combined_pivot_table = pd.concat(sensor_types).pivot_table(index='Activity', columns='Device ID', values='Status', aggfunc='count', fill_value=0)

# Normalize the pivot table by row
normalized_combined_pivot_table = combined_pivot_table.div(combined_pivot_table.sum(axis=1), axis=0)

# Create a distance matrix (1 - correlation)
distance_matrix = 1 - normalized_combined_pivot_table.corr()

# Apply MDS
embedding = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
mds_coordinates = embedding.fit_transform(distance_matrix)

# Create a dataframe with MDS coordinates and sensor labels
mds_df = pd.DataFrame(mds_coordinates, columns=['x', 'y'])
mds_df['Sensor'] = distance_matrix.columns

# Calculate the average coordinates for each activity
activity_coordinates = pd.DataFrame(index=normalized_combined_pivot_table.index, columns=['x', 'y'], dtype=float)

for activity in activity_coordinates.index:
    activity_coordinates.loc[activity] = (normalized_combined_pivot_table.loc[activity] * mds_df.set_index('Sensor')).sum() / normalized_combined_pivot_table.loc[activity].sum()
activity_coordinates /= normalized_combined_pivot_table.sum(axis=1)[:, None]

# Create a scatter plot
plt.figure(figsize=(12, 8))

# Plot motion sensors with the same color
motion_sensor_mask = mds_df['Sensor'].str.startswith('M')
sns.scatterplot(data=mds_df[motion_sensor_mask], x='x', y='y', color='blue', label='Motion Sensors')

# Plot door and temperature sensors with different colors
non_motion_mask = ~motion_sensor_mask
sns.scatterplot(data=mds_df[non_motion_mask], x='x', y='y', hue='Sensor', palette='Set1', legend=False)

# Plot activities
sns.scatterplot(data=activity_coordinates.reset_index(), x='x', y='y', hue='Activity', palette='tab10', marker='*', s=200, legend=False)

# Add labels
for i, sensor in enumerate(mds_df['Sensor']):
    plt.text(mds_df.loc[i, 'x'], mds_df.loc[i, 'y'], sensor, fontsize=9, ha='center', va='center')

for i, activity in enumerate(activity_coordinates.index):
    plt.text(activity_coordinates.loc[activity, 'x'], activity_coordinates.loc[activity, 'y'], activity, fontsize=9, ha='center', va='center', fontweight='bold')

plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.title('MDS Plot of Activities and Sensors')
plt.legend(title='Sensor Type')

# Show the plot
plt.show()