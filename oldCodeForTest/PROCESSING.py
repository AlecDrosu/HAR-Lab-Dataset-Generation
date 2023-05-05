import re
from datetime import datetime
from datetime import timedelta
import time
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# This code will convert the original data file into a csv for the Aruba 
# dataset. Saved as pre_processed_data.csv
def time_to_seconds(time_str):
    try:
        dt = datetime.strptime(time_str, '%H:%M:%S.%f')
    except ValueError:
        dt = datetime.strptime(time_str, '%H:%M:%S')
    
    total_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
    return total_seconds

def pre_processed_data(filename, output_filename):
    # Open the input file for reading
    with open(filename, "r") as f:
        data = f.readlines()

    # Create an empty list to store the processed data
    processed_data = []

    # Possible activities

    sensors = ['D001', 'D002', 'D004', 'M001', 'M002', 'M003', 'M004', 'M005', 'M006', 'M007', 'M008', 'M009', 'M010', 'M011', 'M012', 'M013', 'M014', 'M015', 'M016', 'M017', 'M018', 'M019', 'M020', 'M021', 'M022', 'M023', 'M024', 'M025', 'M026', 'M027', 'M028', 'M029', 'M030', 'M031', 'T001', 'T002', 'T003', 'T004', 'T005']
    statuses = ['16', '16.5', '17', '17.5', '18', '18.5', '19', '19.5', '20', '20.5', '21', '21.5', '22', '22.5', '23', '23.5', '24', '24.5', '25', '25.5', '26', '26.5', '27', '27.5', '28', '28.5', '29', '29.5', '30', '30.5', '31', '31.5', '32', '32.5', '33', '33.5', '34', '34.5', '35', '35.5', '36', '36.5', '37', '37.5', '38', '38.5', '39', '39.5', '40.5', '41.5', '42', '42.5', '43', 'CLOSE', 'OFF', 'ON', 'OPEN']
    activities = ['Bed_to_Toilet', 'Eating', 'Enter_Home', 'Housekeeping', 'Leave_Home', 'Meal_Preparation', 'Relax', 'Respirate', 'Sleeping', 'Wash_Dishes', 'Work']
    activity_statuses = ['begin', 'end']

    # Loop through each line of the data
    for line in data:
        # Split the line into its components
        components = re.split("\s+", line.strip())
        date = components[0]
        time = components[1]
        device_id = components[2]
        device_status = components[3]
        if len(components) > 4:
            activity = components[4]
            activity_status = components[5]

        formatted_date = int(date.replace("-", ""))
        # formatted_time = int(time.replace(":", "")[:6] + time.replace(":", "")[7:])
        formatted_time = time_to_seconds(time)

        if device_id.startswith("M"):
            if device_status.startswith("ON"):
                device_status = "ON"
            elif device_status.startswith("OFF"):
                device_status = "OFF"

        if device_id not in sensors or device_status not in statuses:
            continue 

        combined_device = f"{device_id}_{device_status}"
        combined_activity = f"{activity}_{activity_status}" if len(components) > 4 else ""
        # Append the processed data to the list
        if len(components) > 4:
            if activity not in activities or activity_status not in activity_statuses:
                continue
            processed_data.append([formatted_date, formatted_time, combined_device, combined_activity])
        else:
            processed_data.append([formatted_date, formatted_time, combined_device, ""])

    # Write the processed data to a new file
    with open(output_filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Time", "Device_Status", "Activity"])
        for data in processed_data:
            writer.writerow(data)

    return

# The original data is saved in a way that the model cannot use. 
# The model needs the data to be numerical. This code will convert the previous file 
# into a csv that the model can be trained on. Saved as processed_data.csv

def model_processing_code(filename, output_filename):
    # Load the processed data file
    data = pd.read_csv(filename)

    # Encode the following columns: Timestamp,Device ID,Status,Activity,Activity Status

    device_id_and_status_encoder = LabelEncoder()
    activity_and_status_encoder = LabelEncoder()

    device_id_and_status_encoder.fit(data['Device_Status'])
    activity_and_status_encoder.fit(data['Activity'])

    device_id_and_status_mapping = dict(zip(device_id_and_status_encoder.classes_, device_id_and_status_encoder.transform(device_id_and_status_encoder.classes_)))
    activity_and_status_mapping = dict(zip(activity_and_status_encoder.classes_, activity_and_status_encoder.transform(activity_and_status_encoder.classes_)))

    data['Device_Status'] = device_id_and_status_encoder.transform(data['Device_Status'])
    data['Activity'] = activity_and_status_encoder.transform(data['Activity'])

    data.to_csv(output_filename, index=False)

    encoders_and_mappings = {
        'device_id_and_status_encoder': device_id_and_status_mapping,
        'activity_and_status_encoder': activity_and_status_mapping,
    }
    return encoders_and_mappings

# This code will inport the prediction data from the model. It will then convert the Label Encoded data back into the original labels. Saved as COMPLETE_PREDICTION.csv

def model_post_processing(filename, output_filename, encoders_and_mappings):

    def round_and_inverse_transform(value, mapping, encoder):
        max_val = len(mapping) - 1
        rounded_val = round(value)
        clipped_val = min(max(rounded_val, 0), max_val)
        return encoder.inverse_transform([clipped_val])[0]

    device_id_and_status_encoder = LabelEncoder()
    activity_and_status_encoder = LabelEncoder()

    device_id_and_status_mapping = encoders_and_mappings['device_id_and_status_encoder']
    activity_and_status_mapping = encoders_and_mappings['activity_and_status_encoder']

    device_id_and_status_encoder.classes_ = np.array(list(device_id_and_status_mapping.keys()))
    activity_and_status_encoder.classes_ = np.array(list(activity_and_status_mapping.keys()))

    data = pd.read_csv(filename)
    data.columns = ['Date', 'Time', 'Device_Status', 'Activity']

    data['Device_Status'] = data['Device_Status'].apply(round_and_inverse_transform, args=(device_id_and_status_mapping, device_id_and_status_encoder))
    data['Activity'] = data['Activity'].apply(round_and_inverse_transform, args=(activity_and_status_mapping, activity_and_status_encoder))

    # save the data to a new file
    data.to_csv(output_filename, index=False)
    return

def print_mappings(device_id_and_status_mappings, activity_and_status_mappings):
    d_values = []
    t_values = []
    m_values = []
    other_values = []

    for key, value in device_id_and_status_mappings.items():
        if key[0] == 'D':
            d_values.append((key, value))
        elif key[0] == 'T':
            t_values.append((key, value))
        elif key[0] == 'M':
            m_values.append((key, value))

    for key, value in activity_and_status_mappings.items():
        other_values.append((key, value))

    def print_first_and_last_values(title, values):
        print(f"{title}:")
        if len(values) > 0:
            print(f"{values[0][0]}: {values[0][1]}")
            if len(values) > 1:
                print(f"{values[-1][0]}: {values[-1][1]}")
            print(f"values {values[0][1]} through {values[-1][1]}\n")

    print_first_and_last_values("D Values", d_values)
    print_first_and_last_values("M Values", m_values)
    print_first_and_last_values("T Values", t_values)
    print_first_and_last_values("Other Values", other_values)

def reconstructed_data(predicted_windows_with_date_time, window_size, overlap_ratio):
    step_size = int(window_size * overlap_ratio)
    daily_segments = []

    # Add all data from the first window
    daily_segments.append(predicted_windows_with_date_time[0])

    # For the rest of the windows, only add non-overlapping data
    for window in predicted_windows_with_date_time[1:]:
        daily_segment = window[step_size:, :]
        daily_segments.append(daily_segment)

    # Concatenate all daily segments to form the original dataset format
    original_data_format = np.concatenate(daily_segments, axis=0)

    return original_data_format

# Read the CSV file
def undo_split(filename, output_filename):
    df = pd.read_csv(filename)

    # Separate the Device_Status column
    df[['Device ID', 'Device Status']] = df['Device_Status'].str.split('_', expand=True)
    df.drop('Device_Status', axis=1, inplace=True)

    # Add new columns for Activity and Activity Status
    df['Activity Status'] = None

    # Iterate through the rows and split the 'Activity' column at the last underscore
    for index, row in df.iterrows():
        if pd.notna(row['Activity']) and '_' in row['Activity']:
            parts = row['Activity'].rsplit('_', 1)
            activity, status = parts[0], parts[1]
            df.at[index, 'Activity'] = activity
            df.at[index, 'Activity Status'] = status

        # Transform the date format
        date = str(row['Date']).split('.')[0]  # Remove decimal part if exists
        year, month, day = date[:4], date[4:6], date[6:]
        df.at[index, 'Date'] = f'{year}-{month}-{day}'

        # Transform the time format
        time = float(row['Time'])
        seconds = int(time)
        milliseconds = int((time - seconds) * 1e3)  # Convert fractional part to milliseconds
        if milliseconds == 0:
            milliseconds = 1  # Use 1 millisecond if there are no milliseconds
        time_duration = timedelta(seconds=seconds, milliseconds=milliseconds)
        df.at[index, 'Time'] = str(time_duration)[:-3]  # Remove the last 3 characters (microseconds)

    # Rearrange the columns
    df = df[['Date', 'Time', 'Device ID', 'Device Status', 'Activity', 'Activity Status']]

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_filename, index=False)
    return