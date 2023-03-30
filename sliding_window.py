import pandas as pd
import numpy as np

def preprocess_data(data):
    data_list = []

    for line in data:
        columns = line.strip().split()
        timestamp = pd.to_datetime(columns[0] + ' ' + columns[1])
        device_id = columns[2]
        status = columns[3]
        activity = columns[4] if len(columns) > 4 else None
        activity_status = columns[5] if len(columns) > 5 else None

        data_list.append([timestamp, device_id, status, activity, activity_status])

    return pd.DataFrame(data_list, columns=['timestamp', 'device_id', 'status', 'activity', 'activity_status'])

def segment_data_by_day(data_df):
    data_by_day = data_df.groupby(data_df['timestamp'].dt.date)
    daily_segments = []

    for _, day_data in data_by_day:
        daily_segments.append(day_data)

    return daily_segments

def sliding_window(daily_segments, step_size=100):
    windows = []

    for day_data in daily_segments:
        start = 0
        end = len(day_data)

        while start + step_size <= end:
            window = day_data.iloc[start:start+step_size, :]
            windows.append(window)
            start += step_size

    return windows

with open("Raw Data/Aruba_17/data", "r") as f:
    data = f.readlines()

data_df = preprocess_data(data)
daily_segments = segment_data_by_day(data_df)
windows = sliding_window(daily_segments, step_size=100)

# print the first thousand lines of the first window
print(windows[0].head(1000))