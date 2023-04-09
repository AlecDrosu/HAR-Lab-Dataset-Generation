import pandas as pd

def read_data(file_path):
    return pd.read_csv(file_path, header=0, names=["date", "time", "device_id", "status", "activity", "activity_status"])

def segment_data_by_day(data_df):
    data_by_day = data_df.groupby(data_df["date"])
    daily_segments = []

    for _, day_data in data_by_day:
        daily_segments.append(day_data)

    return daily_segments

def sliding_window(daily_segments, overlap_ratio=0.5):
    windows = []

    for day_data in daily_segments:
        start = 0
        end = len(day_data)
        window_size = len(day_data)  # This will make the window_size equal to the number of sensor readings per day
        step_size = int(window_size * overlap_ratio)

        while start + window_size <= end:
            window = day_data.iloc[start:start+window_size, :]
            windows.append(window)
            start += step_size

    return windows