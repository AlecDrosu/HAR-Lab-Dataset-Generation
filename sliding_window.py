import pandas as pd
import numpy as np

def read_data(file_path):
    return pd.read_csv(file_path, header=0, names=["date", "time", "device_id", "status", "activity", "activity_status"])

def segment_data_by_day(data_df):
    data_by_day = data_df.groupby(data_df["date"])
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