import re

# Open the input file for reading
with open("Raw Data/Aruba_17/data", "r") as f:
    data = f.readlines()

# Create an empty list to store the processed data
processed_data = []

# Possible activities
activities = ["Meal_Preparation", "Relax", "Eating", "Work", "Sleeping", "Wash_Dishes", "Bed_to_Toilet", "Enter_Home", "Leave_Home", "Housekeeping", "Respirate"]

# Loop through each line of the data
for line in data:
    # Split the line into its components
    components = re.split("\s+", line.strip())

    # Check if the line starts with a date
    if re.match("^\d{4}-\d{2}-\d{2}", components[0]):
        date = components[0]
        time = components[1]
        device_id = components[2]
        device_status = components[3]

        # Check if the device is a motion sensor (starts with "M")
        if device_id.startswith("M"):
            device_status = re.sub(r'[^ONOFF]', '', device_status)
            if device_status == "ON":
                if len(components) > 4:
                    activity = components[4] if components[4] in activities else None
                    activity_status = components[5] if len(components) > 5 else None
                    processed_data.append([date, time, device_id, device_status, activity, activity_status])
                else:
                    processed_data.append([date, time, device_id, device_status])
            else:
                processed_data.append([date, time, device_id, device_status])
        # Check if the device is a door sensor (starts with "D")
        elif device_id.startswith("D"):
            if device_status in ["OPEN", "CLOSE"]:
                processed_data.append([date, time, device_id, device_status])
        # Check if the device is a temperature sensor (starts with "T")
        elif device_id.startswith("T"):
            temperature = re.sub(r'[^\d.]', '', device_status)
            temperature = float(temperature)
            processed_data.append([date, time, device_id, temperature, "", ""])
# If the line does not start with a date, it must be a temperature reading
    else:
        device_id = components[0]
        temperature = float(components[1])
        processed_data.append([device_id, temperature])

# Write the processed data to a CSV file

with open("Processed Data/Aruba_17/processed_data.csv", "w") as f:
    f.write("Date,Time,Device ID,Status,Activity,Activity Status\n")
    for row in processed_data:
        date = row[0] if len(row) > 0 else ""
        time = row[1] if len(row) > 1 else ""
        device_id = row[2] if len(row) > 2 else ""
        status = row[3] if len(row) > 3 else ""
        activity = row[4] if len(row) > 4 else ""
        activity_status = row[5] if len(row) > 5 else ""
        f.write(f"{date},{time},{device_id},{status},{activity},{activity_status}\n")