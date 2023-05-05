# with open('../Predictions/Aruba_0501.txt', 'r') as input_file:
#     lines = input_file.readlines()

# # Remove header and convert to raw data format
# lines = lines[1:]  # Remove header
# formatted_lines = []
# for line in lines:
#     parts = line.strip().split(',')
#     formatted_line = f"{parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4]} {parts[5]}"
#     formatted_lines.append(formatted_line)

# # Append the formatted lines to the raw data file
# with open('./dataset/aruba_combined', 'a') as raw_file:
#     for line in formatted_lines:
#         raw_file.write(f"{line}\n")

with open('../Predictions/Aruba_0501.txt', 'r') as input_file:
    lines = input_file.readlines()

# Remove header and convert to raw data format
lines = lines[1:]  # Remove header
formatted_lines = []
for line in lines:
    parts = line.strip().split(',')
    formatted_line = f"{parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4]} {parts[5]}"
    formatted_lines.append(formatted_line)

# Create a new raw data file and write the formatted lines
new_raw_file_name = './dataset/aruba'
with open(new_raw_file_name, 'w') as new_raw_file:
    for line in formatted_lines:
        new_raw_file.write(f"{line}\n")