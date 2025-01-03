import csv
import os

def read_csv(file_path):
    data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    return data

#select and read input files
input_files = ['input_walking.csv', 'input_sitting.csv', 'input_standing.csv', 'input_empty.csv']
data_arrays = []
for file_path in input_files:
    activity = file_path.split('_')[1].split('.')[0]
    data = read_csv(file_path)
    data_arrays.append((activity, data))

def write_data_header(data_arrays, output_dir):
    for activity, data in data_arrays:
        activity_name = activity.replace('_', '')
        header_file_name = f"{activity_name}_data.h"
        header_file_path = os.path.join(output_dir, header_file_name)

        with open(header_file_path, 'w') as f:
            f.write("#ifndef {}\n".format(activity_name.upper()))
            f.write("#define {}\n\n".format(activity_name.upper()))

            f.write(f"// Data for {activity_name}\n")
            f.write(f"const float {activity_name}_data[][num_columns] = {{\n")
            for row in data:
                f.write("  {")
                f.write(", ".join(row))
                f.write("},\n")
            f.write("};\n\n")

            f.write("#endif // {}\n".format(activity_name.upper()))

#specify columns in files
num_columns = 52

#create header files of input data.
output_dir = "data_arrays"
os.makedirs(output_dir, exist_ok=True)
write_data_header(data_arrays, output_dir)