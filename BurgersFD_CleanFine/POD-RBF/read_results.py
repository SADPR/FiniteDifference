import re
import csv

# Define the input and output file paths
input_file = "output_55034827.log"
output_csv = "parsed_data.csv"

# Updated regex pattern to match both integer and decimal values for epsilon
pattern = re.compile(r"(\w+)\sKernel\s\|\sepsilon=([0-9.]+),\sneighbors=(\d+):\sPOD-RBF\sReconstruction\serror:\s([0-9.e-]+)")

# Initialize a list to store the extracted data
data = []

# Open and read the log file
with open(input_file, 'r') as file:
    for line in file:
        # Search for matches in each line
        match = pattern.search(line)
        if match:
            kernel, epsilon, neighbors, error = match.groups()
            data.append([kernel, epsilon, neighbors, error])

# Write the extracted data to a CSV file
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow(["Kernel", "Epsilon", "Neighbors", "Reconstruction Error"])
    # Write the data rows
    writer.writerows(data)

print(f"Data has been parsed and saved to {output_csv}")

