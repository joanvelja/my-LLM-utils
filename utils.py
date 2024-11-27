# Json to CSV conversion script

import csv
import json


def json_to_csv(json_file, csv_file):
    with open(json_file) as file:
        data = json.load(file)

    with open(csv_file, "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(data[0].keys())
        for row in data:
            csv_writer.writerow(row.values())
