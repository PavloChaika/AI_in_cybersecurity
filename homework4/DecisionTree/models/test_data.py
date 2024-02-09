import csv
import random

# Generate some example data
random.seed(42)
num_samples = 100

# Features: feature1, feature2, and feature3
features = {
    "feature1": [random.uniform(0, 10) for _ in range(num_samples)],
    "feature2": [random.uniform(0, 10) for _ in range(num_samples)],
    "feature3": [random.uniform(0, 10) for _ in range(num_samples)],
}

# Target variable: target
target = [random.choice(["A", "B"]) for _ in range(num_samples)]

# Combine into a dataset
dataset = [{"feature1": f1, "feature2": f2, "feature3": f3, "target": t} for f1, f2, f3, t in zip(features["feature1"], features["feature2"], features["feature3"], target)]

# Save the dataset to a CSV file
csv_file_path = "test_dataset.csv"
header = ["feature1", "feature2", "feature3", "target"]

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(dataset)

print(f"CSV file '{csv_file_path}' created successfully.")