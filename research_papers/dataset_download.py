import kagglehub
import json
import pandas as pd
from pathlib import Path

# Download the dataset
path = kagglehub.dataset_download("Cornell-University/arxiv")
print("Path to dataset files:", path)

# The dataset is typically downloaded as a list of paths
# Let's load the metadata file (it should be the largest JSON file)
def find_metadata_file(paths):
    paths = [Path(p) for p in paths]
    json_files = [p for p in paths if p.suffix == '.json']
    # Get the largest JSON file as it's likely the metadata file
    return max(json_files, key=lambda p: p.stat().st_size)

metadata_file = find_metadata_file(path)

# Test load a few papers to see the structure
def peek_data(file_path, n_lines=5):
    papers = []
    with open(file_path, 'r') as f:
        for _ in range(n_lines):
            papers.append(json.loads(f.readline()))
    return papers

sample_papers = peek_data(metadata_file)
print("\nSample paper structure:")
print(json.dumps(sample_papers[0], indent=2))