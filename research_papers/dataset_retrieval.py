import json
import pandas as pd
from pathlib import Path

ARXIV_PATH = Path('/Users/jairadhakrishnan/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/217/arxiv-metadata-oai-snapshot.json')

def peek_data(file_path, n_lines=5):
    papers = []
    with open(file_path, 'r') as f:
        for _ in range(n_lines):
            papers.append(json.loads(f.readline()))
    return papers

# Let's look at the structure first
sample_papers = peek_data(ARXIV_PATH)
print("\nSample paper structure:")
print(json.dumps(sample_papers[0], indent=2))

# Count total number of papers (this might take a minute)
def count_papers():
    with open(ARXIV_PATH, 'r') as f:
        for i, _ in enumerate(f):
            pass
    return i + 1

total_papers = count_papers()
print(f"\nTotal number of papers: {total_papers}")

# Let's also see what categories we have
def analyze_categories(n_papers=10000):  # Sample first 10k papers
    categories = {}
    with open(ARXIV_PATH, 'r') as f:
        for i, line in enumerate(f):
            if i >= n_papers:
                break
            paper = json.loads(line)
            cats = paper.get('categories', '').split()
            for cat in cats:
                categories[cat] = categories.get(cat, 0) + 1
    return categories

print("\nAnalyzing categories from sample...")
categories = analyze_categories()
print("\nTop 10 categories:")
for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{cat}: {count} papers")