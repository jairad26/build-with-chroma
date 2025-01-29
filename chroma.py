# chroma.py
import json
import torch
from pathlib import Path
import chromadb
from tqdm import tqdm
import logging
from embedding_function import MyEmbeddingFunction

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ARXIV_PATH = Path('/Users/jairadhakrishnan/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/216/arxiv-metadata-oai-snapshot.json')

# Define AI-related categories we're interested in
AI_CATEGORIES = {
    'cs.AI',    # Artificial Intelligence
    'cs.LG',    # Machine Learning
    'cs.CL',    # Computation and Language
    'cs.CV',    # Computer Vision
    'cs.NE',    # Neural and Evolutionary Computing
    'stat.ML'   # Machine Learning (Statistics)
}

def load_ai_papers(file_path: Path):
    papers = []
    try:
        with open(file_path, 'r') as f:
            for line in tqdm(f, desc="Loading papers"):
                paper = json.loads(line)
                categories = set(paper.get('categories', '').split())
                if categories & AI_CATEGORIES:
                    cleaned_paper = {
                        'id': paper['id'],
                        'title': paper['title'].replace('\n', ' ').strip(),
                        'abstract': paper['abstract'].replace('\n', ' ').strip(),
                        'authors': paper['authors'],
                        'categories': list(categories),
                        'update_date': paper['update_date'],
                        'text_for_embedding': f"{paper['title']} {paper['abstract']}"
                    }
                    papers.append(cleaned_paper)
    except Exception as e:
        logger.error(f"Error loading papers: {str(e)}")
        raise
    
    return papers

def init_chroma(embedding_batch_size: int = 32):
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        
        embedding_fn = MyEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=embedding_batch_size
        )
        
        collection = client.get_or_create_collection(
            name="arxiv_ai_papers",
            embedding_function=embedding_fn,
            metadata={"description": "AI-related papers from ArXiv"}
        )
        
        return collection
    except Exception as e:
        logger.error(f"Error initializing ChromaDB: {str(e)}")
        raise

def main():
    # Determine optimal batch sizes based on GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        embedding_batch_size = min(128, int(gpu_mem * 32))  # Heuristic: 32 samples per GB
        chroma_batch_size = min(500, int(gpu_mem * 128))    # Larger batches for DB ops
    else:
        embedding_batch_size = 32
        chroma_batch_size = 100

    logger.info(f"Using embedding batch size: {embedding_batch_size}")
    logger.info(f"Using ChromaDB batch size: {chroma_batch_size}")

    try:
        logger.info("Loading AI papers...")
        papers = load_ai_papers(ARXIV_PATH)
        logger.info(f"Found {len(papers)} AI-related papers")
        
        collection = init_chroma(embedding_batch_size)
        
        # Get existing paper IDs to avoid duplicates
        existing_ids = set(collection.get()['ids']) if collection.count() > 0 else set()
        papers_to_add = [p for p in papers if p['id'] not in existing_ids]
        
        logger.info(f"Adding {len(papers_to_add)} new papers to ChromaDB")
        
        for i in tqdm(range(0, len(papers_to_add), chroma_batch_size), desc="Adding to ChromaDB"):
            batch = papers_to_add[i:i + chroma_batch_size]
            
            collection.add(
                ids=[p['id'] for p in batch],
                documents=[p['text_for_embedding'] for p in batch],
                metadatas=[{
                    'title': p['title'],
                    'authors': p['authors'],
                    'categories': ','.join(p['categories']),
                    'update_date': p['update_date']
                } for p in batch]
            )
            
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()