import chromadb
from embedding_function import MyEmbeddingFunction
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    embedding_fn = MyEmbeddingFunction()
    collection = client.get_collection(
        name="arxiv_ai_papers",
        embedding_function=embedding_fn
    )
    return collection

def main():
    collection = init_collection()
    
    # 1. Check total number of documents
    count = collection.count()
    logger.info(f"Total documents in collection: {count}")
    
    # 2. Get a sample of documents to verify content
    sample = collection.get(limit=5)
    logger.info("\n\n---------------\nSample documents:")
    for i in range(len(sample['ids'])):
        logger.info(f"\nID: {sample['ids'][i]}")
        logger.info(f"Title: {sample['metadatas'][i]['title']}")
        logger.info(f"Categories: {sample['metadatas'][i]['categories']}")
    
    # 3. Try some similarity searches
    search_queries = [
        "large language models and their applications",
        "computer vision deep learning",
        "reinforcement learning in games"
    ]
    
    logger.info("\n---------------\nTesting similarity searches:")
    for query in search_queries:
        logger.info(f"\nQuery: {query}")
        results = collection.query(
            query_texts=[query],
            n_results=3
        )
        
        for i in range(len(results['ids'][0])):
            logger.info("\n---------------")
            logger.info(f"\nMatch {i+1}:")
            logger.info(f"Title: {results['metadatas'][0][i]['title']}")
            logger.info(f"Categories: {results['metadatas'][0][i]['categories']}")

if __name__ == "__main__":
    main()