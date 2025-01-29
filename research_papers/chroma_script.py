import networkx as nx
from node2vec import Node2Vec
import torch
import json
from pathlib import Path
from tqdm import tqdm
import logging
import re
from research_papers.research_chroma import AI_CATEGORIES, ARXIV_PATH  # Import from your existing code
import chromadb
from typing import Dict, List, Set, Optional


chroma_client = chromadb.PersistentClient(path="./chroma_db")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
n2v_collection = chroma_client.get_or_create_collection("arxiv_ai_node2vec",
    metadata={"description": "Node2vec embeddings and supplementary data for AI papers"})

class GraphCreator:
    def __init__(self, collection):
        self.collection = collection
        self.ai_paper_ids = set()
        self.graph = nx.DiGraph()
        self.arxiv_path = ARXIV_PATH

    def build_graph(self):
        """Build citation network from ArXiv papers, focusing on AI papers."""
        logger.info("Building AI papers citation network...")

        # First pass: identify AI papers
        logger.info("First pass: identifying AI papers...")
        with open(self.arxiv_path, 'r') as f:
            for line in tqdm(f, desc="Identifying AI papers"):
                paper = json.loads(line)
                if 'id' in paper and self.is_ai_paper(paper.get('categories', '')):
                    self.ai_paper_ids.add(paper['id'])
                    self.graph.add_node(
                        paper['id'],
                        title=paper.get('title', ''),
                        categories=paper.get('categories', '').split()
                    )

        logger.info(f"Found {len(self.ai_paper_ids)} AI-related papers")
        
        # Second pass: add citations between AI papers
        logger.info("Second pass: adding citations...")
        with open(self.arxiv_path, 'r') as f:
            for line in tqdm(f, desc="Processing citations"):
                paper = json.loads(line)
                
                # Only process if it's an AI paper
                if 'id' not in paper or paper['id'] not in self.ai_paper_ids:
                    continue
                
                # Extract citations from abstract and comments
                text_to_search = f"{paper.get('abstract', '')} {paper.get('comments', '')}"
                citations = self.extract_citations(text_to_search)
                
                # Add edges only between AI papers
                for cited_id in citations:
                    if cited_id in self.ai_paper_ids:
                        self.graph.add_edge(paper['id'], cited_id)
        
        # Remove isolated nodes (AI papers with no connections to other AI papers)
        isolated_nodes = list(nx.isolates(self.graph))
        self.graph.remove_nodes_from(isolated_nodes)

        logger.info(f"Final graph has {self.graph.number_of_nodes()} nodes and "
                   f"{self.graph.number_of_edges()} edges")

        # Basic graph statistics
        logger.info("\nGraph Statistics:")
        logger.info(f"Number of connected components: {nx.number_connected_components(self.graph.to_undirected())}")
        logger.info(f"Average degree: {sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes():.2f}")

    def compute_node2vec(self, dimensions=128, walk_length=30, num_walks=200):
        """Compute node2vec embeddings and store with supplementary metadata"""
        
        # Initialize and train node2vec model
        node2vec = Node2Vec(
            self.graph,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=4
        )
        
        model = node2vec.fit(window=10, min_count=1, batch_words=4)

        # Store embeddings and supplementary data in batches
        batch_size = 500
        nodes = list(self.graph.nodes())

        for i in range(0, len(nodes), batch_size):
            batch_nodes = nodes[i:i + batch_size]
            batch_embeddings = [model.wv[node].tolist() for node in batch_nodes]

            # Get citations for each paper in batch
            batch_citations = []
            for node in batch_nodes:
                citations = [succ for succ in self.graph.successors(node)]
                batch_citations.append(citations)

            self.collection.add(
                ids=batch_nodes,
                embeddings=batch_embeddings,
                documents=batch_nodes,  # Store paper ID as document
                metadatas=[{
                    'citations': json.dumps(citations),
                    # Add any other supplementary data not in papers_collection
                    'out_degree': len(citations),
                    'in_degree': len([pred for pred in self.graph.predecessors(node)])
                } for node, citations in zip(batch_nodes, batch_citations)]
            )

        logger.info(f"Stored {len(nodes)} node2vec embeddings with supplementary data")

    
    def extract_citations(self, text: str) -> list:
        """Extract arXiv IDs from text using regex patterns."""
        patterns = [
            r'arxiv:(\d{4}\.\d{4,5})',  # New format: 2101.12345
            r'(\d{4}\.\d{4,5})',        # New format without prefix
            r'arxiv:([a-z\-]+/\d{7})',  # Old format: quant-ph/0101001
            r'([a-z\-]+/\d{7})'         # Old format without prefix
        ]
        
        citations = set()
        for pattern in patterns:
            matches = re.finditer(pattern, text.lower())
            citations.update(match.group(1) for match in matches)
        return list(citations)
    
    def is_ai_paper(self, categories: str) -> bool:
        """Check if paper belongs to AI-related categories."""
        paper_categories = set(categories.split())
        return bool(paper_categories & AI_CATEGORIES)

def main():
    # graph_creator = GraphCreator(n2v_collection)
    # graph_creator.build_graph()
    # graph_creator.compute_node2vec()
    logger.info(n2v_collection.peek(limit=5))

if __name__ == "__main__":
    main()