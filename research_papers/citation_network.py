import networkx as nx
from node2vec import Node2Vec
import torch
import json
from pathlib import Path
from tqdm import tqdm
import logging
import re
from research_papers.research_chroma import AI_CATEGORIES  # Import from your existing code
import chromadb
from typing import Dict, List, Set, Optional



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CitationNetwork:
    def __init__(self, arxiv_path: Optional[Path] = None):
        self.arxiv_path = arxiv_path
        self.graph = nx.DiGraph()
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")

    def build_graph(self):
        """Public method to build or load the graph."""
        self.load_graph()
        return self.graph

    def load_graph(self):
        """Load graph from both ChromaDB collections or build from scratch"""
        try:
            # Try to get both collections
            papers_collection = self.chroma_client.get_or_create_collection("arxiv_ai_papers")
            # Then try to get node2vec collection
            node2vec_collection = self.chroma_client.get_or_create_collection("arxiv_ai_node2vec")
            if node2vec_collection.count() > 0:
                logger.info("Found existing data in both collections, rebuilding graph...")
                self._rebuild_graph_from_collections(papers_collection, node2vec_collection)

        except ValueError as e:
            logger.info(f"ChromaDB collections not found: {e}")

    def _rebuild_graph_from_collections(self, papers_collection, node2vec_collection):
        """Rebuild graph using data from both collections"""
        # Get all papers with their basic metadata
        papers = papers_collection.get(
            include=['metadatas'],
            limit=papers_collection.count()
        )

        # Get supplementary data (citations, etc.)
        node2vec_data = node2vec_collection.get(
            include=['metadatas'],
            limit=node2vec_collection.count()
        )

        # Create lookup for node2vec metadata
        node2vec_metadata = {
            paper_id: metadata
            for paper_id, metadata in zip(node2vec_data['ids'], node2vec_data['metadatas'])
        }

        # Build graph combining both collections' data
        for paper_id, metadata in zip(papers['ids'], papers['metadatas']):
            # Add node with basic metadata from papers collection
            self.graph.add_node(
                paper_id,
                title=metadata.get('title', ''),
                categories=metadata.get('categories', '').split(','),
                authors=metadata.get('authors', ''),
                update_date=metadata.get('update_date', '')
            )

            # Add citation edges from node2vec collection
            if paper_id in node2vec_metadata:
                citations = json.loads(node2vec_metadata[paper_id].get('citations', []))
                for cited_id in citations:
                    if cited_id in papers['ids']:  # Only add edges between known papers
                        self.graph.add_edge(paper_id, cited_id)

        logger.info(f"Rebuilt graph with {self.graph.number_of_nodes()} nodes and "
                   f"{self.graph.number_of_edges()} edges from ChromaDB")

    
    def get_similar_papers(self, paper_id: str, n: int = 5):
        """Find similar papers using node2vec embeddings"""
        try:
            node2vec_collection = self.chroma_client.get_collection("arxiv_ai_node2vec")
            papers_collection = self.chroma_client.get_collection("arxiv_ai_papers")

            paper_n2v = node2vec_collection.get(
                ids=[paper_id],
                include=['embeddings']
            )

            paper_embedding = paper_n2v['embeddings'][0]

            # Query node2vec collection for similar papers
            results = node2vec_collection.query(
                query_embeddings=paper_embedding,
                n_results=n + 1
            )

            similar_papers = []
            for i, node_id in enumerate(results['ids'][0]):
                if node_id != paper_id:
                    # Get paper metadata from papers collection
                    paper_data = papers_collection.get(
                        ids=[node_id],
                        include=['metadatas']
                    )

                    if paper_data['metadatas']:
                        metadata = paper_data['metadatas'][0]
                        # Convert distance to similarity score between 0 and 1
                        distance = results['distances'][0][i]
                        max_distance = max(results['distances'][0])  # Get max distance for normalization
                        similarity = 1 - (distance / max_distance)  # Normalize to 0-1 range

                        similar_papers.append({
                            'id': node_id,
                            'title': metadata.get('title', 'Unknown'),
                            'categories': metadata.get('categories', '').split(','),
                            'similarity': similarity
                        })

            return similar_papers[:n]

        except Exception as e:
            logger.error(f"Error querying similar papers: {str(e)}")
            return []

    def find_research_paths(self, start_id: str, end_id: str, max_paths: int = 3):
        """Find multiple research paths between papers"""
        try:
            paths = list(nx.shortest_simple_paths(self.graph, start_id, end_id))[:max_paths]
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def analyze_influence(self, paper_id: str):
        """Analyze paper's influence in the network"""
        if paper_id not in self.graph:
            return None
            
        analysis = {
            'in_degree': self.graph.in_degree(paper_id),
            'out_degree': self.graph.out_degree(paper_id),
            'pagerank': nx.pagerank(self.graph)[paper_id],
            'betweenness': nx.betweenness_centrality(self.graph, k=1000)[paper_id]
        }
        return analysis

def main():
    arxiv_path = Path('/Users/jairadhakrishnan/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/216/arxiv-metadata-oai-snapshot.json')
    
    # Initialize and build network
    network = CitationNetwork(arxiv_path)
    network.build_graph()
    
    # Example usage
    logger.info("\nTesting network analysis functions:")
    
    # Get a sample AI paper ID
    sample_id = "0710.0410"

    # Test influence analysis
    influence = network.analyze_influence(sample_id)
    logger.info(f"\nInfluence analysis for {sample_id}:")
    logger.info(json.dumps(influence, indent=2))
    
    # Test similar papers
    similar = network.get_similar_papers(sample_id)
    logger.info(f"\nSimilar papers to {sample_id}:")
    for paper_id, similarity in similar:
        logger.info(f"{paper_id}: {similarity:.3f}")

if __name__ == "__main__":
    main()