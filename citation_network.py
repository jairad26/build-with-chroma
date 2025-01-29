import networkx as nx
from node2vec import Node2Vec
import torch
import json
from pathlib import Path
from tqdm import tqdm
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CitationNetwork:
    def __init__(self, arxiv_path: Path):
        self.arxiv_path = arxiv_path
        self.graph = nx.DiGraph()
        self.node2vec_model = None
        self.embeddings = {}
        
    def extract_citations(self, text: str) -> list:
        """Extract arXiv IDs from text using regex patterns."""
        # Pattern for arXiv citations (both old and new format)
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
    
    def build_graph(self):
        """Build citation network from ArXiv papers."""
        logger.info("Building citation network...")
        
        with open(self.arxiv_path, 'r') as f:
            for line in tqdm(f, desc="Processing papers"):
                paper = json.loads(line)
                
                # Only process if paper has an ID
                if 'id' not in paper:
                    continue
                    
                paper_id = paper['id']
                self.graph.add_node(paper_id, title=paper.get('title', ''))
                
                # Extract citations from abstract and comments
                text_to_search = f"{paper.get('abstract', '')} {paper.get('comments', '')}"
                citations = self.extract_citations(text_to_search)
                
                # Add edges for citations
                for cited_id in citations:
                    self.graph.add_edge(paper_id, cited_id)
        
        logger.info(f"Graph built with {self.graph.number_of_nodes()} nodes and "
                   f"{self.graph.number_of_edges()} edges")
    
    def compute_node2vec(self, dimensions=128, walk_length=30, num_walks=200):
        """Compute node2vec embeddings for the citation network."""
        logger.info("Computing node2vec embeddings...")
        
        # Initialize node2vec model
        node2vec = Node2Vec(
            self.graph,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=4  # Parallel walks
        )
        
        # Train the model
        self.node2vec_model = node2vec.fit(
            window=10,
            min_count=1,
            batch_words=4
        )
        
        # Store embeddings in dictionary
        for node in self.graph.nodes():
            try:
                self.embeddings[node] = self.node2vec_model.wv[node]
            except KeyError:
                logger.warning(f"No embedding found for node {node}")
        
        logger.info("Node2vec embeddings computed successfully")
    
    def find_research_path(self, start_id: str, end_id: str, max_length: int = 5):
        """Find shortest research path between two papers."""
        try:
            path = nx.shortest_path(self.graph, start_id, end_id)
            if len(path) > max_length:
                return None
            return path
        except nx.NetworkXNoPath:
            return None
    
    def get_similar_papers(self, paper_id: str, n: int = 5):
        """Find similar papers based on node2vec embeddings."""
        if not self.node2vec_model or paper_id not in self.embeddings:
            return []
        
        similar_nodes = self.node2vec_model.wv.most_similar(paper_id, topn=n)
        return [(node, score) for node, score in similar_nodes]
    
    def analyze_influence(self, paper_id: str):
        """Analyze paper's influence in the network."""
        if paper_id not in self.graph:
            return None
            
        analysis = {
            'in_degree': self.graph.in_degree(paper_id),  # Citations received
            'out_degree': self.graph.out_degree(paper_id),  # Papers cited
            'pagerank': nx.pagerank(self.graph)[paper_id],  # PageRank score
            'betweenness': nx.betweenness_centrality(self.graph, k=1000)[paper_id]  # Betweenness centrality
        }
        return analysis

def main():
    arxiv_path = Path('/Users/jairadhakrishnan/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/216/arxiv-metadata-oai-snapshot.json')
    
    # Initialize and build network
    network = CitationNetwork(arxiv_path)
    network.build_graph()
    
    # Compute node2vec embeddings
    network.compute_node2vec()
    
    # Example usage
    test_paper_id = list(network.graph.nodes())[0]  # Get first paper ID
    
    logger.info("\nTesting network analysis functions:")
    
    # Test influence analysis
    influence = network.analyze_influence(test_paper_id)
    logger.info(f"\nInfluence analysis for {test_paper_id}:")
    logger.info(json.dumps(influence, indent=2))
    
    # Test similar papers
    similar = network.get_similar_papers(test_paper_id)
    logger.info(f"\nSimilar papers to {test_paper_id}:")
    for paper_id, similarity in similar:
        logger.info(f"{paper_id}: {similarity:.3f}")

if __name__ == "__main__":
    main()