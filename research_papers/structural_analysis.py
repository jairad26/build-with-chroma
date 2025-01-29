import streamlit as st
from citation_network import CitationNetwork
from pathlib import Path
import networkx as nx
import logging
from typing import List, Dict, Any
import pandas as pd
from network_viz import NetworkVisualizer
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StructuralAnalyzer:
    def __init__(self, network: CitationNetwork):
        self.network = network
        self.graph = network.graph
    
    def find_structural_analogs(self, paper_id: str, n: int = 5) -> List[Dict]:
        """Find papers with similar citation patterns using multiple structural metrics.
        
        This improved version considers:
        1. Citation pattern similarity (common neighbors)
        2. Role similarity (in/out degree ratio)
        3. Category overlap
        4. Node2vec embedding similarity
        
        Args:
            paper_id: ID of the target paper
            n: Number of similar papers to return
            
        Returns:
            List of dictionaries containing similar papers and their metrics
        """
        if paper_id not in self.graph:
            return []

        # Get node2vec similarities first
        node2vec_papers = self.network.get_similar_papers(paper_id, n=n*2)  # Get more candidates
        candidates = {p['id']: p['similarity'] for p in node2vec_papers}
        
        # Get target paper properties
        target_in_deg = self.graph.in_degree(paper_id)
        target_out_deg = self.graph.out_degree(paper_id)
        target_ratio = target_in_deg / max(1, target_out_deg)
        target_categories = set(self.graph.nodes[paper_id].get('categories', []))
        
        # Get target paper's neighbors
        target_predecessors = set(self.graph.predecessors(paper_id))
        target_successors = set(self.graph.successors(paper_id))
        
        # Calculate structural similarities for candidates
        structural_scores = {}
        for candidate_id in candidates:
            if candidate_id == paper_id:
                continue
                
            # 1. Citation overlap (Jaccard similarity of neighborhoods)
            candidate_predecessors = set(self.graph.predecessors(candidate_id))
            candidate_successors = set(self.graph.successors(candidate_id))
            
            citation_sim = (
                len(target_predecessors & candidate_predecessors) / 
                max(1, len(target_predecessors | candidate_predecessors))
            )
            reference_sim = (
                len(target_successors & candidate_successors) / 
                max(1, len(target_successors | candidate_successors))
            )
            
            # 2. Role similarity (compare in/out degree ratios)
            candidate_in_deg = self.graph.in_degree(candidate_id)
            candidate_out_deg = self.graph.out_degree(candidate_id)
            candidate_ratio = candidate_in_deg / max(1, candidate_out_deg)
            role_sim = 1 / (1 + abs(target_ratio - candidate_ratio))
            
            # 3. Category similarity
            candidate_categories = set(self.graph.nodes[candidate_id].get('categories', []))
            category_sim = len(target_categories & candidate_categories) / max(1, len(target_categories | candidate_categories))
            
            # 4. Combine with node2vec similarity
            n2v_sim = candidates[candidate_id]
            
            # Calculate weighted combination
            final_score = (
                0.3 * citation_sim +
                0.3 * reference_sim +
                0.2 * role_sim +
                0.1 * category_sim +
                0.1 * n2v_sim
            )
            
            structural_scores[candidate_id] = final_score
        
        # Get top N papers by combined score
        top_papers = sorted(structural_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        
        results = []
        for paper_id, score in top_papers:
            node_data = self.graph.nodes[paper_id]
            results.append({
                'id': paper_id,
                'title': node_data.get('title', 'Unknown'),
                'categories': node_data.get('categories', []),
                'similarity': score,
                'in_degree': self.graph.in_degree(paper_id),
                'out_degree': self.graph.out_degree(paper_id),
                'citation_pattern_similarity': {
                    'shared_citations': len(target_predecessors & set(self.graph.predecessors(paper_id))),
                    'shared_references': len(target_successors & set(self.graph.successors(paper_id)))
                }
            })
        
        return results
    
    def get_role_metrics(self, paper_id: str) -> Dict:
        """Analyze the structural role of a paper in the network."""
        if paper_id not in self.graph:
            return None
            
        # Basic metrics
        in_deg = self.graph.in_degree(paper_id)
        out_deg = self.graph.out_degree(paper_id)
        
        # Get local clustering coefficient
        try:
            clustering = nx.clustering(self.graph.to_undirected(), paper_id)
        except:
            clustering = 0
            
        # Compute betweenness centrality for local neighborhood
        local_graph = nx.ego_graph(self.graph, paper_id, radius=2)
        betweenness = nx.betweenness_centrality(local_graph)[paper_id]
        
        return {
            'citations_received': in_deg,
            'references_made': out_deg,
            'citation_ratio': in_deg / max(1, out_deg),
            'local_clustering': clustering,
            'bridging_score': betweenness,
            'is_survey_like': out_deg > 2 * in_deg if out_deg > 10 else False,
            'is_foundational': in_deg > 3 * out_deg if in_deg > 20 else False
        }
    
    def find_research_paths(self, start_id: str, end_id: str, max_paths: int = 3) -> List[List[str]]:
        """Find multiple research paths between papers."""
        try:
            paths = list(nx.shortest_simple_paths(self.graph, start_id, end_id))[:max_paths]
            return paths
        except nx.NetworkXNoPath:
            return []
        except nx.NodeNotFound:
            return []
    
    def analyze_field_bridging(self, paper_id: str) -> Dict:
        """Analyze how a paper bridges different research areas."""
        if paper_id not in self.graph:
            return None
            
        # Get categories of cited papers
        cited_categories = []
        for _, cited in self.graph.out_edges(paper_id):
            cited_categories.extend(self.graph.nodes[cited].get('categories', []))
            
        # Get categories of citing papers
        citing_categories = []
        for citing, _ in self.graph.in_edges(paper_id):
            citing_categories.extend(self.graph.nodes[citing].get('categories', []))
            
        # Count unique categories
        all_categories = set(cited_categories + citing_categories)
        
        return {
            'num_bridged_categories': len(all_categories),
            'cited_categories': pd.Series(cited_categories).value_counts().to_dict(),
            'citing_categories': pd.Series(citing_categories).value_counts().to_dict()
        }

def main():
    st.set_page_config(layout="wide", page_title="Citation Network Structural Analysis")
    st.title("Citation Network Structural Analysis")
    
    @st.cache_resource
    def init_analyzer():
        arxiv_path = Path('/Users/jairadhakrishnan/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/216/arxiv-metadata-oai-snapshot.json')
        network = CitationNetwork(arxiv_path)
        network.build_graph()
        
        return StructuralAnalyzer(network)
    
    analyzer = init_analyzer()
    
    # Sidebar for analysis type selection
    analysis_type = st.sidebar.radio(
        "Analysis Type",
        ["Structural Analogs", "Paper Role Analysis", "Research Paths", "Field Bridging"]
    )
    
    # Main paper input
    paper_id = st.text_input("Enter Paper ID", "2103.00020")
    
    if paper_id in analyzer.graph:
        paper_title = analyzer.graph.nodes[paper_id].get('title', 'Unknown')
        st.write(f"**Selected Paper:** {paper_title}")
        
        if analysis_type == "Structural Analogs":
            n_results = st.slider("Number of similar papers", 5, 20, 10)
            
            with st.spinner("Finding structural analogs..."):
                analogs = analyzer.find_structural_analogs(paper_id, n_results)
                
                st.write("### Papers with Similar Citation Patterns")
                
                # Create visualization with similar papers highlighted
                visualizer = NetworkVisualizer(analyzer.network)
                net = visualizer.create_subgraph_visualization(
                    paper_id,
                    depth=2,
                    max_nodes=100,
                    similar_papers=analogs
                )
                
                # Save and display the network
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
                    net.save_graph(tmp.name)
                    with open(tmp.name, 'r', encoding='utf-8') as f:
                        html = f.read()
                    st.components.v1.html(html, height=800)
                
                # Display similarity details below the graph
                for i, paper in enumerate(analogs, 1):
                    with st.expander(f"{i}. {paper['title']} (Similarity: {paper['similarity']:.3f})"):
                        st.write(f"**ID:** {paper['id']}")
                        st.write(f"**Categories:** {', '.join(paper['categories'])}")
                        st.write(f"**Citations:** {paper['in_degree']}")
                        st.write(f"**References:** {paper['out_degree']}")
        
        elif analysis_type == "Paper Role Analysis":
            with st.spinner("Analyzing paper role..."):
                role = analyzer.get_role_metrics(paper_id)
                
                if role:
                    st.write("### Paper Role Metrics")
                    col1, col2, col3 = st.columns(3)
                    
                    col1.metric("Citations Received", role['citations_received'])
                    col1.metric("References Made", role['references_made'])
                    
                    col2.metric("Citation Ratio", f"{role['citation_ratio']:.2f}")
                    col2.metric("Clustering Coefficient", f"{role['local_clustering']:.3f}")
                    
                    col3.metric("Bridging Score", f"{role['bridging_score']:.3f}")
                    
                    if role['is_survey_like']:
                        st.info("📚 This paper shows characteristics of a survey or review paper")
                    if role['is_foundational']:
                        st.info("🌟 This paper shows characteristics of a foundational paper")
        
        elif analysis_type == "Research Paths":
            end_paper = st.text_input("Enter Target Paper ID")
            if end_paper:
                with st.spinner("Finding research paths..."):
                    paths = analyzer.find_research_paths(paper_id, end_paper)
                    
                    if paths:
                        st.write("### Research Paths Found")
                        for i, path in enumerate(paths, 1):
                            st.write(f"\nPath {i} (Length: {len(path)}):")
                            for p in path:
                                title = analyzer.graph.nodes[p].get('title', p)
                                st.write(f"→ {title}")
                    else:
                        st.warning("No research paths found between these papers")
        
        elif analysis_type == "Field Bridging":
            with st.spinner("Analyzing field bridging..."):
                bridging = analyzer.analyze_field_bridging(paper_id)
                
                if bridging:
                    st.write("### Field Bridging Analysis")
                    st.write(f"This paper bridges {bridging['num_bridged_categories']} different categories")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("#### Categories of Cited Papers")
                        for cat, count in bridging['cited_categories'].items():
                            st.write(f"- {cat}: {count}")
                    
                    with col2:
                        st.write("#### Categories of Citing Papers")
                        for cat, count in bridging['citing_categories'].items():
                            st.write(f"- {cat}: {count}")
    else:
        st.error("Paper ID not found in the network")

if __name__ == "__main__":
    main()