import streamlit as st
from pyvis.network import Network
import networkx as nx
from pathlib import Path
import tempfile
import json
from citation_network import CitationNetwork
import logging
from typing import List, Dict, Any


# Import AI categories from your existing code
from research_papers.research_chroma import AI_CATEGORIES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkVisualizer:
    def __init__(self, network: CitationNetwork):
        self.citation_network = network
        self.graph = network.graph
    
    def create_full_graph_visualization(self, max_nodes: int = 1000):
        """Create a visualization of the entire graph (limited to max_nodes)."""
        net = Network(
            height='800px',
            width='100%',
            bgcolor='#ffffff',
            font_color='black'
        )
        net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100)
        
        # Get the most connected nodes
        nodes_by_degree = sorted(
            self.graph.nodes(data=True),
            key=lambda x: self.graph.degree(x[0]),
            reverse=True
        )[:max_nodes]
        
        # Create subgraph of most connected nodes
        subgraph = self.graph.subgraph([node[0] for node in nodes_by_degree])
        
        # Add nodes
        for node, data in subgraph.nodes(data=True):
            # Calculate size based on degree (normalized)
            size = 10 + (subgraph.degree(node) * 30 / max(1, max(dict(subgraph.degree()).values())))
            title = data.get('title', node)
            categories = data.get('categories', [])
            
            # Color mapping for different AI categories
            category_colors = {
                'cs.AI': '#e41a1c',    # Red
                'cs.LG': '#377eb8',     # Blue
                'cs.CL': '#4daf4a',     # Green
                'cs.CV': '#984ea3',     # Purple
                'cs.NE': '#ff7f00',     # Orange
                'stat.ML': '#a65628'    # Brown
            }
            
            # If paper belongs to multiple categories, use the first one in our priority list
            paper_category = next((cat for cat in AI_CATEGORIES if cat in categories), 'other')
            color = category_colors.get(paper_category, '#999999')
            
            net.add_node(
                node,
                label=title[:20] + "..." if len(title) > 20 else title,
                title=f"ID: {node}\nTitle: {title}\nCategories: {', '.join(categories)}",
                color=color,
                size=size
            )
        
        # Add edges
        for edge in subgraph.edges():
            net.add_edge(edge[0], edge[1], color='#666666', arrows='to')
        
        return net
    
    def create_subgraph_visualization(
        self, 
        center_node: str, 
        depth: int = 2, 
        max_nodes: int = 100,
        include_similar: bool = True,
        n_similar: int = 5
    ):
        """Create visualization of the subgraph around a paper, including similar papers.
        
        Args:
            center_node: ID of the central paper
            depth: How many steps out to explore in citation network
            max_nodes: Maximum number of nodes to show
            include_similar: Whether to include similar papers
            n_similar: Number of similar papers to include
        """
        # Get similar papers first if requested
        similar_papers = []
        if include_similar:
            similar_papers = self.citation_network.get_similar_papers(center_node, n=n_similar)
            
        # Extract subgraph
        subgraph = self._extract_local_subgraph(center_node, depth, max_nodes, similar_papers)
        
        # Create Pyvis network
        net = Network(
            height='800px',
            width='100%',
            bgcolor='#ffffff',
            font_color='black'
        )
        net.force_atlas_2based(
            gravity=-50,
            central_gravity=0.01,
            spring_length=100,
            spring_strength=0.08,
            damping=0.95
        )
        
        # Create lookup dict for similar papers
        similar_paper_ids = {p['id']: p['similarity'] for p in similar_papers}
        
        # Add nodes with improved visual styling
        self._add_nodes_to_visualization(
            net, 
            subgraph, 
            center_node, 
            similar_paper_ids
        )
        
        # Add edges with improved styling
        self._add_edges_to_visualization(
            net, 
            subgraph, 
            center_node,
            similar_paper_ids
        )
        
        return net
        
    def _add_nodes_to_visualization(self, net, subgraph, center_node, similar_paper_ids):
        """Add nodes to the visualization with appropriate styling."""
        # Color scheme
        colors = {
            'center': '#e41a1c',      # Red
            'similar': '#ff7f00',     # Orange
            'cited': '#377eb8',       # Blue
            'citing': '#4daf4a',      # Green
            'both': '#984ea3',        # Purple
            'indirect': '#a65628'     # Brown
        }
        
        for node, data in subgraph.nodes(data=True):
            # Calculate node size based on degree and importance
            base_size = 20
            
            if node == center_node:
                node_type = 'center'
            elif node in similar_paper_ids:
                node_type = 'similar'
                # Adjust size based on similarity score
            elif subgraph.has_edge(center_node, node) and subgraph.has_edge(node, center_node):
                node_type = 'both'
            elif subgraph.has_edge(center_node, node):
                node_type = 'cited'
            elif subgraph.has_edge(node, center_node):
                node_type = 'citing'
            else:
                node_type = 'indirect'
            
            size = base_size
            
            # Create node label and title
            title = data.get('title', node)
            categories = data.get('categories', [])
            
            # Build detailed hover text
            hover_text = [
                f"ID: {node}",
                f"Title: {title}",
                f"Categories: {', '.join(categories)}"
            ]
            
            if node in similar_paper_ids:
                hover_text.append(f"Similarity Score: {similar_paper_ids[node]:.3f}")
            if node != center_node:
                hover_text.append(f"Connection: {node_type.title()}")
            
            net.add_node(
                node,
                label=title[:30] + "..." if len(title) > 30 else title,
                title="\n".join(hover_text),
                color=colors[node_type],
                size=size,
                borderWidth=3 if node == center_node else 1,
                borderWidthSelected=5,
                font={'size': 14 if node == center_node else 12}
            )
            
    def _add_edges_to_visualization(self, net, subgraph, center_node, similar_paper_ids):
        """Add edges to the visualization with appropriate styling."""
        # Add citation edges
        for source, target in subgraph.edges():
            # Determine edge style
            is_central = (source == center_node or target == center_node)
            is_similar = (source in similar_paper_ids or target in similar_paper_ids)
            
            edge_width = 2 if is_central else 1
            edge_color = '#2c3e50' if is_central else '#95a5a6'
            edge_style = 'solid'
            
            net.add_edge(
                source,
                target,
                color=edge_color,
                width=edge_width,
                style=edge_style,
                arrows={'to': {'enabled': True, 'scaleFactor': 0.5}},
                smooth={'type': 'continuous'}
            )
        
        # Add dashed edges for similar papers that aren't connected in citation network
        for similar_id, similarity in similar_paper_ids.items():
            if not subgraph.has_edge(center_node, similar_id) and \
               not subgraph.has_edge(similar_id, center_node):
                net.add_edge(
                    center_node,
                    similar_id,
                    color='#e67e22',
                    width=similarity * 3,  # Width based on similarity
                    style='dashed',
                    dashes='10,10',
                    arrows={'to': {'enabled': False}},
                    smooth={'type': 'continuous'},
                    title=f'Similarity: {similarity:.3f}'
                )
    
    def _extract_local_subgraph(self, center_node: str, depth: int, max_nodes: int, similar_papers: List[Dict]):
        """Extract a local subgraph around a center node, including similar papers."""
        # Start with center node and similar papers
        nodes = {center_node}
        if similar_papers:
            nodes.update(paper['id'] for paper in similar_papers)
        
        # Expand by citation links
        for _ in range(depth):
            new_nodes = set()
            for node in nodes:
                if node in self.graph:  # Check if node exists in main graph
                    new_nodes.update(self.graph.predecessors(node))
                    new_nodes.update(self.graph.successors(node))
            nodes.update(new_nodes)
            if len(nodes) >= max_nodes:
                break
        
        # If we exceeded max_nodes, prioritize important nodes
        if len(nodes) > max_nodes:
            priority_nodes = {center_node}
            priority_nodes.update(paper['id'] for paper in similar_papers)
            
            # Add high-degree nodes
            remaining = nodes - priority_nodes
            degree_sorted = sorted(
                [(n, self.graph.degree(n)) for n in remaining],
                key=lambda x: x[1],
                reverse=True
            )
            priority_nodes.update(
                n for n, _ in degree_sorted[:max_nodes - len(priority_nodes)]
            )
            nodes = priority_nodes
        
        return self.graph.subgraph(nodes)

def main():
    st.set_page_config(layout="wide", page_title="ArXiv Citation Network Visualizer")
    
    st.title("ArXiv AI Papers Citation Network")
    
    # Initialize citation network
    arxiv_path = Path('/Users/jairadhakrishnan/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/216/arxiv-metadata-oai-snapshot.json')
    
    @st.cache_resource
    def load_network():
        network = CitationNetwork(arxiv_path)
        network.build_graph()
        return network
    
    with st.spinner("Loading citation network..."):
        network = load_network()
        visualizer = NetworkVisualizer(network)
    
    # Sidebar controls
    st.sidebar.header("Visualization Controls")
    
    # View selection
    view_type = st.sidebar.radio(
        "Select View",
        ["Full Network", "Paper Neighborhood"]
    )
    
    if view_type == "Full Network":
        max_nodes = st.sidebar.slider(
            "Number of Papers to Show",
            min_value=100,
            max_value=2500,
            value=500,
            step=100,
            help="Show top N most connected papers"
        )
        
        with st.spinner("Generating full network visualization..."):
            net = visualizer.create_full_graph_visualization(max_nodes=max_nodes)
            
            # Save and display the network
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
                net.save_graph(tmp.name)
                with open(tmp.name, 'r', encoding='utf-8') as f:
                    html = f.read()
                st.components.v1.html(html, height=800)
                
            st.info("""
            Node colors:
            - Red: CS.AI (Artificial Intelligence)
            - Blue: CS.LG (Machine Learning)
            - Green: CS.CL (Computation and Language)
            - Purple: CS.CV (Computer Vision)
            - Orange: CS.NE (Neural and Evolutionary Computing)
            - Brown: STAT.ML (Statistics Machine Learning)
            - Gray: Multiple categories
            
            Node size indicates number of connections.
            """)
    
    else:
        # Paper selection
        paper_id = st.sidebar.text_input(
            "Enter Paper ID",
            value="2103.00020",
            help="Enter an arXiv paper ID (e.g., 2103.00020)"
        )
        
        depth = st.sidebar.slider(
            "Exploration Depth",
            min_value=1,
            max_value=4,
            value=2,
            help="Number of steps to explore from the central paper"
        )
        
        max_nodes = st.sidebar.slider(
            "Maximum Papers",
            min_value=50,
            max_value=500,
            value=200,
            help="Maximum number of papers to show"
        )
        
        if paper_id:
            if paper_id in network.graph:
                with st.spinner("Generating visualization..."):
                    
                    net = visualizer.create_subgraph_visualization(
                        paper_id,
                        depth=depth,
                        max_nodes=max_nodes,
                        include_similar=True,  # New parameter
                        n_similar=5            # New parameter
                    )
                    
                    # Save and display the network
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
                        net.save_graph(tmp.name)
                        with open(tmp.name, 'r', encoding='utf-8') as f:
                            html = f.read()
                        st.components.v1.html(html, height=800)
                    
                    st.info("""
                    Node colors:
                    - Red: Selected paper
                    - Blue: Papers cited by selected paper
                    - Green: Papers citing selected paper
                    - Purple: Other related papers
                    
                    Node size indicates number of connections.
                    """)
                    
                    # Display paper info
                    st.subheader("Selected Paper Information")
                    paper_title = network.graph.nodes[paper_id].get('title', 'Title not available')
                    paper_categories = network.graph.nodes[paper_id].get('categories', [])
                    
                    st.write(f"**Title:** {paper_title}")
                    st.write(f"**Categories:** {', '.join(paper_categories)}")
                    
                    # Display network metrics
                    influence = network.analyze_influence(paper_id)
                    if influence:
                        st.write("### Network Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Citations Received", influence['in_degree'])
                        col2.metric("Papers Cited", influence['out_degree'])
                        col3.metric("PageRank Score", f"{influence['pagerank']:.4f}")
                        col4.metric("Betweenness Centrality", f"{influence['betweenness']:.4f}")
            else:
                st.error("Paper ID not found in the network")

if __name__ == "__main__":
    main()