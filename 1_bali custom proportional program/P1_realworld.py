"""
Script for P1 (Network Visualization) and P2-A (Centrality Analysis) 
using the Bali terrorist network dataset.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# --- 1. Setup and Load Data ---
# IMPORTANT: Ensure Bali.txt is in a folder named 'P1-realworld' in the same directory as this script.
FILE_PATH = "P1-realworld/Bali.txt" 
OUTPUT_DIR = "P1-realworld/output"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the directed graph with weights
G = nx.read_edgelist(
    FILE_PATH,
    create_using=nx.DiGraph(),
    data=(("weight", int),)
)

nodes = list(G.nodes())
N = G.number_of_nodes()
print(f"Network Loaded: Nodes={N}, Edges={G.number_of_edges()}")

# --- P1.1: Unweighted, Undirected Plot (Node Size ∝ Degree) ---

G_undirected = G.to_undirected()
node_degrees = dict(G_undirected.degree())
# Scale node size for better visualization (adjust 150 as needed)
node_sizes = [150 * node_degrees[n] for n in nodes]

plt.figure(figsize=(10, 7))
pos = nx.spring_layout(G_undirected, seed=42)

nx.draw_networkx(
    G_undirected, pos,
    node_size=node_sizes,
    node_color='skyblue',
    edge_color='gray',
    with_labels=True,
    font_size=8
)

plt.title("P1.1: Unweighted Undirected Bali Network (Node Size ∝ Degree)")
plt.axis("off")
p1_plot_path = os.path.join(OUTPUT_DIR, "p1_unweighted_undirected_degree.png")
plt.savefig(p1_plot_path, dpi=300)
plt.close()
print(f"Plot P1.1 saved to: {p1_plot_path}")


# --- P1.2: Weighted Directed Plots (Node Size ∝ In-degree, Edge Width ∝ Weight) ---

G_directed = G
in_degrees = dict(G_directed.in_degree())
edge_weights = [G_directed[u][v]['weight'] for u, v in G_directed.edges()]
max_weight = max(edge_weights) if edge_weights else 1
# Normalize and scale edge width (adjust 3.0 as needed)
edge_widths = [3.0 * w / max_weight for w in edge_weights] 
node_sizes_in_degree = [160 * in_degrees[n] for n in G_directed.nodes()]

layouts = {
    "Spring Layout": nx.spring_layout(G_directed, seed=42),
    "Kamada-Kawai Layout": nx.kamada_kawai_layout(G_directed),
    "Circular Layout": nx.circular_layout(G_directed)
}

for layout_name, pos in layouts.items():
    plt.figure(figsize=(10, 7))

    nx.draw_networkx_nodes(
        G_directed, pos,
        node_size=node_sizes_in_degree,
        node_color="lightgreen",
    )

    nx.draw_networkx_edges(
        G_directed, pos,
        width=edge_widths,
        edge_color="orange",
        arrowsize=12
    )

    nx.draw_networkx_labels(G_directed, pos, font_size=8)

    plt.title(f"P1.2: Weighted Directed Bali Network ({layout_name})\\n(Node size ∝ In-degree, Edge width ∝ Weight)")
    plt.axis("off")
    p1_2_plot_path = os.path.join(OUTPUT_DIR, f"p1_directed_weighted_{layout_name.replace(' ', '_')}.png")
    plt.savefig(p1_2_plot_path, dpi=300)
    plt.close()
    print(f"Plot P1.2 ({layout_name}) saved.")


# --- P2-A: Centrality Measures ---
print("\n--- P2-A: Calculating Centrality Measures ---")

# Undirected Centrality
deg_cent_undir = nx.degree_centrality(G_undirected)
bet_cent_undir = nx.betweenness_centrality(G_undirected, weight='weight', normalized=True)
close_cent_undir = nx.closeness_centrality(G_undirected)

# Directed Centrality (Weighted)
in_deg_cent_dir = nx.in_degree_centrality(G_directed)
out_deg_cent_dir = nx.out_degree_centrality(G_directed)
bet_cent_dir = nx.betweenness_centrality(G_directed, weight='weight', normalized=True)
close_cent_dir = nx.closeness_centrality(G_directed, distance='weight') # Use 'weight' as distance
pr_cent_dir = nx.pagerank(G_directed, weight='weight')


# Create Centrality DataFrame
centrality_df = pd.DataFrame({
    "Node": nodes,
    "Degree_Centrality_Undirected": [deg_cent_undir[n] for n in nodes],
    "Betweenness_Centrality_Undirected": [bet_cent_undir[n] for n in nodes],
    "Closeness_Centrality_Undirected": [close_cent_undir[n] for n in nodes],
    "In_Degree_Centrality_Directed": [in_deg_cent_dir[n] for n in nodes],
    "Out_Degree_Centrality_Directed": [out_deg_cent_dir[n] for n in nodes],
    "Betweenness_Centrality_Directed": [bet_cent_dir[n] for n in nodes],
    "Closeness_Centrality_Directed": [close_cent_dir[n] for n in nodes],
    "PageRank_Directed": [pr_cent_dir[n] for n in nodes]
})

centrality_csv_path = os.path.join(OUTPUT_DIR, "p2a_centrality_results.csv")
centrality_df.to_csv(centrality_csv_path, index=False)

print(f"Centrality results saved to: {centrality_csv_path}")

# Display top nodes for quick check
top_pagerank = centrality_df.sort_values("PageRank_Directed", ascending=False).head(5)
print("\nTop 5 Nodes by PageRank Centrality:")
print(top_pagerank.to_string(index=False, float_format="%.4f"))

top_undirected_degree = centrality_df.sort_values("Degree_Centrality_Undirected", ascending=False).head(5)
print("\nTop 5 Nodes by Undirected Degree Centrality:")
print(top_undirected_degree.to_string(index=False, float_format="%.4f"))