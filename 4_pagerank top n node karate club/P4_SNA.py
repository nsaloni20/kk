"""
Social Network Analysis Practical Assignment
------------------------------------------------
Task:
Compute different centrality measures to identify
top-N nodes and compare their ranks with those
obtained using the PageRank method.
Dataset: Zachary's Karate Club Network (NetworkX)

Features:
1. Load Karate Club network
2. Compute Degree, Betweenness, Closeness, and PageRank centralities
3. Extract Top-N influential nodes
4. Compare rankings against PageRank
5. Visualize the network and all centrality measures
6. Save outputs (plots + CSV files) to P4/output/
"""

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import os

# ----------------------------------------------------------
# 0. Prepare Output Directory
# ----------------------------------------------------------
output_dir = "P4/output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"[INFO] Output directory ready at: {output_dir}\n")

# ----------------------------------------------------------
# 1. Load Karate Club Graph
# ----------------------------------------------------------
print("=" * 60)
print("KARATE CLUB NETWORK - BASIC INFORMATION")
print("=" * 60)

G = nx.karate_club_graph()

print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Network density: {nx.density(G):.4f}")
print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
print(f"Is connected: {nx.is_connected(G)}")
print(f"Number of triangles: {sum(nx.triangles(G).values()) // 3}")
print(f"Average clustering coefficient: {nx.average_clustering(G):.4f}")
print("=" * 60, "\n")

# ----------------------------------------------------------
# 2. Network Visualization
# ----------------------------------------------------------
print("[INFO] Generating Karate Club network visualization...")

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42, k=0.4, iterations=50)

nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue',
                       edgecolors='black', linewidths=1.3)
nx.draw_networkx_edges(G, pos, width=1.2, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

plt.title("Zachary's Karate Club Network", fontsize=16, pad=20)
plt.axis('off')
plt.tight_layout()

network_png = os.path.join(output_dir, "karate_network.png")
plt.savefig(network_png)
plt.close()

print(f"[SAVED] Network visualization → {network_png}\n")

# ----------------------------------------------------------
# 3. Compute Centrality Measures
# ----------------------------------------------------------
print("[INFO] Computing centrality measures...\n")

degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
pagerank = nx.pagerank(G)

print("[INFO] Centrality computation completed.\n")

# ----------------------------------------------------------
# 4. Store Results in DataFrame
# ----------------------------------------------------------
df = pd.DataFrame({
    "Node": list(G.nodes()),
    "Degree": [degree_centrality[n] for n in G.nodes()],
    "Betweenness": [betweenness_centrality[n] for n in G.nodes()],
    "Closeness": [closeness_centrality[n] for n in G.nodes()],
    "PageRank": [pagerank[n] for n in G.nodes()]
})

df_path = os.path.join(output_dir, "centrality_values.csv")
df.to_csv(df_path, index=False)
print(f"[SAVED] Centrality values → {df_path}\n")

# ----------------------------------------------------------
# 5. Extract Top-N Nodes
# ----------------------------------------------------------
TOP_N = 5
print(f"[INFO] Extracting Top-{TOP_N} most influential nodes...\n")

top_degree = df.nlargest(TOP_N, "Degree")
top_betweenness = df.nlargest(TOP_N, "Betweenness")
top_closeness = df.nlargest(TOP_N, "Closeness")
top_pagerank = df.nlargest(TOP_N, "PageRank")

# Ranking comparison table
comparison = pd.DataFrame({
    "Rank #": range(1, TOP_N + 1),
    "Top_PageRank": list(top_pagerank["Node"]),
    "Top_Degree": list(top_degree["Node"]),
    "Top_Betweenness": list(top_betweenness["Node"]),
    "Top_Closeness": list(top_closeness["Node"])
})

comp_path = os.path.join(output_dir, "rank_comparison.csv")
comparison.to_csv(comp_path, index=False)
print(f"[SAVED] Ranking comparison table → {comp_path}\n")

# ----------------------------------------------------------
# 6. Plotting Function
# ----------------------------------------------------------
def plot_centrality(values, title, filename):
    plt.figure(figsize=(10, 5))
    plt.bar(values.keys(), values.values(), color='steelblue')
    plt.xlabel("Node")
    plt.ylabel("Centrality Value")
    plt.title(title)
    plt.tight_layout()

    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    plt.close()

    print(f"[SAVED] {title} plot → {path}")

# ----------------------------------------------------------
# 7. Generate Visualizations
# ----------------------------------------------------------
print("[INFO] Generating centrality plots...\n")

plot_centrality(degree_centrality, "Degree Centrality", "degree_centrality.png")
plot_centrality(betweenness_centrality, "Betweenness Centrality", "betweenness_centrality.png")
plot_centrality(closeness_centrality, "Closeness Centrality", "closeness_centrality.png")
plot_centrality(pagerank, "PageRank Scores", "pagerank.png")

print("\n[INFO] All visualizations saved successfully!\n")

# ----------------------------------------------------------
# 8. Display Top-N Results in Console
# ----------------------------------------------------------
print("=" * 60)
print("TOP-N NODE RESULTS")
print("=" * 60)
print("\nTop PageRank Nodes:\n", top_pagerank)
print("\nTop Degree Centrality Nodes:\n", top_degree)
print("\nTop Betweenness Centrality Nodes:\n", top_betweenness)
print("\nTop Closeness Centrality Nodes:\n", top_closeness)

print("\nRank Comparison Table:\n", comparison)
print("\n[COMPLETED] Script executed successfully.")