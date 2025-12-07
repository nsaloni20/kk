import os
import networkx as nx
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# Create output directory for saving the generated plot
# --------------------------------------------------------------
OUTPUT_DIR = "P1-A/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------
# Define a small directed weighted graph for clear visualization
# Each tuple: (source, target, weight)
# --------------------------------------------------------------
edges = [
    ("A", "B", 1),
    ("A", "C", 4),
    ("B", "C", 7),
    ("C", "D", 2),
    ("D", "A", 6)
]

# Build directed graph with edge weights
G = nx.DiGraph()
for u, v, w in edges:
    G.add_edge(u, v, weight=w)

# --------------------------------------------------------------
# Node size proportional to total degree (in-degree + out-degree)
# --------------------------------------------------------------
degree = {n: G.in_degree(n) + G.out_degree(n) for n in G.nodes()}
node_sizes = [degree[n] * 900 for n in G.nodes()]

# --------------------------------------------------------------
# Edge width proportional to weight
# --------------------------------------------------------------
weights = [G[u][v]["weight"] for u, v in G.edges()]
edge_widths = [w * 1.2 for w in weights]

# --------------------------------------------------------------
# Layout uses spring embedding
# Edge length indirectly affected by weight (higher weight = shorter edge)
# Seed ensures stable, repeatable layout
# --------------------------------------------------------------
pos = nx.spring_layout(G, seed=109, k=1.5)

# --------------------------------------------------------------
# Plot the network
# --------------------------------------------------------------
plt.figure(figsize=(10, 7))

# Draw graph nodes
nx.draw_networkx_nodes(
    G, pos,
    node_color="#FF0000",
    node_size=node_sizes,
    edgecolors="black",
    linewidths=1.5,
    alpha=0.7
)

# Draw directed edges with width scaled by weight
nx.draw_networkx_edges(
    G, pos,
    width=edge_widths,
    edge_color="black",
    arrows=True,
    arrowsize=22,
    arrowstyle="-|>"
)

# Draw node labels
nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

# Draw edge weight labels
edge_labels = {(u, v): G[u][v]["weight"] for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")

plt.title("P1-A: Directed Weighted Network\nNode Size ∝ Degree | Edge Width ∝ Weight | Edge Length ∝ Weight")
plt.axis("off")
plt.tight_layout()

# Save final visualization
save_path = os.path.join(OUTPUT_DIR, "P1A_directed_simple.png")
plt.savefig(save_path, dpi=300)
plt.show()

print("Plot saved at:", save_path)
