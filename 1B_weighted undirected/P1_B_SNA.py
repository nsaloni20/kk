import os
import networkx as nx
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# Create output directory for saving the generated plot
# --------------------------------------------------------------
OUTPUT_DIR = "P1-B/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------
# Define a small undirected weighted graph
# Each tuple: (node1, node2, weight)
# --------------------------------------------------------------
edges = [
    ("A", "B", 3),
    ("A", "C", 6),
    ("B", "C", 2),
    ("B", "D", 5),
    ("C", "D", 1),
    ("C", "E", 7),
    ("D", "E", 4),
]

# Build the undirected weighted graph
G = nx.Graph()
for u, v, w in edges:
    G.add_edge(u, v, weight=w)

# --------------------------------------------------------------
# Node size proportional to degree
# --------------------------------------------------------------
degree = {n: G.degree(n) for n in G.nodes()}
node_sizes = [degree[n] * 900 for n in G.nodes()]

# --------------------------------------------------------------
# Edge width proportional to weight
# --------------------------------------------------------------
weights = [G[u][v]["weight"] for u, v in G.edges()]
edge_widths = [w * 1.3 for w in weights]

# --------------------------------------------------------------
# Layout (Spring Layout)
# Higher edge weight → shorter distance in graph layout
# Seed ensures repeatable layout
# --------------------------------------------------------------
pos = nx.spring_layout(G, seed=102, k=0.9)

# --------------------------------------------------------------
# Plot the network
# --------------------------------------------------------------
plt.figure(figsize=(10, 7))

# Draw nodes
nx.draw_networkx_nodes(
    G, pos,
    node_color="#66FF99",
    node_size=node_sizes,
    edgecolors="black",
    linewidths=1.4,
    alpha=0.8
)

# Draw edges
nx.draw_networkx_edges(
    G, pos,
    width=edge_widths,
    edge_color="black"
)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

# Draw edge weights
edge_labels = {(u, v): G[u][v]["weight"] for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")

plt.title(
    "P1-B: Weighted Undirected Network\n"
    "Node Size ∝ Degree | Edge Width ∝ Weight | Edge Length ∝ Weight",
    fontsize=14
)
plt.axis("off")
plt.tight_layout()

# Save the figure
save_path = os.path.join(OUTPUT_DIR, "P1B_undirected_simple.png")
plt.savefig(save_path, dpi=300)
plt.show()

print("Plot saved at:", save_path)
