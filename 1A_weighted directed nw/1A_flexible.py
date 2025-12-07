import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# User input: path to CSV file (3 columns: source, target, weight)
# Example header: source,target,weight
# --------------------------------------------------------------
CSV_PATH = r"1A_weighted directed nw\data\weighted_edges.csv"

OUTPUT_DIR = "P1-A/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
G = nx.from_pandas_edgelist(
    df,
    source="source",
    target="target",
    edge_attr="weight",
    create_using=nx.DiGraph()
)

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

pos = nx.spring_layout(G, seed=109, k=1.5, weight="weight")

plt.figure(figsize=(10, 7))

nx.draw_networkx_nodes(
    G, pos,
    node_color="#FF0000",
    node_size=node_sizes,
    edgecolors="black",
    linewidths=1.5,
    alpha=0.7
)

nx.draw_networkx_edges(
    G, pos,
    width=edge_widths,
    edge_color="black",
    arrows=True,
    arrowsize=22,
    arrowstyle="-|>"
)

nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

edge_labels = {(u, v): G[u][v]["weight"] for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")

plt.title(
    "P1-A: Directed Weighted Network\n"
    "Node Size prop.to Degree | Edge Width prop.to Weight | Edge Length prop.to Weight"
)
plt.axis("off")
plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, "P1A_directed_from_csv.png")
plt.savefig(save_path, dpi=300)
plt.show()

print("Plot saved at:", save_path)
