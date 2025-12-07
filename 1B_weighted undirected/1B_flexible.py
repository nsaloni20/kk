import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

CSV_PATH = r"1B_weighted undirected\data\edges_undirected.csv"

OUTPUT_DIR = "P1-B/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

G = nx.from_pandas_edgelist(
    df,
    source="source",
    target="target",
    edge_attr="weight",
    create_using=nx.Graph()
)

degree = dict(G.degree())
node_sizes = [degree[n] * 900 for n in G.nodes()]

weights = [G[u][v]["weight"] for u, v in G.edges()]
edge_widths = [w * 1.2 for w in weights]

pos = nx.spring_layout(G, seed=109, k=1.5, weight="weight")

plt.figure(figsize=(10, 7))

nx.draw_networkx_nodes(
    G, pos,
    node_color="#00AAFF",
    node_size=node_sizes,
    edgecolors="black",
    linewidths=1.5,
    alpha=0.7
)

nx.draw_networkx_edges(
    G, pos,
    width=edge_widths,
    edge_color="black"
)

nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

edge_labels = {(u, v): G[u][v]["weight"] for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")

plt.title(
    "P1-B: Weighted Undirected Network\n"
    "Node Size ∝ Degree | Edge Width ∝ Weight | Edge Length ∝ Weight"
)
plt.axis("off")
plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, "P1B_undirected_from_csv.png")
plt.savefig(save_path, dpi=300)
plt.show()

print("Plot saved at:", save_path)
