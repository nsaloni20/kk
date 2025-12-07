import os
import math
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = "P2-A/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

G = nx.karate_club_graph()

print("=" * 70)
print("P2-A: Degree Distribution and Network Properties (Karate Club Graph)")
print("=" * 70)
print(f"Number of Nodes: {G.number_of_nodes()}")
print(f"Number of Edges: {G.number_of_edges()}")

# =======================================================================
# MANUAL FORMULAS
# =======================================================================

# ------------------------------
# 1. Degree Centrality (manual)
# ------------------------------
def manual_degree_centrality(G):
    n = len(G) - 1
    return {node: G.degree(node) / n for node in G.nodes()}

# ------------------------------
# 2. Closeness Centrality (manual)
# ------------------------------
def manual_closeness_centrality(G):
    closeness = {}
    for node in G.nodes():
        sp = nx.shortest_path_length(G, node)
        total_dist = sum(sp.values())
        closeness[node] = (len(G) - 1) / total_dist
    return closeness

# ------------------------------
# 3. Betweenness Centrality (manual)
# ------------------------------
def manual_betweenness_centrality(G):
    bet = dict.fromkeys(G, 0.0)

    for s in G.nodes():
        # Single-source shortest paths
        stack = []
        pred = {w: [] for w in G.nodes()}
        sigma = dict.fromkeys(G, 0.0)  # number of shortest paths
        dist = dict.fromkeys(G, math.inf)  # distances

        sigma[s] = 1.0
        dist[s] = 0
        queue = [s]

        while queue:
            v = queue.pop(0)
            stack.append(v)
            for w in G.neighbors(v):
                # Discovering new node
                if dist[w] == math.inf:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                # Shortest path found
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        # Accumulate dependency
        delta = dict.fromkeys(G, 0.0)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                bet[w] += delta[w]

    # Normalize
    scale = 1 / ((len(G) - 1) * (len(G) - 2))
    for v in bet:
        bet[v] *= scale

    return bet

# ------------------------------
# 4. Local Clustering Coefficient (manual)
# ------------------------------
def manual_local_clustering(G):
    C = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        k = len(neighbors)
        if k < 2:
            C[node] = 0.0
            continue

        # count edges between neighbors
        links = 0
        for i in range(k):
            for j in range(i + 1, k):
                if G.has_edge(neighbors[i], neighbors[j]):
                    links += 1

        C[node] = (2 * links) / (k * (k - 1))
    return C

# ------------------------------
# 5. Global Clustering (manual)
# ------------------------------
def manual_global_clustering(G):
    triangles = sum(nx.triangles(G).values()) / 3
    triples = sum(d * (d - 1) / 2 for d in dict(G.degree()).values())
    return triangles / triples

# =======================================================================
# NETWORKX VALUES
# =======================================================================
nx_deg = nx.degree_centrality(G)
nx_close = nx.closeness_centrality(G)
nx_bet = nx.betweenness_centrality(G, normalized=True)
nx_lcc = nx.clustering(G)
nx_gcc = nx.transitivity(G)

# =======================================================================
# MANUAL VALUES
# =======================================================================
man_deg = manual_degree_centrality(G)
man_close = manual_closeness_centrality(G)
man_bet = manual_betweenness_centrality(G)
man_lcc = manual_local_clustering(G)
man_gcc = manual_global_clustering(G)

# =======================================================================
# Comparison Table
# =======================================================================
df = pd.DataFrame({
    "Node": list(G.nodes()),
    "Manual_Degree": [man_deg[n] for n in G.nodes()],
    "NX_Degree": [nx_deg[n] for n in G.nodes()],
    "Manual_Closeness": [man_close[n] for n in G.nodes()],
    "NX_Closeness": [nx_close[n] for n in G.nodes()],
    "Manual_Betweenness": [man_bet[n] for n in G.nodes()],
    "NX_Betweenness": [nx_bet[n] for n in G.nodes()],
    "Manual_Local_Clust": [man_lcc[n] for n in G.nodes()],
    "NX_Local_Clust": [nx_lcc[n] for n in G.nodes()],
})

df.to_csv(os.path.join(OUTPUT_DIR, "centrality_comparison.csv"), index=False)

print("\n===== COMPARISON TABLE (Manual vs NetworkX) =====\n")
print(df.round(4).to_string(index=False))

print("\nGlobal Clustering Coefficient (Manual) :", round(man_gcc, 4))
print("Global Clustering Coefficient (NX)     :", round(nx_gcc, 4))

# =======================================================================
# Plotting Degree Distribution
# =======================================================================
degrees = [G.degree(n) for n in G.nodes()]

plt.figure(figsize=(10, 6))
plt.hist(degrees, bins=10, color="skyblue", edgecolor="black")
plt.title("Degree Distribution (Karate Club Network)")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "degree_distribution.png"), dpi=300)
plt.show()

# =======================================================================
# VISUALIZATION: Karate Club Network (Informative Layout)
# =======================================================================

plt.figure(figsize=(12, 9))

# Compute a nice layout
pos = nx.spring_layout(G, seed=42, k=0.35)

# Node sizes based on degree
node_sizes_vis = [G.degree(n) * 250 for n in G.nodes()]

# Draw nodes
nx.draw_networkx_nodes(
    G, pos,
    node_color="lightgreen",
    node_size=node_sizes_vis,
    edgecolors="black",
    linewidths=1.2,
    alpha=0.9,
)

# Draw edges
nx.draw_networkx_edges(
    G, pos,
    width=1.2,
    alpha=0.7
)

# Labels
nx.draw_networkx_labels(
    G, pos,
    font_size=10,
    font_weight="bold"
)

plt.title(
    "Karate Club Network Visualization\n"
    "Node Size ∝ Degree | Spring Layout for Structure | 34 Real Members",
    fontsize=15,
    fontweight="bold",
    pad=15
)

plt.axis("off")
plt.tight_layout()

vis_path = os.path.join(OUTPUT_DIR, "karate_club_visual.png")
plt.savefig(vis_path, dpi=300)
plt.show()

print("Visualization saved:", vis_path)


print("\nPlot saved:", os.path.join(OUTPUT_DIR, "degree_distribution.png"))
print("\nCompleted Successfully.")
