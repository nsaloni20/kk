import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import random

OUTPUT_DIR = "P2-realworld/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_graph(path, directed=False):
    edges = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue

            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue

            u, v = parts[0], parts[1]
            edges.append((u, v))

    G = nx.DiGraph() if directed else nx.Graph()
    G.add_edges_from(edges)
    G.remove_nodes_from(list(nx.isolates(G)))
    return G


# ===============================================================
# DEGREE & CUMULATIVE DISTRIBUTION
# ===============================================================
def plot_degree_distributions(G, name):
    """Plot P(k) and C(k)."""
    degrees = [deg for _, deg in G.degree()]
    degree_count = Counter(degrees)

    max_k = max(degrees)
    k_vals = list(range(1, max_k + 1))

    p_k = [degree_count.get(k, 0) / len(degrees) for k in k_vals]
    cum_pk = np.cumsum(p_k[::-1])[::-1]

    plt.figure(figsize=(12, 5))

    # P(k)
    plt.subplot(1, 2, 1)
    plt.scatter(k_vals, p_k, s=12)
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"P(k) – Degree Distribution: {name}")
    plt.xlabel("Degree (k)")
    plt.ylabel("P(k)")

    # C(k)
    plt.subplot(1, 2, 2)
    plt.scatter(k_vals, cum_pk, s=12, color="red")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"C(k) – Cumulative Distribution: {name}")
    plt.xlabel("Degree (k)")
    plt.ylabel("C(k)")

    out = os.path.join(OUTPUT_DIR, f"{name}_degree_plots.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"✓ Saved degree plots → {out}")


# ===============================================================
# METRICS (FULL FOR SMALL, LIMITED FOR LARGE)
# ===============================================================
def compute_graph_metrics(G, name):
    """Compute required SNA metrics with scaling-aware computation."""
    N = G.number_of_nodes()
    E = G.number_of_edges()

    metrics = {
        "Network": name,
        "Nodes": N,
        "Edges": E,
        "Density": round(nx.density(G), 5),
        "Avg Clustering Coeff": round(nx.average_clustering(G), 4),
    }

    # ===========================================================
    # SMALL–MEDIUM NETWORKS (Karate, Books, EuroRoad, Hamsterster)
    # ===========================================================
    if N <= 5000:
        print("   Computing centralities and path lengths (FULL)...")

        deg_cent = nx.degree_centrality(G)
        clo_cent = nx.closeness_centrality(G)
        bet_cent = nx.betweenness_centrality(G)

        metrics["Avg Degree Centrality"] = round(np.mean(list(deg_cent.values())), 5)
        metrics["Avg Closeness Centrality"] = round(np.mean(list(clo_cent.values())), 5)
        metrics["Avg Betweenness Centrality"] = round(np.mean(list(bet_cent.values())), 5)

        # Path metrics (use LCC if needed)
        if nx.is_connected(G):
            metrics["ASPL (Exact)"] = round(nx.average_shortest_path_length(G), 4)
            metrics["Diameter (Exact)"] = nx.diameter(G)
        else:
            LCC = G.subgraph(max(nx.connected_components(G), key=len))
            metrics["ASPL (Exact)"] = round(nx.average_shortest_path_length(LCC), 4)
            metrics["Diameter (Exact)"] = nx.diameter(LCC)

        return metrics

    # ===========================================================
    # VERY LARGE NETWORK (Amazon)
    # ===========================================================
    print("   Large network detected → using sampled metrics + ETA...")

    # Degree centrality (fast)
    deg_cent = nx.degree_centrality(G)
    metrics["Avg Degree Centrality"] = round(np.mean(list(deg_cent.values())), 5)

    # ------- SAMPLE-BASED CLOSENESS (1000 nodes) -------
    sample_size = 1000
    nodes = list(G.nodes())
    sample_nodes = random.sample(nodes, sample_size)

    print(f"   Computing sampled closeness for {sample_size} nodes (ETA visible)...")

    closeness_vals = []
    for node in tqdm(sample_nodes, desc="   Closeness ETA", ncols=100):
        try:
            closeness_vals.append(nx.closeness_centrality(G, u=node))
        except:
            pass

    metrics["Avg Closeness Centrality"] = round(float(np.mean(closeness_vals)), 5)

    # Too large:
    metrics["Avg Betweenness Centrality"] = "SKIPPED (too large)"
    metrics["ASPL (Exact)"] = "SKIPPED (too large)"
    metrics["Diameter (Exact)"] = "SKIPPED (too large)"

    return metrics


# ===============================================================
# MAIN EXECUTION
# ===============================================================
datasets = {
    "PoliticalBooks (~100)": "P2-realworld/dataset/PoliticalBooks.txt",
    "EuroRoad (~1k)": "P2-realworld/dataset/road-euroroad.edges",
    "Hamsterster (~2.4k → ~10k proxy)": "P2-realworld/dataset/soc-hamsterster.edges",
    "Amazon (~335k)": "P2-realworld/dataset/com-amazon.ungraph.txt"
}

results = []

print("\n================= P2 REAL-WORLD NETWORK ANALYSIS =================")

for name, path in datasets.items():
    print(f"\nLoading {name} ...")
    G = load_graph(path)

    print(f" - Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    plot_degree_distributions(G, name)

    print(" - Computing metrics...")
    metrics = compute_graph_metrics(G, name)
    results.append(metrics)


# ===============================================================
# SAVE SUMMARY CSV
# ===============================================================
df = pd.DataFrame(results)
out_csv = os.path.join(OUTPUT_DIR, "network_summary.csv")
df.to_csv(out_csv, index=False)

print("\n===================== SUMMARY SAVED =====================")
print(f"CSV → {out_csv}")
print(df.to_string(index=False))
print("========================================================\n")
