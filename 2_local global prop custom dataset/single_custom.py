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

# ====== CHOOSE YOUR SINGLE DATASET HERE ======
NETWORK_NAME = "PoliticalBooks (~100)"
DATA_PATH = "2_local global prop custom dataset\dataset\PoliticalBooks.txt"   # change to any one file
# ============================================

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

def plot_degree_distributions(G, name):
    degrees = [deg for _, deg in G.degree()]
    degree_count = Counter(degrees)

    max_k = max(degrees)
    k_vals = list(range(1, max_k + 1))

    p_k = [degree_count.get(k, 0) / len(degrees) for k in k_vals]
    cum_pk = np.cumsum(p_k[::-1])[::-1]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(k_vals, p_k, s=12)
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"P(k) – Degree Distribution: {name}")
    plt.xlabel("Degree (k)")
    plt.ylabel("P(k)")

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

def compute_graph_metrics(G, name):
    N = G.number_of_nodes()
    E = G.number_of_edges()

    metrics = {
        "Network": name,
        "Nodes": N,
        "Edges": E,
        "Density": round(nx.density(G), 5),
        "Avg Clustering Coeff": round(nx.average_clustering(G), 4),
    }

    if N <= 5000:
        print("   Computing centralities and path lengths (FULL)...")

        deg_cent = nx.degree_centrality(G)
        clo_cent = nx.closeness_centrality(G)
        bet_cent = nx.betweenness_centrality(G)

        metrics["Avg Degree Centrality"] = round(np.mean(list(deg_cent.values())), 5)
        metrics["Avg Closeness Centrality"] = round(np.mean(list(clo_cent.values())), 5)
        metrics["Avg Betweenness Centrality"] = round(np.mean(list(bet_cent.values())), 5)

        if nx.is_connected(G):
            metrics["ASPL (Exact)"] = round(nx.average_shortest_path_length(G), 4)
            metrics["Diameter (Exact)"] = nx.diameter(G)
        else:
            LCC = G.subgraph(max(nx.connected_components(G), key=len))
            metrics["ASPL (Exact)"] = round(nx.average_shortest_path_length(LCC), 4)
            metrics["Diameter (Exact)"] = nx.diameter(LCC)

        return metrics

    print("   Large network detected → using sampled metrics + ETA...")

    deg_cent = nx.degree_centrality(G)
    metrics["Avg Degree Centrality"] = round(np.mean(list(deg_cent.values())), 5)

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
    metrics["Avg Betweenness Centrality"] = "SKIPPED (too large)"
    metrics["ASPL (Exact)"] = "SKIPPED (too large)"
    metrics["Diameter (Exact)"] = "SKIPPED (too large)"

    return metrics

print("\n================= P2 REAL-WORLD NETWORK ANALYSIS =================")

print(f"\nLoading {NETWORK_NAME} ...")
G = load_graph(DATA_PATH)
print(f" - Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

plot_degree_distributions(G, NETWORK_NAME)

print(" - Computing metrics...")
metrics = compute_graph_metrics(G, NETWORK_NAME)

df = pd.DataFrame([metrics])
out_csv = os.path.join(OUTPUT_DIR, "network_summary_single.csv")
df.to_csv(out_csv, index=False)

print("\n===================== SUMMARY SAVED =====================")
print(f"CSV → {out_csv}")
print(df.to_string(index=False))
print("========================================================\n")
