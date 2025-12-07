import os
import networkx as nx
import matplotlib.pyplot as plt

# Output directory
OUTPUT_DIR = "P2-A/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Real-World Network (Karate Club Graph)
G = nx.karate_club_graph()

print("="*70)
print("P2-A: Degree Distribution and Network Properties")
print("="*70)
print(f"Number of Nodes: {G.number_of_nodes()}")
print(f"Number of Edges: {G.number_of_edges()}")


# ============================================================
# DEGREE DISTRIBUTION
# ============================================================
degrees = [G.degree(n) for n in G.nodes()]

plt.figure(figsize=(8, 5))
plt.hist(degrees, bins=10, edgecolor="black")
plt.title("Degree Distribution (Karate Club Network)")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "degree_distribution.png"), dpi=300)
plt.show()

print("\nDegree distribution plot saved!")


# ============================================================
# LOCAL PROPERTIES
# ============================================================

local_clustering = nx.clustering(G)
print("\nLocal Clustering Coefficient (per node):")
for node, coeff in local_clustering.items():
    print(f"Node {node}: {coeff:.4f}")


# ============================================================
# GLOBAL PROPERTIES
# ============================================================

global_clustering = nx.transitivity(G)
degree_cent = nx.degree_centrality(G)
closeness_cent = nx.closeness_centrality(G)
betweenness_cent = nx.betweenness_centrality(G, normalized=True)

print("\nGlobal Properties:")
print(f"Global Clustering Coefficient: {global_clustering:.4f}")
print("\nDegree Centrality:")
print(degree_cent)
print("\nCloseness Centrality:")
print(closeness_cent)
print("\nBetweenness Centrality:")
print(betweenness_cent)

print("\nCompleted Successfully!")
