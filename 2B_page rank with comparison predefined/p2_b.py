"""
P2-B: PageRank Implementation and Validation
--------------------------------------------

This program performs:
1. Manual PageRank calculation (iterative)
2. Calculates iterations to convergence
3. Validates with NetworkX PageRank
4. Saves convergence plot & comparison CSV
5. Prints top-ranked nodes
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Output directory
OUTPUT_DIR = "P2-B/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Karate Club Graph
G = nx.karate_club_graph()
nodes = list(G.nodes())
N = len(nodes)
node_index = {node: idx for idx, node in enumerate(nodes)}

# PageRank parameters
damping_factor = 0.85         # d
alpha = 1 - damping_factor    # random jump probability
tolerance = 1e-6
max_iter = 1000

# Build row-stochastic adjacency matrix
A = np.zeros((N, N), dtype=float)

for u in nodes:
    i = node_index[u]
    neighbors = list(G.neighbors(u))
    outdeg = len(neighbors)

    if outdeg == 0:
        # Dangling node: distribute uniformly
        A[i, :] = 1.0 / N
    else:
        prob = 1.0 / outdeg
        for v in neighbors:
            j = node_index[v]
            A[i, j] = prob

# Uniform random-jump vector
E = np.ones(N, dtype=float) / N

# Initialize PageRank vector
R_old = np.ones(N, dtype=float) / N
history = []
converged = False

# Iterative PageRank
for iteration in range(1, max_iter + 1):
    R_new = (1 - alpha) * (R_old @ A) + alpha * E
    diff = np.abs(R_new - R_old)
    max_diff = diff.max()
    history.append(max_diff)

    if max_diff <= tolerance:
        converged = True
        final_iter = iteration
        break

    R_old = R_new

if not converged:
    final_iter = max_iter

# Normalize final PageRank
R_manual = R_new / R_new.sum()

# NetworkX PageRank for validation
R_nx_dict = nx.pagerank(G, alpha=damping_factor)
R_nx = np.array([R_nx_dict[n] for n in nodes], dtype=float)
R_nx = R_nx / R_nx.sum()

# Comparison table
df = pd.DataFrame({
    "Node": nodes,
    "Manual_PageRank": R_manual,
    "NetworkX_PageRank": R_nx,
    "Absolute_Difference": np.abs(R_manual - R_nx)
})

csv_path = os.path.join(OUTPUT_DIR, "pagerank_comparison.csv")
df.to_csv(csv_path, index=False)

# Convergence plot
plt.figure(figsize=(8, 5))
plt.plot(history, marker="o")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Max |ΔPageRank|")
plt.title("PageRank Convergence (Manual Implementation)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
convergence_plot = os.path.join(OUTPUT_DIR, "pagerank_convergence.png")
plt.tight_layout()
plt.savefig(convergence_plot, dpi=300)
plt.show()

# Print results
print("="*70)
print("P2-B: PageRank Implementation & Validation")
print("="*70)
print(f"Network: Karate Club (Nodes = {N})")
print(f"Damping Factor: {damping_factor}, Random Jump Prob: {alpha}")
print(f"Convergence Tolerance: {tolerance}")
print()

if converged:
    print(f"Manual PageRank converged in {final_iter} iterations.")
else:
    print(f"Did NOT converge within {max_iter} iterations.")

print("\nTop Nodes by Manual PageRank:")
print(df.sort_values("Manual_PageRank", ascending=False).head(10).to_string(index=False, float_format="%.6f"))
print("\nTop Nodes by NetworkX PageRank:")
print(df.sort_values("NetworkX_PageRank", ascending=False).head(10).to_string(index=False, float_format="%.6f"))

print(f"\nComparison CSV saved at: {csv_path}")
print(f"Convergence plot saved at: {convergence_plot}")
print("="*70)
