"""
P2-B: PageRank Implementation and Validation
--------------------------------------------

This program performs the following tasks:

1. Manually implements the PageRank algorithm using an iterative matrix-based formulation.
2. Calculates the number of iterations required for the PageRank vector to converge.
3. Validates the manually computed PageRank values by comparing them with NetworkX's built-in PageRank function.
4. Saves:
    - A convergence plot
    - A comparison CSV file
5. Prints top-ranked nodes and diagnostic information.

"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------
OUTPUT_DIR = "P2-B/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Load Karate Club Network
# ---------------------------------------------------------------------
G = nx.karate_club_graph()
nodes = list(G.nodes())
N = len(nodes)
node_index = {node: idx for idx, node in enumerate(nodes)}

# ---------------------------------------------------------------------
# Parameters for PageRank
# ---------------------------------------------------------------------
damping_factor = 0.85        # Probability of following outgoing links
alpha = 1 - damping_factor   # Probability of making a random jump
tolerance = 1e-6             # Convergence threshold
max_iter = 1000              # Fail-safe iteration cap

# ---------------------------------------------------------------------
# Build Row-Stochastic Adjacency Matrix A
# Each row represents the distribution of probability from a node.
# A[i][j] = 1 / outdegree(i) if edge i → j exists, else 0
# ---------------------------------------------------------------------
A = np.zeros((N, N), dtype=float)

for u in nodes:
    i = node_index[u]
    neighbors = list(G.neighbors(u))
    outdeg = len(neighbors)

    if outdeg == 0:
        # Dangling node (no outgoing links)
        continue

    prob = 1.0 / outdeg
    for v in neighbors:
        j = node_index[v]
        A[i][j] = prob

# ---------------------------------------------------------------------
# Uniform random-jump vector
# ---------------------------------------------------------------------
E = np.ones(N, dtype=float) / N

# ---------------------------------------------------------------------
# Initialize PageRank vector (uniform distribution)
# ---------------------------------------------------------------------
R_old = np.ones(N, dtype=float) / N

# For storing convergence behavior
history = []
converged = False

# ---------------------------------------------------------------------
# Manual PageRank Iteration
# R_new = (1 - alpha) * R_old * A + alpha * E
# Convergence when all components change less than tolerance
# ---------------------------------------------------------------------
for iteration in range(1, max_iter + 1):

    R_new = (1 - alpha) * (R_old @ A) + alpha * E

    diff = np.abs(R_new - R_old)
    max_diff = diff.max()
    history.append(max_diff)

    # Element-wise convergence check
    if np.all(diff <= tolerance):
        converged = True
        final_iter = iteration
        break

    R_old = R_new

if not converged:
    final_iter = max_iter

# Final normalized PageRank vector
R_manual = R_new / R_new.sum()

# ---------------------------------------------------------------------
# NetworkX PageRank for Validation
# ---------------------------------------------------------------------
R_nx_dict = nx.pagerank(G, alpha=damping_factor)
R_nx = np.array([R_nx_dict[n] for n in nodes], dtype=float)
R_nx = R_nx / R_nx.sum()

# ---------------------------------------------------------------------
# Build comparison table
# ---------------------------------------------------------------------
df = pd.DataFrame({
    "Node": nodes,
    "Manual_PageRank": R_manual,
    "NetworkX_PageRank": R_nx,
    "Absolute_Difference": np.abs(R_manual - R_nx)
})

csv_path = os.path.join(OUTPUT_DIR, "pagerank_comparison.csv")
df.to_csv(csv_path, index=False)

# ---------------------------------------------------------------------
# Save convergence plot
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(history, marker="o")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Max |ΔPageRank|")
plt.title("PageRank Convergence Behavior (Manual Implementation)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

convergence_plot = os.path.join(OUTPUT_DIR, "pagerank_convergence.png")
plt.tight_layout()
plt.savefig(convergence_plot, dpi=300)
plt.show()

# ---------------------------------------------------------------------
# Print final results
# ---------------------------------------------------------------------
print("=" * 70)
print("P2-B: PageRank Implementation & Validation")
print("=" * 70)
print(f"Network: Karate Club (Nodes = {N})")
print(f"Damping Factor (d): {damping_factor}")
print(f"Random Jump Probability (alpha): {alpha}")
print(f"Convergence Tolerance: {tolerance}")
print()

if converged:
    print(f"Manual PageRank converged in {final_iter} iterations.")
else:
    print(f"Did NOT converge within {max_iter} iterations.")

print()
print("Top Nodes by Manual PageRank:")
print(df.sort_values("Manual_PageRank", ascending=False).head(10).to_string(index=False, float_format="%.6f"))
print()
print("Top Nodes by NetworkX PageRank:")
print(df.sort_values("NetworkX_PageRank", ascending=False).head(10).to_string(index=False, float_format="%.6f"))

print()
print(f"Comparison CSV saved at: {csv_path}")
print(f"Convergence plot saved at: {convergence_plot}")
print("=" * 70)