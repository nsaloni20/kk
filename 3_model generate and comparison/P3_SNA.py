"""
P3 - Social Network Analysis Practical Assignment

Task:
Generate networks of 1000 nodes each using:
  1. Erdős-Rényi Random Network Model
  2. Watts-Strogatz (Small World) Network Model
  3. Barabási-Albert (Preferential Attachment) Network Model

Compute and compare their characteristics.
Store all visualizations inside: ./P3/output/

"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# =============================================================================
# Output Directory Setup
# =============================================================================
OUTPUT_DIR = "P3/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Function: Compute Network Characteristics
# =============================================================================
def compute_characteristics(G):
    """
    Computes standard graph characteristics.

    Parameters:
        G (networkx.Graph): The input graph.

    Returns:
        dict: Dictionary containing graph characteristics.
    """
    characteristics = {
        "Nodes": G.number_of_nodes(),
        "Edges": G.number_of_edges(),
        "Density": round(nx.density(G), 4),
        "Average Degree": round(sum(dict(G.degree()).values()) / G.number_of_nodes(), 2),
        "Clustering Coefficient": round(nx.average_clustering(G), 4),
        "Is Connected": nx.is_connected(G),
    }

    if characteristics["Is Connected"]:
        characteristics["Average Path Length"] = round(nx.average_shortest_path_length(G), 4)
    else:
        characteristics["Average Path Length"] = "Not Connected"

    return characteristics


# =============================================================================
# Function: Pretty Print Model Characteristics
# =============================================================================
def print_characteristics(model_name, char_dict):
    """
    Pretty prints characteristics for a given model.
    """
    print(f"\n--- {model_name} ---")
    for key, value in char_dict.items():
        print(f"{key}: {value}")


# =============================================================================
# Function: Visualize Graph and Save Figure
# =============================================================================
def visualize_graph(G, title, filename):
    """
    Visualizes the graph using spring layout and saves to output folder.
    """
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, node_size=25, node_color="skyblue", alpha=0.85)
    nx.draw_networkx_edges(G, pos, width=0.3, alpha=0.4)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":

    print("=" * 75)
    print("SOCIAL NETWORK ANALYSIS PRACTICAL - NETWORK GENERATION & ANALYSIS")
    print("=" * 75)

    N = 1000  # total nodes

    # -------------------------------------------------------------------------
    # Generate Graph Models
    # -------------------------------------------------------------------------
    print("\nGenerating 1000-node network models...")

    G_er = nx.erdos_renyi_graph(N, p=0.01, seed=42)
    G_ws = nx.watts_strogatz_graph(N, k=6, p=0.04, seed=42)
    G_ba = nx.barabasi_albert_graph(N, m=3, seed=42)

    print("✓ Erdős-Rényi Graph Created")
    print("✓ Watts-Strogatz Graph Created")
    print("✓ Barabási-Albert Graph Created")

    # -------------------------------------------------------------------------
    # Compute Characteristics
    # -------------------------------------------------------------------------
    print("\nComputing characteristics...")

    char_er = compute_characteristics(G_er)
    char_ws = compute_characteristics(G_ws)
    char_ba = compute_characteristics(G_ba)

    print_characteristics("Erdős-Rényi Model", char_er)
    print_characteristics("Watts-Strogatz Model", char_ws)
    print_characteristics("Barabási-Albert Model", char_ba)

    print("\n✓ Characteristics computed successfully.")

    # =============================================================================
    # Characteristics Comparison (Table + CSV + Bar Plot)
    # =============================================================================

    comparison_df = pd.DataFrame({
        "Characteristic": list(char_er.keys()),
        "Erdos_Renyi": list(char_er.values()),
        "Watts_Strogatz": list(char_ws.values()),
        "Barabasi_Albert": list(char_ba.values())
    })

    print("\n\n==================== CHARACTERISTICS COMPARISON ====================")
    print(comparison_df.to_string(index=False))

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "characteristics_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n✓ Comparison CSV saved at: {csv_path}")

    # Bar chart comparison (numeric characteristics only)
    numeric_df = comparison_df[
        comparison_df["Characteristic"].isin(["Density", "Average Degree", "Clustering Coefficient"])
    ]

    plt.figure(figsize=(10, 6))
    x = np.arange(len(numeric_df["Characteristic"]))
    width = 0.25

    plt.bar(x - width, numeric_df["Erdos_Renyi"], width, label="Erdos-Renyi")
    plt.bar(x, numeric_df["Watts_Strogatz"], width, label="Watts-Strogatz")
    plt.bar(x + width, numeric_df["Barabasi_Albert"], width, label="Barabasi-Albert")

    plt.xticks(x, numeric_df["Characteristic"], fontsize=11)
    plt.ylabel("Value", fontsize=12)
    plt.title("Comparison of Network Characteristics", fontsize=14, fontweight="bold")
    plt.legend()

    comparison_plot_path = os.path.join(OUTPUT_DIR, "characteristics_comparison.png")
    plt.tight_layout()
    plt.savefig(comparison_plot_path, dpi=300)
    plt.close()

    print(f"✓ Comparison plot saved at: {comparison_plot_path}")

    # -------------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------------
    print("\nGenerating and saving visualizations...")

    visualize_graph(G_er, "Erdős-Rényi Random Network (1000 nodes)", "ER_model.png")
    visualize_graph(G_ws, "Watts-Strogatz Small World Network (1000 nodes)", "WS_model.png")
    visualize_graph(G_ba, "Barabási-Albert Scale-Free Network (1000 nodes)", "BA_model.png")

    print("✓ Visualizations saved in: P3/output/")

    print("\nALL TASKS COMPLETED SUCCESSFULLY.")
    print("=" * 75)