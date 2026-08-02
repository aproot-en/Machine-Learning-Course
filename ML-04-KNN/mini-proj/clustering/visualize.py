




## Create plots without displaying them (for servers without GUI) 


import matplotlib
matplotlib.use("Agg")     

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
def plot_elbow(k_values, inertias, out_path):



    plt.figure(figsize=(7, 4.5))
    plt.plot(k_values, inertias, "o-")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
 #   plt.title("Elbow Method")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
def plot_clusters(X_raw, labels, out_path, x_name="Weight (kg)", y_name="Height (cm)"):

    plt.figure(figsize=(7.5, 6))

    for c in range(labels.max() + 1):
        members = labels == c
        plt.scatter(X_raw[members, 0], X_raw[members, 1],
                    s=20, alpha=0.6, label=f"Cluster {c}")

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title("K-Means Clustering result")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
