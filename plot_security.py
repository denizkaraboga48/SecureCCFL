import os
import pandas as pd
import matplotlib.pyplot as plt


try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

LOG_DIR = "logs"
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def _save_annotated_bar(ax, values, fmt="{:.2f}"):
    for i, v in enumerate(values):
        ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=10)


def plot_sybil_detection():
    """
    Fig 5-a: Sybil (Cosine + K-means + Reputation)
    Girdi: logs/sybil_similarity_reputation.csv, logs/sybil_kmeans.csv
      - Columns (sybil_similarity_reputation.csv): client_id, cluster, cluster_size, sim_score, reputation, combined, flagged
      - Columns (sybil_kmeans.csv): client_id, cluster, cluster_size
    Çıktı: plots/fig5a_sybil_scatter.png, plots/fig5a_sybil_cluster_sizes.png
    """
    path_main = os.path.join(LOG_DIR, "sybil_similarity_reputation.csv")
    path_km = os.path.join(LOG_DIR, "sybil_kmeans.csv")

    if not os.path.exists(path_main):
        print("[!] sybil_similarity_reputation.csv bulunamadı. Fig 5-a atlanıyor.")
        return

    df = pd.read_csv(path_main)
    if os.path.exists(path_km):
        df_km = pd.read_csv(path_km)
    else:
  
        df_km = df[["client_id", "cluster", "cluster_size"]].copy() if "cluster" in df.columns else None


    plt.figure(figsize=(8, 6))
    if _HAS_SNS:
        ax = sns.scatterplot(
            data=df,
            x="sim_score", y="combined",
            hue="flagged", style="cluster" if "cluster" in df.columns else None,
            s=70
        )
    else:
        ax = plt.gca()
        colors = ["C0" if f == 0 else "C3" for f in df.get("flagged", [0]*len(df))]
        ax.scatter(df["sim_score"], df["combined"], c=colors)
    plt.title("Fig 5-a: Sybil Detection — sim_score vs combined")
    plt.xlabel("Average Cosine Similarity (sim_score)")
    plt.ylabel("Combined Score (λ·sim + (1−λ)·rep)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "fig5a_sybil_scatter.png"))
    plt.close()

  
    if df_km is not None and "cluster_size" in df_km.columns and "cluster" in df_km.columns:
        cluster_sizes = df_km.drop_duplicates("cluster")[["cluster", "cluster_size"]].sort_values("cluster")
        plt.figure(figsize=(7, 5))
        ax = plt.gca()
        ax.bar(cluster_sizes["cluster"].astype(str), cluster_sizes["cluster_size"])
        plt.title("Fig 5-a (Ek): K-means Cluster Sizes")
        plt.xlabel("Cluster")
        plt.ylabel("Cluster Size (#clients)")
        for i, v in enumerate(cluster_sizes["cluster_size"].tolist()):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "fig5a_sybil_cluster_sizes.png"))
        plt.close()


def plot_poisoning_impact():
    """
    Fig 5-b: Poisoning Detection (Robust Z-scores)
    Girdi: logs/poisoning_norms.csv
      - Columns: client_id, l2_norm, robust_z, flagged
    Çıktı: plots/fig5b_poisoning_zscores.png
    """
    path = os.path.join(LOG_DIR, "poisoning_norms.csv")
    if not os.path.exists(path):
        print("[!] poisoning_norms.csv bulunamadı. Fig 5-b atlanıyor.")
        return

    df = pd.read_csv(path)

    df = df.sort_values("client_id").reset_index(drop=True)

    plt.figure(figsize=(10, 5.5))
    ax = plt.gca()
    if _HAS_SNS:
        sns.barplot(
            data=df,
            x="client_id", y="robust_z",
            hue="flagged", dodge=False
        )
    else:
        colors = ["C0" if f == 0 else "C3" for f in df.get("flagged", [0]*len(df))]
        ax.bar(df["client_id"].astype(str), df["robust_z"], color=colors)
    plt.title("Fig 5-b: Poisoning Detection — Robust Z-scores")
    plt.xlabel("Client ID")
    plt.ylabel("Robust Z-score (|z| > 3 ⇒ flagged)")
    plt.legend(title="Flagged", loc="upper right") if _HAS_SNS else None
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "fig5b_poisoning_zscores.png"))
    plt.close()


def plot_communication_overhead():
    """
    Fig 5-c: Communication Overhead (Makaledeki sabit sayılar)
    FL=6.0 MB, CCFL=6.8 MB, Secure-CCFL=9.4 MB
    Çıktı: plots/fig5c_comm_overhead.png
    """
    systems = ["FL", "CCFL", "Secure-CCFL"]
    overhead = [6.0, 6.8, 9.4]  
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.bar(systems, overhead)
    plt.title("Fig 5-c: Communication Overhead per Round (MB)")
    plt.ylabel("Data per Round (MB)")
    plt.grid(True, axis="y", alpha=0.25)
    _save_annotated_bar(ax, overhead, fmt="{:.1f} MB")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "fig5c_comm_overhead.png"))
    plt.close()


def main():
    plot_sybil_detection()
    plot_poisoning_impact()
    plot_communication_overhead()
    print("[✓] Grafikler çizildi → klasör: plots/")


if __name__ == "__main__":
    main()
