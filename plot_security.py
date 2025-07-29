# plot_security.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

LOG_DIR = "logs"
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_sybil_detection():
    path = os.path.join(LOG_DIR, "sybil_similarity.csv")
    if not os.path.exists(path):
        print("[!] sybil_similarity.csv bulunamadı.")
        return

    df = pd.read_csv(path)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, cmap='coolwarm', square=True, cbar=True)
    plt.title("Fig 5-a: Sybil Similarity Matrix")
    plt.xlabel("Client")
    plt.ylabel("Client")
    plt.savefig(os.path.join(PLOT_DIR, "fig5a_sybil_similarity_matrix.png"))
    plt.close()

def plot_poisoning_impact():
    path = os.path.join(LOG_DIR, "poisoning_zscores.csv")
    if not os.path.exists(path):
        print("[!] poisoning_zscores.csv bulunamadı.")
        return

    df = pd.read_csv(path)
    plt.figure(figsize=(8, 6))
    sns.barplot(x="client_id", y="z_score", hue="flagged", data=df, palette={0: "blue", 1: "red"})
    plt.title("Fig 5-b: Poisoning Detection Z-Scores")
    plt.xlabel("Client ID")
    plt.ylabel("Average Z-Score")
    plt.legend(title="Flagged", loc="upper right")
    plt.savefig(os.path.join(PLOT_DIR, "fig5b_poisoning_zscores.png"))
    plt.close()

def plot_communication_overhead():
    systems = ["Plain FL", "Secure CCFL"]
    overhead = [6, 9.4]  # MB

    plt.figure()
    plt.bar(systems, overhead, color=['blue', 'orange'])
    plt.title("Fig 5-c: Communication Overhead")
    plt.ylabel("Data Sent per Round (MB)")
    plt.savefig(os.path.join(PLOT_DIR, "fig5c_comm_overhead.png"))
    plt.close()

def main():
    plot_sybil_detection()
    plot_poisoning_impact()
    plot_communication_overhead()
    print("[✓] Grafikler çizildi → klasör: plots/")

if __name__ == "__main__":
    main()
