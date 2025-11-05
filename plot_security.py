import os
import ast
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


def _labels_on_bars(ax, values, fmt="{:.2f}"):
    for i, v in enumerate(values):
        ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=10)

def plot_sybil_detection():
    path_main = os.path.join(LOG_DIR, "sybil_similarity_reputation.csv")
    path_km = os.path.join(LOG_DIR, "sybil_kmeans.csv")

    if not os.path.exists(path_main):
        print("[!] sybil_similarity_reputation.csv bulunamadı. Fig 5-a atlanıyor.")
        return

    df = pd.read_csv(path_main)
    df_km = None
    if os.path.exists(path_km):
        df_km = pd.read_csv(path_km)
    elif {"cluster", "cluster_size"}.issubset(df.columns):
        df_km = df[["client_id", "cluster", "cluster_size"]].copy()

    plt.figure(figsize=(8, 6))
    if _HAS_SNS:
        sns.scatterplot(
            data=df,
            x="sim_score", y="combined",
            hue="flagged",
            style="cluster" if "cluster" in df.columns else None,
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

    # K-means küme büyüklükleri
    if df_km is not None and {"cluster", "cluster_size"}.issubset(df_km.columns):
        cluster_sizes = df_km.drop_duplicates("cluster")["cluster_size"].astype(int)
        labels = df_km.drop_duplicates("cluster")["cluster"].astype(str)
        plt.figure(figsize=(7, 5))
        ax = plt.gca()
        ax.bar(labels, cluster_sizes)
        plt.title("Fig 5-a (Ek): K-means Cluster Sizes")
        plt.xlabel("Cluster")
        plt.ylabel("Cluster Size (#clients)")
        for i, v in enumerate(cluster_sizes.tolist()):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "fig5a_sybil_cluster_sizes.png"))
        plt.close()


def plot_poisoning_impact():
    path = os.path.join(LOG_DIR, "poisoning_norms.csv")
    if not os.path.exists(path):
        print("[!] poisoning_norms.csv bulunamadı. Fig 5-b atlanıyor.")
        return

    df = pd.read_csv(path).sort_values("client_id").reset_index(drop=True)

    plt.figure(figsize=(10, 5.5))
    if _HAS_SNS:
        sns.barplot(data=df, x="client_id", y="robust_z", hue="flagged", dodge=False)
        plt.legend(title="Flagged", loc="upper right")
    else:
        ax = plt.gca()
        colors = ["C0" if f == 0 else "C3" for f in df.get("flagged", [0]*len(df))]
        ax.bar(df["client_id"].astype(str), df["robust_z"], color=colors)
    plt.title("Fig 5-b: Poisoning Detection — Robust Z-scores")
    plt.xlabel("Client ID")
    plt.ylabel("Robust Z-score (|z| > 3 ⇒ flagged)")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "fig5b_poisoning_zscores.png"))
    plt.close()


def _parse_cfg_string(cfg_str: str) -> dict:
    try:
        d = ast.literal_eval(cfg_str)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _load_crypto_bytes_per_round() -> pd.DataFrame:
    path = os.path.join(LOG_DIR, "crypto", "crypto_metrics.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["round", "crypto_bytes"]).astype({"round": int, "crypto_bytes": int})
    df = pd.read_csv(path)

    def _to_int_safe(x):
        try:
            return int(x)
        except Exception:
            return -1

    if "round" not in df.columns:
        return pd.DataFrame(columns=["round", "crypto_bytes"])  
    df["round_int"] = df["round"].apply(_to_int_safe)

    rows = []
    for r in sorted(df["round_int"].unique()):
        if r < 0:
            continue
        dfr = df[df["round_int"] == r]
        d_cli = dfr[dfr["mode"] == "server_decrypt"]
        if len(d_cli) > 0:
            crypto_bytes = d_cli["cipher_bytes"].fillna(0).astype(int).sum()
        else:
            d_sum = dfr[dfr["mode"] == "server_sum_decrypt"]
            crypto_bytes = int(d_sum["cipher_bytes"].fillna(0).astype(int).sum()) if len(d_sum) > 0 else 0
        rows.append({"round": r, "crypto_bytes": crypto_bytes})
    return pd.DataFrame(rows)


def _load_zkp_bytes_per_round() -> pd.DataFrame:
    path = os.path.join(LOG_DIR, "zkp", "zkp_metrics.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["round", "zkp_bytes"]).astype({"round": int, "zkp_bytes": int})
    df = pd.read_csv(path)

    def _to_int_safe(x):
        try:
            return int(x)
        except Exception:
            return -1

    if not {"round", "size_bytes"}.issubset(df.columns):
        return pd.DataFrame(columns=["round", "zkp_bytes"])  
    df["round_int"] = df["round"].apply(_to_int_safe)
    d = df[df["round_int"] >= 0].groupby("round_int")["size_bytes"].sum().reset_index()
    d.columns = ["round", "zkp_bytes"]
    return d


def _load_comm_bytes_per_round() -> pd.DataFrame:
    """Opsiyonel: istemci tarafındaki comm_metrics.csv varsa uplink/downlink ayrıştır."""
    path = os.path.join(LOG_DIR, "comm", "comm_metrics.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["round", "uplink_bytes", "downlink_bytes"]).astype({
            "round": int, "uplink_bytes": int, "downlink_bytes": int
        })
    df = pd.read_csv(path)
    if not {"round", "direction", "bytes"}.issubset(df.columns):
        return pd.DataFrame(columns=["round", "uplink_bytes", "downlink_bytes"])  # güvenlik
    pivot = df.pivot_table(index="round", columns="direction", values="bytes", aggfunc="sum").fillna(0)
    pivot = pivot.rename(columns={"up": "uplink_bytes", "down": "downlink_bytes"}).reset_index()
    if "uplink_bytes" not in pivot.columns:
        pivot["uplink_bytes"] = 0
    if "downlink_bytes" not in pivot.columns:
        pivot["downlink_bytes"] = 0
    return pivot.astype({"round": int, "uplink_bytes": int, "downlink_bytes": int})


def _load_cfg_per_round() -> pd.DataFrame:
    path = os.path.join(LOG_DIR, "chain", "chain_log.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["round", "category"])
    df = pd.read_csv(path)
    if not {"round", "mode"}.issubset(df.columns):
        return pd.DataFrame(columns=["round", "category"])

    rows = []
    for _, row in df.iterrows():
        try:
            r = int(row.get("round"))
        except Exception:
            continue
        cfg = _parse_cfg_string(str(row.get("mode", "{}")))
        secagg = int(cfg.get("secagg", 0))
        zkp = int(cfg.get("zkp", 0))
        sybil = int(cfg.get("sybil", 0))
        poison = int(cfg.get("poison", 0))
        ldp = int(cfg.get("ldp", 0))
        clouds = int(cfg.get("clouds", cfg.get("domains", 1)))

        any_security = any([secagg, zkp, sybil, poison, ldp])
        if not any_security:
            if clouds and clouds > 1:
                cat = "CCFL"
            else:
                cat = "FL"
        else:
            cat = "Secure-CCFL"
        rows.append({"round": r, "category": cat})
    return pd.DataFrame(rows).drop_duplicates(subset=["round"], keep="last")


def plot_communication_overhead_from_csv():
    crypto_df = _load_crypto_bytes_per_round()
    zkp_df = _load_zkp_bytes_per_round()
    comm_df = _load_comm_bytes_per_round()  
    cfg_df = _load_cfg_per_round()

    if len(crypto_df) == 0 and len(zkp_df) == 0 and len(comm_df) == 0:
        print("[!] crypto/zkp/comm CSV’leri bulunamadı. Fig 5-c atlanıyor.")
        return

    merged = pd.merge(crypto_df, zkp_df, on="round", how="outer").fillna(0)

    if len(comm_df) > 0:
        merged = pd.merge(merged, comm_df, on="round", how="left").fillna(0)
    else:
        merged["uplink_bytes"] = 0
        merged["downlink_bytes"] = 0


    merged["total_bytes"] = merged.get("crypto_bytes", 0) + merged.get("zkp_bytes", 0) \
                             + merged.get("uplink_bytes", 0) + merged.get("downlink_bytes", 0)


    if len(cfg_df) > 0:
        merged = pd.merge(merged, cfg_df, on="round", how="left")
    else:
        merged["category"] = "Secure-CCFL"

    merged["total_mb"] = merged["total_bytes"] / (1024 * 1024)

    grp = merged.groupby("category")["total_mb"].mean().reset_index()

    order = ["FL", "CCFL", "Secure-CCFL"]
    grp["order"] = grp["category"].apply(lambda x: order.index(x) if x in order else len(order))
    grp = grp.sort_values("order")

    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.bar(grp["category"], grp["total_mb"])
    plt.title("Fig 5-c: Communication Overhead per Round (from CSV)")
    plt.ylabel("Data per Round (MB)")
    plt.grid(True, axis="y", alpha=0.25)
    _labels_on_bars(ax, grp["total_mb"].tolist(), fmt="{:.2f} MB")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "fig5c_comm_overhead.png"))
    plt.close()



def main():
    plot_sybil_detection()
    plot_poisoning_impact()
    plot_communication_overhead_from_csv()
    print("[✓] Grafikler çizildi → plots/ klasörüne bakın.")


if __name__ == "__main__":
    main()
