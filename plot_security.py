import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOG_DIR = "logs"
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False


def _labels_on_bars(ax, values, fmt="{:.2f}"):
    for i, v in enumerate(values):
        ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=10)


def _parse_cfg_string(cfg_str: str) -> dict:
    try:
        d = ast.literal_eval(str(cfg_str))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _category_from_cfg(cfg: dict) -> str:
    secagg = int(cfg.get("secagg", 0))
    zkp = int(cfg.get("zkp", 0))
    sybil = int(cfg.get("sybil", 0))
    poison = int(cfg.get("poison", 0))
    ldp = int(cfg.get("ldp", 0))
    clouds = int(cfg.get("clouds", cfg.get("domains", 1)))
    any_security = any([secagg, zkp, sybil, poison, ldp])
    if not any_security:
        return "CCFL" if clouds and clouds > 1 else "FL"
    return "Secure-CCFL"


def _poisoning_rate_from_cfg(cfg: dict) -> float:
    for k in ["adv_fraction", "poison_frac", "poisoning_rate", "adv_rate"]:
        if k in cfg:
            try:
                return float(cfg[k])
            except Exception:
                pass
    return 0.0


def plot_sybil_precision_recall_from_csv():
    main_path = os.path.join(LOG_DIR, "sybil_similarity_reputation.csv")
    if not os.path.exists(main_path):
        print("[!] sybil_similarity_reputation.csv not found — skipping (a)")
        return
    df = pd.read_csv(main_path)
    gt_path = os.path.join(LOG_DIR, "ground_truth_sybil.csv")
    if os.path.exists(gt_path):
        gt = pd.read_csv(gt_path)
        if "round" in gt.columns and "round" in df.columns:
            df = pd.merge(df, gt[["round", "client_id", "is_sybil"]], on=["round", "client_id"], how="left")
        else:
            df = pd.merge(df, gt[["client_id", "is_sybil"]], on=["client_id"], how="left")
    if "is_sybil" not in df.columns:
        if {"cluster", "cluster_size"}.issubset(df.columns):
            cutoff = max(2, int(max(1, len(df)) * 0.2))
            small = df["cluster_size"] < cutoff
            df["is_sybil"] = small.astype(int)
        else:
            print("[!] No ground truth and no cluster_size — cannot derive labels for (a)")
            return
    thresholds = [0.2, 0.4, 0.6, 0.8, 0.9]
    prec, rec = [], []
    for thr in thresholds:
        pred = (df["combined"].values > thr).astype(int)
        y = df["is_sybil"].values.astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec.append(precision)
        rec.append(recall)
    plt.figure(figsize=(7, 4.5))
    plt.plot(thresholds, prec, marker='o', label='Precision')
    plt.plot(thresholds, rec, marker='o', label='Recall')
    plt.title('a) Sybil Detection Performance vs Reputation Threshold')
    plt.xlabel('Reputation Threshold')
    plt.ylabel('Detection Metric')
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    out = os.path.join(PLOT_DIR, 'fig_sybil_threshold.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[✓] {out}")


def plot_accuracy_vs_poisoning_from_csv():
    eval_path = os.path.join(LOG_DIR, "extended_eval.csv")
    chain_path = os.path.join(LOG_DIR, "chain", "chain_log.csv")
    if not (os.path.exists(eval_path) and os.path.exists(chain_path)):
        print("[!] extended_eval.csv or chain_log.csv not found — skipping (b)")
        return
    df = pd.read_csv(eval_path)
    ch = pd.read_csv(chain_path)
    ch["cfg"] = ch["mode"].apply(_parse_cfg_string)
    ch["category"] = ch["cfg"].apply(_category_from_cfg)
    ch["poison_rate"] = ch["cfg"].apply(_poisoning_rate_from_cfg)
    if "round" not in df.columns:
        print("[!] 'round' column missing in extended_eval.csv — skipping (b)")
        return
    merged = pd.merge(df, ch[["round", "category", "poison_rate"]], on="round", how="left")
    merged["poison_pct"] = (merged["poison_rate"].fillna(0) * 100).round().astype(int)
    g = merged.groupby(["category", "poison_pct"])['accuracy_pct'].mean().reset_index()
    order = ["FL", "CCFL", "Secure-CCFL"]
    markers = {"FL": 'o', "CCFL": 's', "Secure-CCFL": '^'}
    plt.figure(figsize=(8, 5))
    for cat in order:
        sub = g[g['category'] == cat].sort_values('poison_pct')
        if len(sub) == 0:
            continue
        plt.plot(sub['poison_pct'], sub['accuracy_pct'], marker=markers[cat], linestyle='-', label=cat)
        for xv, yv in zip(sub['poison_pct'], sub['accuracy_pct']):
            plt.text(xv, yv + 0.6, f"{yv:.0f}", ha='center', va='bottom', fontsize=9)
    plt.title('b ) Global Accuracy vs Poisoning Rate')
    plt.xlabel('Poisoning Rate (%)')
    plt.ylabel('Global Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Mode', loc='lower left')
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, 'Figure_1.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[✓] {out}")


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

    df["round_int"] = df.get("round", -1).apply(_to_int_safe)
    rows = []
    for r in sorted(df["round_int"].unique()):
        if r < 0:
            continue
        dfr = df[df["round_int"] == r]
        d_cli = dfr[dfr.get("mode", "") == "server_decrypt"]
        if len(d_cli) > 0:
            crypto_bytes = d_cli.get("cipher_bytes", 0).fillna(0).astype(int).sum()
        else:
            d_sum = dfr[dfr.get("mode", "") == "server_sum_decrypt"]
            crypto_bytes = int(d_sum.get("cipher_bytes", 0).fillna(0).astype(int).sum()) if len(d_sum) > 0 else 0
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

    df["round_int"] = df.get("round", -1).apply(_to_int_safe)
    d = df[df["round_int"] >= 0].groupby("round_int")["size_bytes"].sum().reset_index()
    d.columns = ["round", "zkp_bytes"]
    return d


def _load_comm_bytes_per_round() -> pd.DataFrame:
    path = os.path.join(LOG_DIR, "comm", "comm_metrics.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["round", "uplink_bytes", "downlink_bytes"]).astype({
            "round": int, "uplink_bytes": int, "downlink_bytes": int
        })
    df = pd.read_csv(path)
    if not {"round", "direction", "bytes"}.issubset(df.columns):
        return pd.DataFrame(columns=["round", "uplink_bytes", "downlink_bytes"])
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
        cfg = _parse_cfg_string(row.get("mode", "{}"))
        rows.append({"round": r, "category": _category_from_cfg(cfg)})
    return pd.DataFrame(rows).drop_duplicates(subset=["round"], keep="last")


def plot_communication_overhead_from_csv():
    crypto_df = _load_crypto_bytes_per_round()
    zkp_df = _load_zkp_bytes_per_round()
    comm_df = _load_comm_bytes_per_round()
    cfg_df = _load_cfg_per_round()
    if len(crypto_df) == 0 and len(zkp_df) == 0 and len(comm_df) == 0:
        print("[!] crypto/zkp/comm CSVs not found — skipping (c)")
        return
    merged = pd.merge(crypto_df, zkp_df, on="round", how="outer").fillna(0)
    if len(comm_df) > 0:
        merged = pd.merge(merged, comm_df, on="round", how="left").fillna(0)
    else:
        merged["uplink_bytes"], merged["downlink_bytes"] = 0, 0
    merged["total_bytes"] = (
        merged.get("crypto_bytes", 0)
        + merged.get("zkp_bytes", 0)
        + merged.get("uplink_bytes", 0_
