import os
import csv
import time
import pickle
import hashlib
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from secure_aggregation import (
    load_private_key,
    load_public_key,
    decrypt_vector,
    decrypt_and_sum_vectors,
)

LOG_DIR = "logs"
MODEL_DIR = "models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

REPUTATION_PATH = os.path.join(LOG_DIR, "reputation.csv")


def _client_label(i: int, client_labels: Optional[List[str]] = None, clients_per_cloud: int = 5) -> str:
    if client_labels and i < len(client_labels):
        return client_labels[i]
    cloud_idx = i // clients_per_cloud
    cloud_letter = chr(ord("A") + cloud_idx) if cloud_idx < 26 else f"#{cloud_idx}"
    return f"[Cloud {cloud_letter}] Client {(i % clients_per_cloud) + 1}"


def _load_or_init_reputation(n_clients: int):
    if os.path.exists(REPUTATION_PATH):
        try:
            rep = pd.read_csv(REPUTATION_PATH)
            if len(rep) == n_clients and "reputation" in rep.columns:
                return rep["reputation"].values.astype(np.float32)
        except Exception:
            pass
    # Not: daha tarafsız başlangıç için 0.5 ile başlatıyoruz
    return np.full(n_clients, 0.5, dtype=np.float32)


def _save_reputation(rep: np.ndarray):
    pd.DataFrame({"reputation": rep}).to_csv(REPUTATION_PATH, index=False)


def _mean_stack(updates: List[np.ndarray]) -> np.ndarray:
    X = np.vstack([np.asarray(u, dtype=np.float32) for u in updates])
    return np.mean(X, axis=0).astype(np.float32)


def _prev_plus_delta_and_save(avg_delta: np.ndarray, out_path: str):
    avg_delta = np.atleast_1d(np.asarray(avg_delta, dtype=np.float32))
    if os.path.exists(out_path):
        try:
            prev = np.load(out_path).astype(np.float32)
            if prev.shape != avg_delta.shape:
                prev = np.zeros_like(avg_delta, dtype=np.float32)
        except Exception:
            prev = np.zeros_like(avg_delta, dtype=np.float32)
    else:
        prev = np.zeros_like(avg_delta, dtype=np.float32)
    new_weights = prev + avg_delta
    np.save(out_path, new_weights)
    print(f"[Server] Global model (fc-only, ABSOLUTE) saved to {out_path}")


def _decrypt_all(client_updates, private_key, current_round: int = -1):
    decrypted_updates = []
    os.makedirs(os.path.join(LOG_DIR, "crypto"), exist_ok=True)
    csv_path = os.path.join(LOG_DIR, "crypto", "crypto_metrics.csv")
    need_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(
                ["round", "client", "cipher_bytes", "plain_bytes", "encrypt_ms", "decrypt_ms", "mode"]
            )
        for i, arr in enumerate(client_updates):
            try:
                cipher_bytes = sum(len(pickle.dumps(e, protocol=pickle.HIGHEST_PROTOCOL)) for e in arr)
            except Exception:
                try:
                    cipher_bytes = sum(len(e) for e in arr)
                except Exception:
                    cipher_bytes = len(arr)
            t0 = time.perf_counter()
            plain_list = decrypt_vector(arr, private_key)
            dec_ms = (time.perf_counter() - t0) * 1000.0
            plain = np.asarray(plain_list, dtype=np.float32)
            decrypted_updates.append(plain)
            w.writerow(
                [
                    current_round,
                    i,
                    cipher_bytes,
                    int(plain.nbytes),
                    "",
                    f"{dec_ms:.3f}",
                    "server_decrypt",
                ]
            )
    return decrypted_updates


def _extract_updates_and_proofs(client_updates):
    updates, proofs = [], []
    for item in client_updates:
        if isinstance(item, dict) and "update" in item:
            updates.append(item["update"])
            proofs.append(item.get("zkp", None))
        else:
            updates.append(item)
            proofs.append(None)
    return updates, proofs


def _zkp_verify_log(round_id: int, client_idx: int, size: int, time_ms: float):
    os.makedirs(os.path.join(LOG_DIR, "zkp"), exist_ok=True)
    path = os.path.join(LOG_DIR, "zkp", "zkp_metrics.csv")
    need_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "round",
                "cloud",
                "client_idx",
                "client_id",
                "phase",
                "size_bytes",
                "time_ms",
                "accepted",
            ],
        )
        if need_header:
            w.writeheader()
        w.writerow(
            {
                "round": round_id,
                "cloud": "",
                "client_idx": client_idx,
                "client_id": client_idx,
                "phase": "verify",
                "size_bytes": size,
                "time_ms": f"{time_ms:.3f}",
                "accepted": 1,
            }
        )


def _verify_synthetic_proofs(proofs, current_round: int = -1):
    any_verified = False
    for i, p in enumerate(proofs):
        if p is None:
            continue
        t0 = time.perf_counter()
        _ = hashlib.blake2b(p).digest()
        dt = (time.perf_counter() - t0) * 1000.0
        _zkp_verify_log(current_round, i, len(p), dt)
        any_verified = True
    if any_verified:
        print("[Server][ZKP] Synthetic proof verification simulated for received payloads.")


def _evaluate_and_log_accuracy(out_path: str, current_round: int, num_clients: int, cfg: dict):
    try:
        if not os.path.exists(out_path):
            print("[Eval] Skipped: no global model file found:", out_path)
            return
        from model import ResNetForMNIST
        from data_utils import get_mnist_dataset
        import torch
        from torch.utils.data import DataLoader
        arch = (cfg or {}).get("arch", "resnet34")
        net = ResNetForMNIST(num_classes=10, arch=arch, freeze_backbone=True)
        net.eval()
        flat = np.load(out_path).astype(np.float32)
        has_bias = net.backbone.fc.bias is not None
        w_num = net.backbone.fc.weight.numel()
        w_vals = torch.tensor(flat[:w_num]).view_as(net.backbone.fc.weight)
        with torch.no_grad():
            net.backbone.fc.weight.copy_(w_vals)
            if has_bias:
                b_slice = flat[w_num:w_num + net.backbone.fc.bias.numel()]
                if b_slice.size > 0:
                    b_vals = torch.tensor(b_slice).view_as(net.backbone.fc.bias)
                    net.backbone.fc.bias.copy_(b_vals)
        _, test = get_mnist_dataset()
        test_loader = DataLoader(test, batch_size=256, shuffle=False)
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                logits = net(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = 100.0 * correct / max(1, total)
        targeted_asr = ""
        targeted_correct = ""
        targeted_total = ""
        try:
            target_label = (cfg or {}).get("backdoor_target_label", None)
            trigger_value = float((cfg or {}).get("backdoor_trigger_value", 1.0))
            if target_label is not None:
                synth_n = 128
                device = next(net.parameters()).device if any(p.requires_grad for p in net.parameters()) else "cpu"
                import torch
                x = torch.rand(synth_n, 1, 28, 28) * 255.0
                x[:, :, 0:2, 0:2] = torch.clamp(x[:, :, 0:2, 0:2] + trigger_value * 255.0, 0, 255)
                x = x / 255.0
                x = x.to(device)
                with torch.no_grad():
                    logits = net(x)
                    pred = logits.argmax(dim=1).cpu().numpy()
                targeted_total = synth_n
                targeted_correct = int((pred == int(target_label)).sum())
                targeted_asr = f"{100.0 * targeted_correct / max(1, targeted_total):.2f}"
                print(f"[Eval] Targeted ASR: {targeted_asr}% ({targeted_correct}/{targeted_total})")
        except Exception as _e:
            print("[Eval] Targeted ASR eval skipped:", _e)
        os.makedirs(LOG_DIR, exist_ok=True)
        csv_path = os.path.join(LOG_DIR, "extended_eval.csv")
        need_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if need_header:
                w.writerow([
                    "round", "num_clients", "arch",
                    "accuracy_pct", "correct", "total",
                    "targeted_asr_pct", "asr_correct", "asr_total"
                ])
            w.writerow([
                current_round, num_clients, arch,
                f"{acc:.2f}", correct, total,
                targeted_asr, targeted_correct, targeted_total
            ])
    except Exception as e:
        print("[Eval] Warning: evaluation failed:", e)


def _poisoning_detection(decrypted_updates, z_thresh: float = 3.0):
    if not decrypted_updates:
        return []
    norms = np.array([np.linalg.norm(np.asarray(u, dtype=np.float32)) for u in decrypted_updates], dtype=np.float32)
    mu, sigma = norms.mean(), norms.std() + 1e-8
    robust_z = (norms - mu) / sigma
    flags = (np.abs(robust_z) > z_thresh).astype(int)
    pd.DataFrame(
        {
            "client_id": list(range(len(norms))),
            "l2_norm": norms,
            "robust_z": robust_z,
            "flagged": flags,
        }
    ).to_csv(os.path.join(LOG_DIR, "poisoning_norms.csv"), index=False)
    print("[Server] Running Poisoning detection")
    poisoned_indices = [i for i, f in enumerate(flags) if f == 1]
    if poisoned_indices:
        print("[Poisoning Detection] Poisoned clients detected:")
        for i in poisoned_indices:
            print(f"  →  {_client_label(i)}")
    else:
        print("[Poisoning Detection] No clients flagged.")
    return poisoned_indices


def _sybil_detection(
    decrypted_updates,
    poisoned_indices=None,
    save_matrix: bool = True,
    client_labels: Optional[List[str]] = None,
):
    print("[Server] Running Sybil detection (similarity + K-means + reputation)")
    if not decrypted_updates:
        print("[Sybil Detection] No client updates available.")
        return []
    N = len(decrypted_updates)
    poisoned_set = set(poisoned_indices or [])
    candidates = [i for i in range(N) if i not in poisoned_set]
    if not candidates:
        print("[Sybil Detection] No non-poison candidates to evaluate.")
        return []

    X = np.vstack([np.asarray(u, np.float32) for u in decrypted_updates])
    S = cosine_similarity(X)
    np.fill_diagonal(S, 1.0)

    if N > 1:
        S_no_self = S.copy()
        np.fill_diagonal(S_no_self, -1.0)
        sim_score = S_no_self.max(axis=1).astype(np.float32)
    else:
        sim_score = np.zeros(N, dtype=np.float32)

    k = 3 if N >= 3 else 2
    try:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        cluster_labels = km.fit_predict(S)
    except Exception:
        cluster_labels = np.zeros(N, dtype=int)
        k = 1

    cluster_sizes = np.bincount(cluster_labels, minlength=k)

    sim_thr = 0.95
    lambda_sim = 0.6
    rep_alpha = 0.6
    rep = _load_or_init_reputation(N)
    rep = (1 - rep_alpha) * rep + rep_alpha * (sim_score > sim_thr).astype(np.float32)

    combined = (lambda_sim * sim_score) + ((1.0 - lambda_sim) * rep)
    combined_threshold = 0.60

    small_cut = max(2, int(max(1, N) * 0.2))
    small_clusters = set(np.where(cluster_sizes < small_cut)[0].tolist())

    high_sim_candidates = set(np.where(sim_score > sim_thr)[0].tolist())
    cluster_candidates = set(i for i, c in enumerate(cluster_labels) if c in small_clusters)

    cluster_candidates = [i for i in cluster_candidates if i in candidates]
    high_sim_candidates = [i for i in high_sim_candidates if i in candidates]

    sybil_flags = np.zeros(N, dtype=int)
    for i in set(cluster_candidates) | set(high_sim_candidates):
        if combined[i] > combined_threshold:
            sybil_flags[i] = 1

    df = pd.DataFrame(
        {
            "client_id": list(range(N)),
            "cluster": cluster_labels,
            "cluster_size": [cluster_sizes[c] for c in cluster_labels],
            "sim_score": sim_score,
            "reputation": rep,
            "combined": combined,
            "flagged": sybil_flags,
        }
    )
    df.to_csv(os.path.join(LOG_DIR, "sybil_similarity_reputation.csv"), index=False)

    pd.DataFrame(
        {
            "client_id": list(range(N)),
            "cluster": cluster_labels,
            "cluster_size": [cluster_sizes[c] for c in cluster_labels],
        }
    ).to_csv(os.path.join(LOG_DIR, "sybil_kmeans.csv"), index=False)

    _save_reputation(rep)

    flagged_indices = [i for i, f in enumerate(sybil_flags) if f == 1]
    if flagged_indices:
        print("[Sybil Detection] Suspected Sybil clients:")
        for i in flagged_indices:
            label = _client_label(i, client_labels=client_labels)
            print(
                f"  →  {label} | cluster={cluster_labels[i]} size={cluster_sizes[cluster_labels[i]]} "
                f"sim={sim_score[i]:.3f} rep={rep[i]:.3f} comb={combined[i]:.3f}"
            )
    else:
        print("[Sybil Detection] No clients flagged.")
    return flagged_indices


def _secure_aggregate_and_decrypt(client_updates, pubkey, privkey, average: bool = True):
    os.makedirs(os.path.join(LOG_DIR, "crypto"), exist_ok=True)
    csv_path = os.path.join(LOG_DIR, "crypto", "crypto_metrics.csv")
    need_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(["round", "client", "cipher_bytes", "plain_bytes", "encrypt_ms", "decrypt_ms", "mode"])
    try:
        total_cipher_bytes = sum(
            sum(len(pickle.dumps(e, protocol=pickle.HIGHEST_PROTOCOL)) for e in arr) for arr in client_updates
        )
    except Exception:
        try:
            total_cipher_bytes = sum(sum(len(e) for e in arr) for arr in client_updates)
        except Exception:
            total_cipher_bytes = sum(len(arr) for arr in client_updates)
    t0 = time.perf_counter()
    summed_plain = decrypt_and_sum_vectors(client_updates, privkey)
    dec_ms = (time.perf_counter() - t0) * 1000
    arr = np.asarray(summed_plain, dtype=np.float32)
    avg = arr / max(1, len(client_updates)) if average else arr
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(
            [
                "",
                "SUM",
                int(total_cipher_bytes),
                int(arr.nbytes),
                "",
                f"{dec_ms:.3f}",
                "server_sum_decrypt",
            ]
        )
    return avg.astype(np.float32)


def _digest_bytes_from_update(u) -> bytes:
    try:
        if isinstance(u, np.ndarray):
            return u.astype(np.float32, copy=False).tobytes()
        if isinstance(u, list):
            arr = np.asarray(u, dtype=np.float32)
            return arr.tobytes()
        return str(u).encode("utf-8", errors="ignore")
    except Exception:
        return b""


def _compute_update_hash_prefix(updates: List) -> str:
    h = hashlib.sha256()
    for u in updates:
        h.update(_digest_bytes_from_update(u))
    return h.hexdigest()[:16]


def _chain_log(round_id: int,
               n_clients: int,
               cfg: dict,
               update_hash_prefix: str,
               zkp_count: int,
               n_poisoned: int,
               n_sybil: int,
               poisoned_indices=None,
               sybil_indices=None):
    os.makedirs(os.path.join(LOG_DIR, "chain"), exist_ok=True)
    path = os.path.join(LOG_DIR, "chain", "chain_log.csv")
    need_header = not os.path.exists(path)
    poisoned_indices = poisoned_indices or []
    sybil_indices = sybil_indices or []
    row = {
        "round": round_id,
        "n_clients": n_clients,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "zkp_count": zkp_count,
        "n_poisoned": n_poisoned,
        "n_sybil": n_sybil,
        "poisoned_indices": "|".join(map(str, poisoned_indices)),
        "sybil_indices": "|".join(map(str, sybil_indices)),
        "update_hash_prefix": update_hash_prefix,
        "mode": str(cfg),
    }
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if need_header:
            w.writeheader()
        w.writerow(row)
    print(f"[Blockchain] Off-chain audit log updated (round={round_id}, flags P={poisoned_indices}, S={sybil_indices})")


def _chain_log_detailed(round_id: int, flags_len: int, poisoned_indices, sybil_indices):
    os.makedirs(os.path.join(LOG_DIR, "chain"), exist_ok=True)
    path = os.path.join(LOG_DIR, "chain", "chain_anomalies.csv")
    need_header = not os.path.exists(path)
    poisoned_set = set(poisoned_indices or [])
    sybil_set = set(sybil_indices or [])
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["round", "client_idx", "poisoned", "sybil"])
        if need_header:
            w.writeheader()
        for i in range(flags_len):
            w.writerow({
                "round": round_id,
                "client_idx": i,
                "poisoned": int(i in poisoned_set),
                "sybil": int(i in sybil_set),
            })


def process_round(
    client_updates: List[list],
    cfg: dict = None,
    public_key=None,
    private_key=None,
    client_labels: Optional[List[str]] = None,
    average_updates: bool = True,
    current_round: int = -1,
) -> None:
    cfg = cfg or {}
    print("Federated server starting with cfg=", cfg)
    client_updates, zkp_payloads = _extract_updates_and_proofs(client_updates)
    if cfg.get("zkp", 0) or any(p is not None for p in zkp_payloads):
        _verify_synthetic_proofs(zkp_payloads, current_round=current_round)
    privkey = private_key or load_private_key() if cfg.get("secagg", 0) else None
    pubkey = public_key or load_public_key() if cfg.get("secagg", 0) else None
    out_path = os.path.join(MODEL_DIR, "global_model.npy")
    decrypted_updates = []
    poisoned_indices, sybil_indices = [], []
    if cfg.get("sybil", 0) or cfg.get("poison", 0):
        if not privkey:
            privkey = load_private_key()
        decrypted_updates = _decrypt_all(client_updates, privkey, current_round=current_round)
        poisoned_indices = _poisoning_detection(decrypted_updates, z_thresh=1.5) if cfg.get("poison", 0) else []
        sybil_indices = _sybil_detection(
            decrypted_updates,
            poisoned_indices=poisoned_indices,
            save_matrix=True,
            client_labels=client_labels,
        ) if cfg.get("sybil", 0) else []
        flagged = set(poisoned_indices) | set(sybil_indices)
        pool = [u for i, u in enumerate(decrypted_updates) if i not in flagged] or decrypted_updates
        avg_delta = _mean_stack(pool) if average_updates else np.sum(np.vstack(pool), axis=0).astype(np.float32)
        _prev_plus_delta_and_save(avg_delta, out_path)
    else:
        if cfg.get("secagg", 0):
            avg_delta = _secure_aggregate_and_decrypt(client_updates, pubkey, privkey, average=average_updates)
            _prev_plus_delta_and_save(avg_delta, out_path)
        else:
            try:
                plain = [np.atleast_1d(np.asarray(u, dtype=np.float32)) for u in client_updates]
            except Exception:
                raise RuntimeError("Expecting plaintext updates when secagg==0 and detection disabled.")
            avg_delta = _mean_stack(plain) if average_updates else np.sum(np.vstack(plain), axis=0).astype(np.float32)
            _prev_plus_delta_and_save(avg_delta, out_path)
    if cfg.get("chain", 0):
        updates_for_digest = decrypted_updates if len(decrypted_updates) > 0 else client_updates
        upd_hash = _compute_update_hash_prefix(updates_for_digest)
        _chain_log(
            round_id=current_round,
            n_clients=len(client_updates),
            cfg=cfg,
            update_hash_prefix=upd_hash,
            zkp_count=sum(1 for p in zkp_payloads if p is not None),
            n_poisoned=len(poisoned_indices),
            n_sybil=len(sybil_indices),
            poisoned_indices=poisoned_indices,
            sybil_indices=sybil_indices,
        )
        _chain_log_detailed(current_round, len(client_updates), poisoned_indices, sybil_indices)
    try:
        _evaluate_and_log_accuracy(out_path=out_path, current_round=current_round, num_clients=len(client_updates), cfg=cfg)
    except Exception as _e:
        print("[Eval] Non-fatal evaluation error:", _e)


class Server:
    def __init__(self, public_key=None, private_key=None, cfg: dict = None, logs_dir: str = LOG_DIR, models_dir: str = MODEL_DIR):
        self.public_key = public_key or (load_public_key() if cfg and cfg.get("secagg", 0) else None)
        self.private_key = private_key or (load_private_key() if cfg and cfg.get("secagg", 0) else None)
        self.logs_dir = logs_dir
        self.models_dir = models_dir
        self.cfg = cfg or {}
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def process_round(self, client_updates, client_labels=None, average_updates: bool = True, current_round: int = -1):
        process_round(
            client_updates=client_updates,
            cfg=self.cfg,
            public_key=self.public_key,
            private_key=self.private_key,
            client_labels=client_labels,
            average_updates=average_updates,
            current_round=current_round,
        )
