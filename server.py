# server.py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from secure_aggregation import generate_key_pair, load_private_key, aggregate_encrypted, decrypt_model_update
import os
import pandas as pd

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def detect_sybil_clients(all_updates, poison_flags=None):
    updates_matrix = np.vstack(all_updates)
    sim_matrix = cosine_similarity(updates_matrix)

    avg_similarities = np.mean(sim_matrix, axis=1)
    #for i, sim in enumerate(avg_similarities):
    #    print(f"[DEBUG] Client {i} | Avg similarity: {sim:.4f}")

    candidate_indices = [i for i in range(len(avg_similarities)) if not poison_flags or not poison_flags[i]]

    threshold = np.percentile([avg_similarities[i] for i in candidate_indices], 10)
    sybil_indices = [i for i in candidate_indices if avg_similarities[i] < threshold]

    print("[Sybil Detection] Sybil detected:")
    for idx in sybil_indices:
        cloud = ["A", "B", "C"][idx // 5]
        print(f"  →  [Cloud {cloud}] Client {(idx % 5) + 1}")
    return sybil_indices


def detect_poisoned_clients(decrypted_updates):
    flattened = [np.array(update) for update in decrypted_updates]

    means = np.mean(flattened, axis=0)
    stds = np.std(flattened, axis=0)

    z_scores = []
    for update in flattened:
        z = np.abs((update - means) / (stds + 1e-6))
        z_scores.append(np.mean(z))
    #print(f"[Poisoning Detection] z_scores: {z_scores}")
    flags = [int(z > 2.0) for z in z_scores]

    df = pd.DataFrame({
        "client_id": list(range(len(flags))),
        "z_score": z_scores,
        "flagged": flags
    })
    os.makedirs(LOG_DIR, exist_ok=True)
    df.to_csv(os.path.join(LOG_DIR, "poisoning_zscores.csv"), index=False)

    #print(f"[Poisoning Detection] flags: {flags}")
    cloud_map = {0: "A", 1: "B", 2: "C"}
    print(f"[Poisoning Detection] Poisoned clients detected:")
    for idx, flagged in enumerate(flags):
        if flagged:
            cloud_code = cloud_map.get(idx // 5, "?")
            print(f"  →  [Cloud {cloud_code}] Client {(idx % 5) + 1}")

    return flags

def process_round(client_updates):
    print("Federated server starting")
    privkey = load_private_key()

    print("[Server] Decrypting individual client updates for detection")
    decrypted_updates = []
    for update in client_updates:
        decrypted_client_update = []
        for param in update:
            decrypted_param = [privkey.decrypt(x) for x in param]
            decrypted_client_update.extend(decrypted_param)
        decrypted_updates.append(np.array(decrypted_client_update))

    lengths = [len(x) for x in decrypted_updates]
    print("[Server] Decrypted update lengths:", lengths)
    assert len(set(lengths)) == 1, "Client updates have mismatched lengths!"

       # Poisoning Detection
    print("[Server] Running Poisoning detection")
    poison_flags = detect_poisoned_clients(decrypted_updates)

    # Sybil Detection
    print("[Server] Running Sybil detection")
    sybil_indices = detect_sybil_clients(decrypted_updates, poison_flags)


    # Sybil & Poison Overlap Log
    overlap = set(sybil_indices).intersection(set(i for i, f in enumerate(poison_flags) if f))
    if overlap:
        print(f"[WARNING] Clients detected as both Sybil and Poisoned: {list(overlap)}")


    # Secure Aggregation
    print("[Server] Starting secure aggregation")
    aggregated = aggregate_encrypted(client_updates)

    # Decrypt aggregated model
    print("[Server] Decrypting aggregated model")
    decrypted = decrypt_model_update(aggregated, privkey)

    # Global model update
    flat_model = np.concatenate([np.array(p) for p in decrypted])
    model_path = os.path.join("models", "global_model.npy")
    os.makedirs("models", exist_ok=True)
    np.save(model_path, flat_model)
    print(f"[Server] Global model updated and saved to {model_path}")

def load_global_model():
    model_path = os.path.join("models", "global_model.npy")
    if not os.path.exists(model_path):
        print("[Server] No existing global model found.")
        return None
    model = np.load(model_path)
    print(f"[Server] Global model loaded from {model_path}")
    return model
