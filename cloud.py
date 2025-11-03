import time
import random
from client import Client
from data_utils import get_mnist_dataset, split_non_iid
from secure_aggregation import load_public_key

def simulate_cloud(cloud_id, start_client_id=0, attack_map=None, round_id=None,
                   arch="resnet34", classes_per_client=2, latency_range=(10,100),
                   clients_per_cloud: int = 5, cfg=None):
    cfg = cfg or {}
    print(f"[Cloud {cloud_id}] Starting (start_client_id={start_client_id}, n_clients={clients_per_cloud}) cfg={cfg}\n")

    train_dataset, _ = get_mnist_dataset()
    loaders = split_non_iid(
        train_dataset,
        num_clients=clients_per_cloud,
        num_classes=10,
        classes_per_client=classes_per_client
    )

    pubkey = load_public_key() if cfg.get("secagg", 0) else None
    clients = []
    labels = []
    for i in range(clients_per_cloud):
        cid = start_client_id + i
        attack_type = attack_map.get(cid, None) if attack_map else None
        c = Client(
            cid=cid,
            dataloader=loaders[i],
            attack_type=attack_type,
            pubkey=pubkey,
            cloud_id=cloud_id,
            arch=arch,
            c_id=clients_per_cloud,
            zkp_enabled=cfg.get("zkp", 0) == 1,
        )
        clients.append(c)
        labels.append(f"[Cloud {cloud_id}] Client {i+1}")

    client_updates = []
    for idx, c in enumerate(clients):
        print(f"[Cloud {cloud_id}] Client {((c.cid) % clients_per_cloud) + 1} preparing (round={round_id})")

        if hasattr(c, "disable_ldp"):
            if not cfg.get("ldp", 0):
                try:
                    c.disable_ldp()
                    print(f"[Cloud {cloud_id}] Client {((c.cid) % clients_per_cloud) + 1} LDP disabled per cfg")
                except Exception:
                    pass
            else:
                if hasattr(c, "enable_ldp"):
                    try:
                        c.enable_ldp()
                        print(f"[Cloud {cloud_id}] Client {((c.cid) % clients_per_cloud) + 1} LDP enabled per cfg")
                    except Exception:
                        pass

        if not cfg.get("poison", 0) and getattr(c, "attack_type", None) == "poison":
            c.attack_type = None
        if not cfg.get("sybil", 0) and getattr(c, "attack_type", None) == "sybil":
            c.attack_type = None

        c.train(epochs=1, round_id=round_id)

        delay_ms = random.uniform(*latency_range)
        time.sleep(delay_ms / 1000.0)
        encrypted_update = c.get_model_update()
        client_updates.append(encrypted_update)
        print()

    print(f"[Cloud {cloud_id}] sending Model updates to server\n")
    return client_updates, labels
