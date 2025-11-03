import os
import time
import csv
import pickle
import random
import argparse
from string import ascii_uppercase

from secure_aggregation import generate_key_pair
from secure_aggregation import load_public_key
from cloud import simulate_cloud
from server import Server

import numpy as np
import torch
from model import ResNetForMNIST


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
KEYS_DIR = os.path.join(PROJECT_ROOT, "keys")
PUBKEY_PATH = os.path.join(KEYS_DIR, "public_key.pkl")
PRIVKEY_PATH = os.path.join(KEYS_DIR, "private_key.pkl")

LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
RESULTS_CSV = os.path.join(LOGS_DIR, "extended_eval.csv")
if not os.path.exists(RESULTS_CSV):
    with open(RESULTS_CSV, "w", newline="") as f:
        csv.writer(f).writerow([
            "mode", "arch", "classes_per_client", "latency_low_ms", "latency_high_ms",
            "adversary_fraction", "round", "num_clients", "update_len",
            "payload_comm_MB_est"
        ])

EXTRA_LATENCIES = [(10, 100), (50, 300)]
ADV_FRACTIONS   = [0.10, 0.30, 0.50]
CPC_LEVELS      = [1, 2, 5]
ARCHES          = ["resnet34"]


def _generate_or_load_paillier_keys(pub_path=PUBKEY_PATH, priv_path=PRIVKEY_PATH):
    os.makedirs(KEYS_DIR, exist_ok=True)
    if not (os.path.exists(pub_path) and os.path.exists(priv_path)):
        print("[Keys] Missing Paillier keys. Generating new key pair...")
        generate_key_pair(pub_path=PUBKEY_PATH, priv_path=PRIVKEY_PATH)
    with open(pub_path, "rb") as f:
        public_key = pickle.load(f)
    with open(priv_path, "rb") as f:
        private_key = pickle.load(f)
    pub = load_public_key()
    try:
        print(f"[SECURE-AGG] Paillier key bits: {pub.n.bit_length()}")
    except Exception:
        pass
    print(f"[Keys] Loaded public/private keys from:\n  {pub_path}\n  {priv_path}")
    return public_key, private_key


def _make_attack_map(total_clients: int, adv_fraction: float, seed: int = 42):
    num_adv = int(round(total_clients * adv_fraction))
    num_sybil = (num_adv + 1) // 2
    num_poison = num_adv // 2
    ids = list(range(total_clients))
    random.seed(seed)
    random.shuffle(ids)
    amap = {}
    for i in ids[:num_sybil]:
        amap[i] = "sybil"
    for i in ids[num_sybil:num_sybil + num_poison]:
        amap[i] = "poison"
    return amap


def _init_global_if_missing(arch: str, models_dir: str = None):
    models_dir = models_dir or os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    out_path = os.path.join(models_dir, "global_model.npy")
    if os.path.exists(out_path):
        print(f"[Init] Found existing global model at {out_path} (no init needed).")
        return
    print("[Init] No global model found. Initializing fc-only absolute weights...")
    tmp = ResNetForMNIST(num_classes=10, arch=arch, freeze_backbone=True)
    fc_w = tmp.backbone.fc.weight.detach().view(-1).cpu().numpy()
    if tmp.backbone.fc.bias is not None:
        fc_b = tmp.backbone.fc.bias.detach().view(-1).cpu().numpy()
        flat = np.concatenate([fc_w, fc_b]).astype(np.float32)
    else:
        flat = fc_w.astype(np.float32)
    np.save(out_path, flat)
    print(f"[Init] Saved initial absolute fc weights to {out_path} (shape={flat.shape[0]}).")


def run_simulation(
    rounds: int,
    arch: str,
    classes_per_client: int,
    latency_range,
    adv_fraction: float,
    mode: str,
    num_clouds: int,
    clients_per_cloud: int
):
    if mode == "fl":
        cfg = dict(clouds=1, ldp=0, sybil=0, poison=0, secagg=0, chain=0, zkp=0)
        num_clouds_eff = 1
    elif mode == "ccfl":
        cfg = dict(clouds=num_clouds, ldp=0, sybil=0, poison=0, secagg=0, chain=0, zkp=0)
        num_clouds_eff = num_clouds
    elif mode == "secure":
        cfg = dict(clouds=num_clouds, ldp=1, sybil=1, poison=1, secagg=1, chain=1, zkp=1)
        num_clouds_eff = num_clouds
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"[Sim] Starting in mode='{mode}' with cfg={cfg}")
    _init_global_if_missing(arch=arch, models_dir=os.path.join(PROJECT_ROOT, "models"))
    public_key = private_key = None
    if cfg["secagg"]:
        public_key, private_key = _generate_or_load_paillier_keys()
    server = Server(public_key=public_key, private_key=private_key, cfg=cfg)
    total_clients = num_clouds_eff * clients_per_cloud
    attack_map = _make_attack_map(total_clients, adv_fraction) if (cfg["sybil"] or cfg["poison"]) else {}
    CLOUD_NAMES = [ascii_uppercase[i] for i in range(num_clouds_eff)]
    for rnd in range(1, rounds + 1):
        print(f"\n=========== ROUND {rnd}/{rounds} ===========\n")
        print(f"[Config] mode={mode} | arch={arch} | classes_per_client={classes_per_client} | "
              f"latency={latency_range} ms | adversaries={int(adv_fraction*100)}%")
        print("[DEBUG] attack_map:", attack_map if attack_map else "<empty>")
        all_updates = []
        all_labels = []
        for ci, cloud_letter in enumerate(CLOUD_NAMES):
            start_id = ci * clients_per_cloud
            ups, labs = simulate_cloud(
                cloud_letter,
                start_client_id=start_id,
                attack_map=attack_map,
                round_id=rnd,
                arch=arch,
                classes_per_client=classes_per_client,
                latency_range=latency_range,
                clients_per_cloud=clients_per_cloud,
                cfg=cfg
            )
            global_labels = []
            for i_local in range(len(labs)):
                global_idx = start_id + i_local
                global_labels.append(f"[Cloud {cloud_letter}] Client {(i_local+1)} (id={global_idx})")
            all_updates.extend(ups)
            all_labels.extend(global_labels)
        if len(all_updates) != total_clients:
            print(f"[Warning] Expected {total_clients} updates but got {len(all_updates)}. Check simulate_cloud clients_per_cloud.")
        print(f"[Debug] total collected updates: {len(all_updates)}")
        server.process_round(all_updates, client_labels=all_labels, current_round=rnd)
        try:
            update_len = len(all_updates[0]) if all_updates else 0
            payload_comm_mb = update_len * len(all_updates) * 256 / (1024 * 1024)
        except Exception:
            update_len = 0
            payload_comm_mb = 0.0
        with open(RESULTS_CSV, "a", newline="") as f:
            csv.writer(f).writerow([
                mode,
                arch,
                classes_per_client,
                latency_range[0], latency_range[1],
                adv_fraction,
                rnd,
                len(all_updates),
                update_len,
                f"{payload_comm_mb:.2f}",
            ])
        time.sleep(1)
    print("\n[Simulation] All rounds completed.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["fl", "ccfl", "secure"], default="secure",
                   help="fl=tek bulut, ccfl=çok bulut, secure=çok bulut + güvenlik katmanları")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--arch", type=str, default="resnet34")
    p.add_argument("--classes_per_client", type=int, default=2)
    p.add_argument("--latency_low", type=int, default=50)
    p.add_argument("--latency_high", type=int, default=300)
    p.add_argument("--adv_fraction", type=float, default=0.30)
    p.add_argument("--clouds", type=int, default=12)
    p.add_argument("--clients_per_cloud", type=int, default=15)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    latency_range = (args.latency_low, args.latency_high)
    run_simulation(
        rounds=args.rounds,
        arch=args.arch,
        classes_per_client=args.classes_per_client,
        latency_range=latency_range,
        adv_fraction=args.adv_fraction,
        mode=args.mode,
        num_clouds=args.clouds,
        clients_per_cloud=args.clients_per_cloud
    )
