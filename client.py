import os
import csv
import time
import pickle
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from dp_utils import apply_opacus
from model import ResNetForMNIST


def _dp_append_csv(csv_path, row_dict):
    header = [
        "timestamp",
        "round",
        "cloud",
        "client_idx",
        "client_id",
        "sigma",
        "delta",
        "sample_rate",
        "steps",
        "epsilon",
    ]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row_dict)


def _comm_append(row):
    os.makedirs("logs/comm", exist_ok=True)
    path = "logs/comm/comm_metrics.csv"
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "round",
                "cloud",
                "client_idx",
                "client_id",
                "dir",
                "bytes",
                "mode",
            ],
        )
        if new:
            w.writeheader()
        w.writerow(row)


def _zkp_append(row_dict):
    os.makedirs("logs/zkp", exist_ok=True)
    path = "logs/zkp/zkp_metrics.csv"
    new = not os.path.exists(path)
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
        if new:
            w.writeheader()
        w.writerow(row_dict)


class Client:
    def __init__(
        self,
        cid,
        dataloader,
        attack_type=None,
        pubkey=None,
        cloud_id=None,
        arch="resnet34",
        c_id=None,
        zkp_enabled=None,
        backdoor_inject_count: int = 5,
        backdoor_target_label: int = 0,
        backdoor_trigger_value: float = 1.0,
    ):
        self.cid = cid
        self.model = ResNetForMNIST(num_classes=10, arch=arch, freeze_backbone=True)
        self.dataloader = dataloader
        self.criterion = nn.CrossEntropyLoss()

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.SGD(
            trainable_params, lr=0.1, momentum=0.9, weight_decay=5e-4
        )

        self.attack_type = attack_type
        self.pubkey = pubkey
        self.cloud_id = cloud_id
        self.cloudclient = ((self.cid) % c_id) + 1 if c_id else self.cid

        self.ldp_enabled = True
        self.round_id = -1

        self._fc_init_w = None
        self._fc_init_b = None

        if zkp_enabled is None:
            self.zkp_enabled = self.pubkey is not None
        else:
            self.zkp_enabled = bool(zkp_enabled)

        self.backdoor_inject_count = int(backdoor_inject_count)
        self.backdoor_target_label = int(backdoor_target_label)
        self.backdoor_trigger_value = float(backdoor_trigger_value)

    def disable_ldp(self):
        self.ldp_enabled = False

    def enable_ldp(self):
        self.ldp_enabled = True

    def load_global_model(self, path="models/global_model.npy"):
        if os.path.exists(path):
            print(f"[Cloud {self.cloud_id}] Client {self.cloudclient} is loading global model...")
            flat_weights = np.load(path)
            total_params = sum(p.numel() for p in self.model.parameters())
            fc_w = self.model.backbone.fc.weight
            fc_b = self.model.backbone.fc.bias
            fc_count = fc_w.numel() + (fc_b.numel() if fc_b is not None else 0)

            if flat_weights.size == total_params:
                offset = 0
                for param in self.model.parameters():
                    size = param.data.numel()
                    values = flat_weights[offset : offset + size]
                    param.data = torch.tensor(values, dtype=param.data.dtype).view_as(param.data)
                    offset += size
            elif flat_weights.size == fc_count:
                w_num = fc_w.numel()
                w_vals = torch.tensor(flat_weights[:w_num], dtype=fc_w.dtype).view_as(fc_w)
                with torch.no_grad():
                    fc_w.copy_(w_vals)
                    if fc_b is not None:
                        b_vals = torch.tensor(
                            flat_weights[w_num : w_num + fc_b.numel()],
                            dtype=fc_b.dtype,
                        ).view_as(fc_b)
                        fc_b.copy_(b_vals)
            else:
                print(f"[Cloud {self.cloud_id}] Client {self.cloudclient} incompatible global vector length: {flat_weights.size}")

            print(f"[Cloud {self.cloud_id}] Client {self.cloudclient} successfully loaded global model.")

            fc_len = fc_w.numel() + (fc_b.numel() if fc_b is not None else 0)
            global_bytes = 4 * int(fc_len)
            _comm_append(
                {
                    "round": self.round_id,
                    "cloud": self.cloud_id,
                    "client_idx": self.cloudclient,
                    "client_id": self.cid,
                    "dir": "down",
                    "bytes": global_bytes,
                    "mode": "global_broadcast_plain",
                }
            )
        else:
            print(f"[Cloud {self.cloud_id}] Client {self.cloudclient} could not find global model at {path}")

    def _snapshot_fc(self):
        w = self.model.backbone.fc.weight.detach().clone().view(-1).cpu()
        b = (
            self.model.backbone.fc.bias.detach().clone().view(-1).cpu()
            if self.model.backbone.fc.bias is not None
            else None
        )
        self._fc_init_w = w
        self._fc_init_b = b

    def _inject_backdoor_into_dataset(self):
        try:
            ds = getattr(self.dataloader, "dataset", None)
            if ds is None:
                raise RuntimeError("Dataloader has no .dataset attribute; cannot inject in-place.")

            n = len(ds)
            k = min(self.backdoor_inject_count, n)
            if k <= 0:
                return 0

            rng = np.random.default_rng(seed=(self.cid + max(0, int(self.round_id))))
            indices = rng.choice(n, size=k, replace=False)

            for idx in indices:
                try:
                    if hasattr(ds, "data"):
                        item = ds.data[idx]
                        if isinstance(item, np.ndarray):
                            arr = item.astype(np.float32)
                            if arr.ndim == 2:
                                arr[0:2, 0:2] = np.clip(arr[0:2, 0:2] + self.backdoor_trigger_value * 255.0, 0, 255)
                            else:
                                arr[0:2, 0:2, :] = np.clip(arr[0:2, 0:2, :] + self.backdoor_trigger_value * 255.0, 0, 255)
                            ds.data[idx] = arr.astype(ds.data.dtype)
                        else:
                            t = item.clone().detach().float()
                            if t.dim() == 2:
                                t[0:2, 0:2] = torch.clamp(t[0:2, 0:2] + self.backdoor_trigger_value * 255.0, 0, 255)
                            else:
                                t[0:2, 0:2, :] = torch.clamp(t[0:2, 0:2, :] + self.backdoor_trigger_value * 255.0, 0, 255)
                            ds.data[idx] = t.type(item.dtype)
                    if hasattr(ds, "targets"):
                        try:
                            ds.targets[idx] = self.backdoor_target_label
                        except Exception:
                            if hasattr(ds.targets, "__setitem__"):
                                ds.targets[idx] = type(ds.targets)(self.backdoor_target_label)
                    elif hasattr(ds, "labels"):
                        ds.labels[idx] = self.backdoor_target_label
                except Exception:
                    continue

            print(f"[Client {self.cid}] Backdoor injection: injected {k} samples -> target {self.backdoor_target_label}")
            return k
        except Exception as e:
            print(f"[Client {self.cid}] Backdoor injection failed (in-place): {e}")
            return 0

    def train(self, epochs=1, round_id=None):
        self.round_id = -1 if round_id is None else int(round_id)

        self.load_global_model()
        self.model.train()

        if self.ldp_enabled:
            try:
                self.model, self.optimizer, self.dataloader = apply_opacus(
                    self.model,
                    self.optimizer,
                    self.dataloader,
                    client_id=self.cid,
                    cloud_id=self.cloud_id,
                    noise_multiplier=0.5,
                    max_grad_norm=0.7,
                    target_delta=1e-5,
                )
            except Exception as e:
                print(f"[Client {self.cid}] Warning: apply_opacus failed: {e}. Continuing without DP.")

        if self.attack_type == "backdoor":
            injected = self._inject_backdoor_into_dataset()
            if injected == 0:
                print(f"[Client {self.cid}] Backdoor injection skipped or failed; proceeding without injected samples.")

        self._snapshot_fc()

        for _ in range(epochs):
            for data, target in self.dataloader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                if hasattr(self.model, "_dp_meta") and self.ldp_enabled:
                    self.model._dp_meta["steps"] += 1

        if hasattr(self.model, "_dp_meta") and self.ldp_enabled:
            pe = self.model._dp_meta["engine"]
            delt = self.model._dp_meta["delta"]
            eps = pe.get_epsilon(delt)
            sr = self.model._dp_meta["sample_rate"]
            sig = self.model._dp_meta["sigma"]
            stp = self.model._dp_meta["steps"]
            logp = self.model._dp_meta["log_csv"]
            print(
                f"[DP][Cloud {self.cloud_id}][Client {self.cloudclient}] "
                f"sigma={sig}, delta={delt}, sample_rate={sr:.6f}, steps={stp} -> epsilon={eps:.3f}"
            )

            _dp_append_csv(
                logp,
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "round": self.round_id,
                    "cloud": self.cloud_id,
                    "client_idx": self.cloudclient,
                    "client_id": self.cid,
                    "sigma": sig,
                    "delta": delt,
                    "sample_rate": f"{sr:.8f}",
                    "steps": stp,
                    "epsilon": f"{eps:.6f}",
                },
            )

    def _make_synthetic_proof(self, plain_len: int, key_bits: int):
        base = 2048
        per_param = 12
        size = base + per_param * int(plain_len)
        payload = os.urandom(size)
        return payload, size

    def get_model_update(self):
        w_after = self.model.backbone.fc.weight.detach().view(-1).cpu()
        b_after = (
            self.model.backbone.fc.bias.detach().view(-1).cpu()
            if self.model.backbone.fc.bias is not None
            else None
        )

        if self._fc_init_w is None:
            base_w = torch.zeros_like(w_after)
            base_b = torch.zeros_like(b_after) if b_after is not None else None
        else:
            base_w = self._fc_init_w
            base_b = self._fc_init_b

        parts = [w_after - base_w]
        if b_after is not None:
            if base_b is None:
                base_b = torch.zeros_like(b_after)
            parts.append(b_after - base_b)

        vec = torch.cat(parts)

        if self.attack_type == "poison":
            print(f"[Cloud {self.cloud_id}] Client {self.cloudclient} ***** Poisoned attack applied *****")
            alpha = 0.5
            noise_sigma = 5.0
            vec = (1 - 2 * alpha) * vec + torch.randn_like(vec) * noise_sigma

        if self.attack_type == "sybil":
            print(f"[Cloud {self.cloud_id}] Client {self.cloudclient} ***** Sybil attack applied *****")
            torch.manual_seed(42)
            direction = torch.randn_like(vec)
            direction = direction / (direction.norm() + 1e-8)
            vec = 0.1 * direction + torch.randn_like(vec) * 0.005

        if torch.isnan(vec).any() or torch.isinf(vec).any():
            print(f"[Client {self.cid}] ⚠️ NaN or inf detected in vector! Replacing with safe values.")
            vec = torch.nan_to_num(vec, nan=0.0, posinf=1e3, neginf=-1e3)

        plain_vec = vec.cpu().numpy()

        if self.pubkey is not None:
            print(f"[Cloud {self.cloud_id}] Client {self.cloudclient} is encrypting model updates...")
            t0 = time.perf_counter()
            encrypted = [self.pubkey.encrypt(float(x)) for x in plain_vec.tolist()]
            t1 = time.perf_counter()
            enc_time_ms = (t1 - t0) * 1000.0
            try:
                cipher_bytes = sum(len(pickle.dumps(e, protocol=pickle.HIGHEST_PROTOCOL)) for e in encrypted)
            except Exception:
                cipher_bytes = -1
            key_bits = -1
            try:
                key_bits = int(self.pubkey.n.bit_length())
            except Exception:
                pass

            os.makedirs("logs/crypto", exist_ok=True)
            with open("logs/crypto/crypto_metrics.csv", "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if f.tell() == 0:
                    w.writerow(
                        [
                            "round",
                            "cloud",
                            "client_idx",
                            "client_id",
                            "key_bits",
                            "num_params",
                            "enc_time_ms",
                            "dec_time_ms",
                            "cipher_bytes",
                            "mode",
                        ]
                    )
                w.writerow([self.round_id, self.cloud_id, self.cloudclient, self.cid, key_bits, len(plain_vec), f"{enc_time_ms:.3f}", "", cipher_bytes, "client_encrypt"])

            _comm_append(
                {
                    "round": self.round_id,
                    "cloud": self.cloud_id,
                    "client_idx": self.cloudclient,
                    "client_id": self.cid,
                    "dir": "up",
                    "bytes": cipher_bytes,
                    "mode": "client_encrypt",
                }
            )

            print(f"[Cloud {self.cloud_id}] Client {self.cloudclient} successfully encrypted model weights.")

            if self.zkp_enabled:
                t0p = time.perf_counter()
                zkp_payload, zkp_size = self._make_synthetic_proof(len(plain_vec), key_bits)
                gen_ms = (time.perf_counter() - t0p) * 1000.0

                _zkp_append({"round": self.round_id, "cloud": self.cloud_id, "client_idx": self.cloudclient, "client_id": self.cid, "phase": "gen", "size_bytes": zkp_size, "time_ms": f"{gen_ms:.3f}", "accepted": ""})

                _comm_append({"round": self.round_id, "cloud": self.cloud_id, "client_idx": self.cloudclient, "client_id": self.cid, "dir": "up", "bytes": zkp_size, "mode": "zkp_proof"})

                return {"update": encrypted, "zkp": zkp_payload, "zkp_meta": {"size": zkp_size, "gen_ms": gen_ms, "key_bits": key_bits}}

            return encrypted

        plain_bytes = 4 * len(plain_vec)
        _comm_append({"round": self.round_id, "cloud": self.cloud_id, "client_idx": self.cloudclient, "client_id": self.cid, "dir": "up", "bytes": plain_bytes, "mode": "client_plain"})
        return plain_vec
