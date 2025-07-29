import torch
import torch.nn as nn
import torch.optim as optim
from dp_utils import apply_opacus
from model import TinyCNN
import numpy as np
from phe import paillier
import os

class Client:

    
    def __init__(self, cid, dataloader, attack_type=None, pubkey=None, cloud_id=None):
        self.cid = cid
        self.model = TinyCNN()
        self.dataloader = dataloader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.attack_type = attack_type  # None, 'sybil', 'poison'
        self.pubkey = pubkey
        self.cloud_id = cloud_id 
        self.cloudclient=((self.cid)%5)+1    
    def load_global_model(self, path="models/global_model.npy"):
        if os.path.exists(path):
            print(f"[Cloud {self.cloud_id}] Client {self.cloudclient} is loading global model...")
            flat_weights = np.load(path)
            offset = 0
            for param in self.model.parameters():
                size = param.data.numel()
                values = flat_weights[offset:offset + size]
                param.data = torch.tensor(values, dtype=param.data.dtype).view_as(param.data)
                offset += size
            print(f"[Cloud {self.cloud_id}] Client {self.cloudclient} successfully loaded global model.")
        else:
            print(f"[Cloud {self.cloud_id}] Client {self.cloudclient} could not find global model at {path}")

    def train(self, epochs=1):
        self.load_global_model()  # Global modeli yükle
        self.model.train()
        self.model, self.optimizer, self.dataloader = apply_opacus(
            self.model, self.optimizer, self.dataloader,
            client_id=self.cid, cloud_id=self.cloud_id
        )
        for _ in range(epochs):
            for data, target in self.dataloader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

    def get_model_update(self):
        if self.attack_type == 'poison':
            print(f"[Cloud {self.cloud_id}] Client {self.cloudclient} ***** Poisoned attack applied *****")
            for param in self.model.parameters():
                param.data = -param.data + torch.randn_like(param.data) * 30
                param.data = torch.nan_to_num(param.data, nan=0.0, posinf=1.0, neginf=-1.0)

        if self.attack_type == 'sybil':
            print(f"[Cloud {self.cloud_id}] Client {self.cloudclient} ***** Sybil attack applied *****")
        weights = []
        print(f"[Cloud {self.cloud_id}] Client {self.cloudclient} is encrypting model updates...")
        for param in self.model.parameters():
            tensor = param.data.view(-1)

            if self.attack_type == 'sybil':
               tensor = torch.ones_like(tensor) * 0.333 + torch.randn_like(tensor) * 0.01

            # NaN/Inf kontrolü
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"[Client {self.cid}] ⚠️ NaN or inf detected in tensor! Replacing with safe values.")
                tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e3, neginf=-1e3)

            encrypted = [self.pubkey.encrypt(float(x)) for x in tensor.tolist()]
            weights.append(encrypted)

        print(f"[Cloud {self.cloud_id}] Client {self.cloudclient} successfully encrypted model weights.")
        return weights
