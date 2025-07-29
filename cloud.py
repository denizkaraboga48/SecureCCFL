import random
from client import Client
from data_utils import get_mnist_dataset, split_non_iid
from secure_aggregation import load_public_key
import numpy as np

def simulate_cloud(cloud_id, start_client_id=0, attack_map=None):

    print(f"[Cloud {cloud_id}] Starting")
    print()
    train_dataset, _ = get_mnist_dataset()
    loaders = split_non_iid(train_dataset, num_clients=5)

    pubkey = load_public_key()
    clients = []

    for i in range(5):
        cid = start_client_id + i
        attack_type = attack_map.get(cid, None) if attack_map else None
        c = Client(cid=cid, dataloader=loaders[i], attack_type=attack_type, pubkey=pubkey, cloud_id=cloud_id)
        clients.append(c)

    client_updates = []
    for c in clients:
        print(f"[Cloud {cloud_id}] Client {((c.cid)%5)+1} training")
        c.train(epochs=1)
        encrypted_update = c.get_model_update()
        client_updates.append(encrypted_update)
        print()

    print(f"[Cloud {cloud_id}] sending Model updates to server")
    print()
    return client_updates
