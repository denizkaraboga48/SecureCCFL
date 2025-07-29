# secure_aggregation.py
from phe import paillier
import pickle
import os

KEY_DIR = "keys"
PUBKEY_FILE = os.path.join(KEY_DIR, "paillier_pubkey.pkl")
PRIVKEY_FILE = os.path.join(KEY_DIR, "paillier_privkey.pkl")

def generate_key_pair():
    if not os.path.exists(KEY_DIR):
        os.makedirs(KEY_DIR)
    if not os.path.exists(PUBKEY_FILE) or not os.path.exists(PRIVKEY_FILE):
        pubkey, privkey = paillier.generate_paillier_keypair()
        with open(PUBKEY_FILE, "wb") as f:
            pickle.dump(pubkey, f)
        with open(PRIVKEY_FILE, "wb") as f:
            pickle.dump(privkey, f)

def load_public_key():
    with open(PUBKEY_FILE, "rb") as f:
        return pickle.load(f)

def load_private_key():
    with open(PRIVKEY_FILE, "rb") as f:
        return pickle.load(f)

def aggregate_encrypted(updates):
    """updates: List of List of encrypted weights (same shape)"""
    aggregated = []
    for param_i in range(len(updates[0])):
        summed = updates[0][param_i]
        for u in updates[1:]:
            summed = [a + b for a, b in zip(summed, u[param_i])]
        aggregated.append(summed)
    return aggregated

def decrypt_model_update(encrypted_update, privkey):
    """Decrypts aggregated encrypted model"""
    decrypted = []
    for param in encrypted_update:
        param_dec = [privkey.decrypt(x) for x in param]
        decrypted.append(param_dec)
    return decrypted
