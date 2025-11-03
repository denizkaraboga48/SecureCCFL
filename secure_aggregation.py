import os
import pickle
import time
from typing import List, Sequence, Tuple, Union

import numpy as np
from phe import paillier
from phe.paillier import EncryptedNumber

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
KEY_DIR = os.path.join(_THIS_DIR, "keys")
PUBKEY_FILE = os.path.join(KEY_DIR, "public_key.pkl")
PRIVKEY_FILE = os.path.join(KEY_DIR, "private_key.pkl")
os.makedirs(KEY_DIR, exist_ok=True)

def generate_key_pair(pub_path: str = PUBKEY_FILE,
                      priv_path: str = PRIVKEY_FILE,
                      n_length: int = 1024) -> Tuple[paillier.PaillierPublicKey, paillier.PaillierPrivateKey]:
    os.makedirs(os.path.dirname(pub_path), exist_ok=True)
    os.makedirs(os.path.dirname(priv_path), exist_ok=True)
    public_key, private_key = paillier.generate_paillier_keypair(n_length=n_length)
    with open(pub_path, "wb") as f:
        pickle.dump(public_key, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(priv_path, "wb") as f:
        pickle.dump(private_key, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[Keys] Generated Paillier keys:\n  {os.path.abspath(pub_path)}\n  {os.path.abspath(priv_path)}")
    return public_key, private_key

def load_public_key(path: str = PUBKEY_FILE) -> paillier.PaillierPublicKey:
    if not os.path.exists(path):
        generate_key_pair(PUBKEY_FILE, PRIVKEY_FILE, n_length=1024)
    with open(path, "rb") as f:
        return pickle.load(f)

def load_private_key(path: str = PRIVKEY_FILE) -> paillier.PaillierPrivateKey:
    if not os.path.exists(path):
        generate_key_pair(PUBKEY_FILE, PRIVKEY_FILE, n_length=1024)
    with open(path, "rb") as f:
        return pickle.load(f)

def get_public_key_bits(pubkey) -> int:
    try:
        return int(pubkey.n.bit_length())
    except Exception:
        return -1

def _flatten_vector(vec: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    return np.asarray(vec, dtype=float).reshape(-1)

def encrypt_vector(pubkey: paillier.PaillierPublicKey,
                   vector: Union[Sequence[float], np.ndarray]) -> List[EncryptedNumber]:
    flat = _flatten_vector(vector)
    return [pubkey.encrypt(float(x)) for x in flat]

def decrypt_vector(encrypted: Union[List[EncryptedNumber], np.ndarray, EncryptedNumber, List[List[EncryptedNumber]]],
                   privkey: paillier.PaillierPrivateKey) -> List[float]:
    if isinstance(encrypted, list) and encrypted and isinstance(encrypted[0], list):
        out: List[float] = []
        for block in encrypted:
            out.extend(float(privkey.decrypt(x)) for x in block)
        return out
    if isinstance(encrypted, (list, np.ndarray)):
        return [float(privkey.decrypt(x)) for x in encrypted]
    if isinstance(encrypted, EncryptedNumber):
        return [float(privkey.decrypt(encrypted))]
    raise TypeError("decrypt_vector: unsupported input type.")

def aggregate_encrypted(list_of_encrypted_vectors: List[List[EncryptedNumber]]) -> List[EncryptedNumber]:
    if not list_of_encrypted_vectors:
        return []
    L = len(list_of_encrypted_vectors[0])
    for i, v in enumerate(list_of_encrypted_vectors):
        if len(v) != L:
            raise ValueError(f"Encrypted vector length mismatch at client {i}: {len(v)} vs {L}")
    agg = []
    for j in range(L):
        s = list_of_encrypted_vectors[0][j]
        for i in range(1, len(list_of_encrypted_vectors)):
            s = s + list_of_encrypted_vectors[i][j]
        agg.append(s)
    return agg

def decrypt_and_sum_vectors(list_of_encrypted_vectors: List[List[EncryptedNumber]],
                            privkey: paillier.PaillierPrivateKey) -> List[float]:
    if not list_of_encrypted_vectors:
        return []
    summed_cipher = aggregate_encrypted(list_of_encrypted_vectors)
    return [float(privkey.decrypt(x)) for x in summed_cipher]

def encrypt_vector_with_timing(pubkey: paillier.PaillierPublicKey,
                               vec: Union[Sequence[float], np.ndarray]):
    flat = _flatten_vector(vec)
    t0 = time.perf_counter()
    enc_list = [pubkey.encrypt(float(x)) for x in flat]
    t1 = time.perf_counter()
    enc_time_ms = (t1 - t0) * 1000.0
    try:
        cipher_bytes = sum(len(pickle.dumps(e, protocol=pickle.HIGHEST_PROTOCOL)) for e in enc_list)
    except Exception:
        cipher_bytes = -1
    return enc_list, enc_time_ms, cipher_bytes

def decrypt_vector_with_timing(privkey: paillier.PaillierPrivateKey,
                               enc_list: List[EncryptedNumber]):
    t0 = time.perf_counter()
    plain = [privkey.decrypt(e) for e in enc_list]
    t1 = time.perf_counter()
    dec_time_ms = (t1 - t0) * 1000.0
    return np.array(plain, dtype=float), dec_time_ms
