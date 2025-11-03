# Secure Cross-Cloud Federated Learning (Secure-CCFL)

> **Research-grade simulation framework** for secure federated learning across multiple cloud providers with privacy, encryption, anomaly detection, and auditability.

---

## Overview

This repository implements a full-stack **Secure Cross-Cloud Federated Learning** simulation, integrating:

- Differential Privacy (Opacus)
- Homomorphic Encryption (Paillier)
- Sybil Detection (similarity + reputation)
- Model Poisoning 
- Backdoor Attacks
- Synthetic ZKP payloads + verification timing
- Secure aggregation
- Blockchain-style off-chain audit logging
- Non-IID client data distribution across clouds

This is designed for **academic research on Secure Cross-Cloud Federated Learning (Secure-CCFL).*.

---

## Architecture

```
Clients → Cloud Coordinators → Federated Server
```

## Project Structure

```
.
├── run_simulation.py        
├── cloud.py                 
├── client.py                
├── server.py                
├── secure_aggregation.py    
├── model.py                 
├── data_utils.py  
├── plot_security.py           
└── logs/                    
```

---

## Requirements

Python ≥ 3.10

```
pip install torch torchvision opacus phe numpy pandas scikit-learn
```

or:

```
pip install -r requirements.txt
```

---

## Running Experiments

### Default (Secure-CCFL)

```
python run_simulation.py
```

### Modes

| Mode | Command | Description |
|---|---|---|
FL | `--mode fl` | Single-cloud, no security  
CCFL | `--mode ccfl` | Multi-cloud, no security  
Secure-CCFL | `--mode secure` | Multi-cloud + all defenses 

Example:

```
python run_simulation.py --mode secure --rounds 5 --clouds 3 --clients_per_cloud 5
```

---

## Outputs & Metrics

All results logged in `logs/`:

| Category | File |
|---|---|
Accuracy & ASR | `logs/extended_eval.csv`  
DP privacy ε | `logs/dp/`  
Crypto time & bytes | `logs/crypto/`  
Communication cost | `logs/comm/`  
Sybil scores & reputation | `logs/sybil_similarity_reputation.csv`  
Poison detection norms | `logs/poisoning_norms.csv`  
ZKP proof timing | `logs/zkp/`  
Chain audit log | `logs/chain/`  

---


## Citation

```
Karaboğa, D. Secure Cross-Cloud Federated Learning Simulation Framework, 2025.
```

---

## Author

**Deniz Karaboğa**  
denizkaraboga@gmail.com
