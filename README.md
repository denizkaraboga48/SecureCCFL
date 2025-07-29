# Secure Cross-Cloud Federated Learning (CCFL) Simulation

This project implements a simplified, security-focused simulation of **Cross-Cloud Federated Learning (CCFL)**. The system includes layered defense mechanisms against Sybil and Poisoning attacks, along with Differential Privacy and Homomorphic Encryption for secure aggregation.

## Features

- 3 independent cloud regions (Cloud A, B, and C)
- 5 clients per cloud (15 total)
- Simulation of Sybil and Poisoning attacks
- Differential Privacy using Opacus
- Homomorphic Encryption using Paillier scheme
- Secure model aggregation and update
- Logging of all outputs to `.csv` and `.npy` files

## Dataset

The simulation uses the **MNIST** dataset as a sample input. The data is partitioned in a non-IID fashion, meaning each client receives a subset of the classes to simulate realistic heterogeneous distributions across clouds.

## Requirements

- Python 3.10+
- PyTorch
- Opacus
- PHE (`python-paillier`)
- NumPy, Pandas, Matplotlib, scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

To start the simulation:

```bash
python run_simulation.py
```

## Example Output

```text
Secure CCFL Simulation Starting...

=========== ROUND 1/3 ===========

[Cloud A] Starting

[Cloud A] Client 1 training
[Cloud A] Client 1 is loading global model...
[Cloud A] Client 1 successfully loaded global model.
[Cloud A] Client 1 Differential Privacy completed.
[Cloud A] Client 1 is encrypting model updates...
[Cloud A] Client 1 successfully encrypted model weights.
...
[Cloud A] Client 2 training
...
[Cloud A] Client 3 ***** Sybil attack applied *****
...

[Cloud A] sending Model updates to server

....
[Cloud B] Client 2 ***** Poisoned attack applied *****
...

Federated server starting
[Server] Decrypting individual client updates for detection
[Server] Running Poisoning detection
[Poisoning Detection] Poisoned clients detected:
  →  [Cloud B] Client 2
  →  [Cloud C] Client 3
[Server] Running Sybil detection
[Sybil Detection] Sybil clients detected:
  →  [Cloud A] Client 3
  →  [Cloud B] Client 4
[Server] Starting secure aggregation
[Server] Decrypting aggregated model
[Server] Global model updated and saved to models\global_model.npy
```

## Logs

- `logs/`: Contains Sybil similarity matrices, poisoning detection results, and communication overhead stats
- `models/`: Stores updated global model files (`.npy`)


## Contact

**Deniz Karaboğa**  
Email: denizkaraboga@gmail.com
