# run_simulation.py
from secure_aggregation import generate_key_pair
from cloud import simulate_cloud
from server import process_round
import time
import os

def run_simulation(rounds=1):
    print("[Secure CCFL Simulation Starting...")
    
    # Key generate (Paillier)
    generate_key_pair()

    attack_map = {
        2: 'sybil',     # Cloud A, Client 3
        6: 'poison',    # Cloud B, Client 2
        8: 'sybil',     # Cloud B, Client 4
        12: 'poison',   # Cloud C, Client 3
    }

    for rnd in range(1, rounds + 1):
        print(f"\n=========== ROUND {rnd}/{rounds} ===========")
        print()
        updates_A = simulate_cloud('A', start_client_id=0, attack_map=attack_map)
        updates_B = simulate_cloud('B', start_client_id=5, attack_map=attack_map)
        updates_C = simulate_cloud('C', start_client_id=10, attack_map=attack_map)

        all_updates = updates_A + updates_B + updates_C

        process_round(all_updates)

        time.sleep(2)

    print("\n[Simulation] All rounds completed.")

if __name__ == "__main__":
    run_simulation(rounds=1)
