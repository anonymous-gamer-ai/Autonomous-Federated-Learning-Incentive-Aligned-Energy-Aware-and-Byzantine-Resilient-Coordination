# main.py
import time
import random
from config import NUM_CLIENTS, TOTAL_ROUNDS
from client import AgenticClient
from coordinator import Coordinator
from prom_metrics import start_metrics_server

def setup_environment():
    print("Initializing Agentic-BFL Ecosystem...")
    
    # 1. Start Prometheus Server
    start_metrics_server()
    
    # 2. Create Agents
    # We will purposely make Agent 3 and 7 malicious to test the LLM
    clients = {}
    for i in range(1, NUM_CLIENTS + 1):
        is_bad = True if i in [3, 7] else False
        clients[i] = AgenticClient(client_id=i, is_malicious=is_bad)
        status = "Malicious" if is_bad else "Honest"
        print(f"   Created Agent {i} ({status})")
        
    # 3. Create Coordinator
    coordinator = Coordinator(clients)
    return coordinator

def run_simulation():
    system = setup_environment()
    
    # Run Rounds
    for r in range(TOTAL_ROUNDS):
        system.start_round()
        time.sleep(2) # Pause for readability
        
    print("\nSimulation Complete. Check metrics at http://localhost:8000")

if __name__ == "__main__":
    run_simulation()