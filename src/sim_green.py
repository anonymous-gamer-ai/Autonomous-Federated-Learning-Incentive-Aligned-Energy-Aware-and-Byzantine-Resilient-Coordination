import numpy as np

# -------------------------
# Configuration
# -------------------------
num_clients = 5
communication_rounds = 10
Tref = 70  # reference temperature in °C
beta = 0.05  # thermal sensitivity
gamma = 5  # security penalty
alpha = 0.1  # trust memory decay
lambda1, lambda2, lambda3 = 0.5, 0.5, 1.0
K = 3  # number of clients to select per round
delta = 0.1  # minimum confidence threshold

# -------------------------
# Client State
# -------------------------
clients = {}
for i in range(num_clients):
    clients[i] = {
        "D": np.random.uniform(500, 1000),  # MB of data processed
        "E": np.random.uniform(100, 300),   # energy consumed (J)
        "T": np.random.uniform(60, 90),     # average temperature
        "acc": np.random.uniform(0.5, 0.9), # initial accuracy
        "malicious": np.random.rand() < 0.2, # some clients may be malicious
        "trust": 0.5                         # initial trust
    }

# -------------------------
# Functions
# -------------------------
def client_efficiency(D, E, T, Tref=Tref, beta=beta):
    """Compute energy-aware client efficiency"""
    return (D / E) * np.exp(-beta * (T - Tref))

def client_reward(delta_acc, eta, lambda1=lambda1, lambda2=lambda2):
    """Compute client reward"""
    eta_norm = eta / max(eta, 1e-8)  # normalize
    return lambda1 * delta_acc + lambda2 * eta_norm

def confidence_score(trust, delta_acc, eta, malicious_flag, gamma=gamma):
    """Compute confidence score for client selection"""
    sigma = lambda x: 1 / (1 + np.exp(-x))
    return trust * sigma(delta_acc) * sigma(eta) * np.exp(-gamma * malicious_flag)

def update_trust(trust, delta_acc, eta, malicious_flag, alpha=alpha,
                 lambda1=lambda1, lambda2=lambda2, lambda3=lambda3):
    """Update trust ledger"""
    sigma = lambda x: 1 / (1 + np.exp(-x))
    utility = lambda1*sigma(delta_acc) + lambda2*sigma(eta) - lambda3*malicious_flag
    return (1 - alpha)*trust + alpha*utility

# -------------------------
# Federated Learning Simulation
# -------------------------
for t in range(communication_rounds):
    print(f"\n--- Round {t+1} ---")
    
    # Compute client efficiencies
    for i in clients:
        c = clients[i]
        c["eta"] = client_efficiency(c["D"], c["E"], c["T"])
        c["delta_acc"] = c["acc"] - np.mean([cl["acc"] for cl in clients.values()])
    
    # Compute confidence scores
    for i in clients:
        c = clients[i]
        c["confidence"] = confidence_score(c["trust"], c["delta_acc"], c["eta"], int(c["malicious"]))
    
    # Select top-K clients above threshold
    selected = [i for i, c in sorted(clients.items(), key=lambda x: x[1]["confidence"], reverse=True)
                if c["confidence"] >= delta][:K]
    print("Selected clients:", selected)
    
    # Compute rewards and update trust
    for i in clients:
        c = clients[i]
        c["reward"] = client_reward(c["delta_acc"], c["eta"])
        c["trust"] = update_trust(c["trust"], c["delta_acc"], c["eta"], int(c["malicious"]))
        # Simulate small random accuracy update
        c["acc"] = min(1.0, c["acc"] + np.random.uniform(-0.01, 0.02))
    
    # Print summary
    for i in clients:
        c = clients[i]
        print(f"Client {i}: acc={c['acc']:.3f}, eta={c['eta']:.3f}, trust={c['trust']:.3f}, reward={c['reward']:.3f}")
