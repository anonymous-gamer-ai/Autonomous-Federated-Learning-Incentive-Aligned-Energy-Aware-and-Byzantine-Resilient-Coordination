# client.py
import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
from prom_metrics import PROM_ENERGY, PROM_BALANCE, PROM_MALICIOUS

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

class AgenticClient:
    def __init__(self, client_id, is_malicious=False):
        self.id = client_id
        self.model = SimpleModel()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.is_malicious = is_malicious
        self.reputation = 1.0  # Starts perfect
        self.coin_balance = 0.0
        
        # Report status to Prometheus immediately
        PROM_MALICIOUS.labels(client_id=self.id).set(1 if self.is_malicious else 0)

    def get_metadata(self):
        """Returns the state for the LLM to analyze"""
        return {
            "id": self.id,
            "malicious": self.is_malicious,
            "reputation": self.reputation,
            "energy": random.uniform(10.0, 100.0), # Simulated historical energy usage
            "latency": random.uniform(0.1, 0.5)
        }

    def train(self, global_weights):
        start_time = time.time()
        
        # Load Global Weights
        self.model.load_state_dict(global_weights)
        
        # Simulate Training
        # If malicious, we might poison the weights (simplified here)
        loss = 0.0
        for _ in range(5): # Local Epochs
            inputs = torch.randn(10)
            target = torch.randn(1)
            self.optimizer.zero_grad()
            output = self.model(inputs)
            loss = nn.MSELoss()(output, target)
            loss.backward()
            
            # ATTACK VECTOR: Gradient Ascent instead of Descent
            if self.is_malicious:
                for param in self.model.parameters():
                    param.grad.data *= -10.0 # Poisoning
            
            self.optimizer.step()
            
        # Green Metric Calculation
        duration = time.time() - start_time
        energy_cost = duration * 50.0  # 50 Watts
        
        # Update Prometheus
        PROM_ENERGY.labels(client_id=self.id).set(energy_cost)
        
        return self.model.state_dict(), loss.item(), energy_cost
        
    def receive_reward(self, amount):
        self.coin_balance += amount
        PROM_BALANCE.labels(client_id=self.id).set(self.coin_balance)