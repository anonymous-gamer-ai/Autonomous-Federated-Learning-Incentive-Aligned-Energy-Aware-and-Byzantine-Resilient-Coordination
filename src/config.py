# config.py
import os

# Simulation Settings
NUM_CLIENTS = 4
TOTAL_ROUNDS = 5
MIN_CLIENTS_PER_ROUND = 3

# LLM Settings
# If you have a key, set it here or in environment variables. 
# If None, the system uses a deterministic fallback (simulated LLM).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None) 
LLM_MODEL = "gpt-4"

# Federated Learning Settings
GLOBAL_EPOCHS = 1
LOCAL_EPOCHS = 3
LEARNING_RATE = 0.01

# Green Metric Settings
ALPHA_REWARD = 10.0
POWER_CONSUMPTION_W = 50.0