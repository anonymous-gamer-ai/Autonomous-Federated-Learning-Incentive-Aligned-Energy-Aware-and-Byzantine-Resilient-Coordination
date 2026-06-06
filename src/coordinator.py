import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import base64
import io
import sys
import os  # Added for path checking
import random # Added for sampling
from kafka import KafkaConsumer, KafkaProducer
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from prometheus_client import start_http_server, Gauge

# ----------------------------
# Configuration
# ----------------------------
NODE_ID = "client-1"
BROKER = "localhost:9092"
UPDATE_TOPIC = "fl.updates"
CONTROL_TOPIC = "fl.control"
TRAIN_PATH = "./data/training" # Update this to your actual data path
BATCH_SIZE = 32

# Prometheus Metrics
PROM_PORT = 8001 
acc_gauge = Gauge('client_accuracy', 'Model Accuracy', ['client_id'])

# ----------------------------
# 1. Safe JSON Deserializer
# ----------------------------
def safe_deserialize(m):
    try:
        if m is None: return None
        decoded = m.decode('utf-8')
        if not decoded: return None 
        return json.loads(decoded)
    except json.JSONDecodeError:
        return None 
    except Exception as e:
        print(f"[{NODE_ID}] Deserialization error: {e}")
        return None

# ----------------------------
# 2. Model Definition
# ----------------------------
# UPDATED: Matches the Coordinator's expected architecture (c1, c2, 9216 fc1)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Matches keys "c1" and "c2" from your error log
        self.c1 = nn.Conv2d(1, 32, 3, 1)  
        self.c2 = nn.Conv2d(32, 64, 3, 1) 
        
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # 9216 comes from: 64 channels * 12 * 12 (after pooling)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 100)

    def forward(self, x):
        x = self.c1(x)
        x = F.relu(x)
        x = self.c2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# ----------------------------
# 3. Data Loading (Updated with your snippet)
# ----------------------------
def load_data():
    print(f"[{NODE_ID}] Loading local dataset from {TRAIN_PATH}...")
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    # Ensure data path exists or handle error gracefully
    if not os.path.exists(TRAIN_PATH):
        print(f"[{NODE_ID}] ERROR: Training path '{TRAIN_PATH}' not found!")
        # Return an empty loader or handle exit to prevent crash
        sys.exit(1)
    else:
        full_dataset = datasets.ImageFolder(root=TRAIN_PATH, transform=transform)

        # Limit to 50 images per class for simulation speed
        class_indices = {}
        for idx, (_, label) in enumerate(full_dataset):
            class_indices.setdefault(label, []).append(idx)

        #limited_indices = []
       # for indices in class_indices.values():
          #  limited_indices.extend(random.sample(indices, min(50, len(indices))))

        train_dataset = full_dataset
        loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        print(f"[{NODE_ID}] Dataset ready with {len(train_dataset)} images.")
        return loader

# ----------------------------
# 4. Training Function
# ----------------------------
def train(model, device, train_loader, epochs=1):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
    duration = time.time() - start_time
    energy_joules = duration * 50.0 
    return energy_joules

# ----------------------------
# Main Logic
# ----------------------------
def main():
    print(f"[{NODE_ID}] Metrics server started on port {PROM_PORT}")
    start_http_server(PROM_PORT)

    device = torch.device("cpu")
    model = SimpleCNN().to(device)
    
    # Load data using the updated function
    train_loader = load_data()

    consumer = KafkaConsumer(
        CONTROL_TOPIC,
        bootstrap_servers=BROKER,
        value_deserializer=safe_deserialize,
        group_id=NODE_ID,
        auto_offset_reset='latest'
    )
    
    producer = KafkaProducer(
        bootstrap_servers=BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        max_request_size=1073741824
    )

    print(f"[{NODE_ID}] Listening for 'START_ROUND' on {CONTROL_TOPIC}...")

    for msg in consumer:
        try:
            data = msg.value
            if data is None: continue
                
            msg_type = data.get("type")

            #if msg_type == "TRAINING_COMPLETE":
             #   print(f"[{NODE_ID}] Training Complete! Final Global Acc: {data.get('final_acc')}")
             #   sys.exit(0)

            if msg_type == "START_ROUND":
                round_num = data.get("round")
                print(f"\n[{NODE_ID}] --- Starting Round {round_num} ---")

                if "weights_b64" in data:
                    b64_str = data["weights_b64"]
                    bytes_data = base64.b64decode(b64_str)
                    buffer = io.BytesIO(bytes_data)
                    global_weights = torch.load(buffer)
                    # This will now succeed because SimpleCNN matches the weights
                    model.load_state_dict(global_weights)
                    print(f"[{NODE_ID}] Global weights loaded.")

                energy = train(model, device, train_loader, epochs=1)
                
                model.eval()
                correct = 0
                with torch.no_grad():
                    for d, t in train_loader:
                        out = model(d)
                        pred = out.argmax(dim=1, keepdim=True)
                        correct += pred.eq(t.view_as(pred)).sum().item()
                acc = correct / len(train_loader.dataset)
                acc_gauge.labels(client_id=NODE_ID).set(acc)

                print(f"[{NODE_ID}] Training finished. Acc: {acc:.2f}, Energy: {energy:.2f}J")

                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                weights_b64 = base64.b64encode(buffer.read()).decode('utf-8')

                update_msg = {
                    "node_id": NODE_ID,
                    "dataset": "EuroSAT-Shard", 
                    "samples": len(train_loader.dataset),
                    "accuracy": acc,
                    "energy_j": energy,
                    "weights_b64": weights_b64
                }
                
                producer.send(UPDATE_TOPIC, update_msg)
                producer.flush()
                print(f"[{NODE_ID}] Update sent to Coordinator.")

        except Exception as e:
            print(f"[{NODE_ID}] Error in loop: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()