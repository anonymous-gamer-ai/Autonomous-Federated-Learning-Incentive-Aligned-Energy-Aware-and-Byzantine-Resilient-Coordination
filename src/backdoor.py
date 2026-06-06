import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
# ----------------------------
# 1. Define Model Architecture
# (Must match Client architecture exactly)
# ----------------------------
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Matches client keys "c1" and "c2"
        # Matches input 28x28 -> yielding 9216 flattened features
        self.c1 = nn.Conv2d(1, 32, 3, 1)  # 28x28 -> 26x26
        self.c2 = nn.Conv2d(32, 64, 3, 1) # 26x26 -> 24x24
        
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
        
        x = F.max_pool2d(x, 2) # 24x24 -> 12x12
        x = self.dropout1(x)
        
        x = torch.flatten(x, 1) # Flatten 64*12*12 = 9216
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
import os
TEST_LOADER = None
TEST_DATA_PATH = "./data/testing" # Updated path

def get_test_loader():
    global TEST_LOADER
    if TEST_LOADER is None:
        print(f"[Backdoor] Loading Validation Set from {TEST_DATA_PATH}...")
        
        # MUST match Client transform exactly (1 channel, 28x28)
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), 
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])
        
        if not os.path.exists(TEST_DATA_PATH):
            print(f"[Backdoor] CRITICAL ERROR: Test path '{TEST_DATA_PATH}' not found!")
            return None

        # ImageFolder expects subfolders for each class (e.g., ./data/testing/0, ./data/testing/1)
        dataset = datasets.ImageFolder(root=TEST_DATA_PATH, transform=transform)
        
        # We use a smaller batch size for validation to reduce memory spike
        TEST_LOADER = DataLoader(dataset, batch_size=1000, shuffle=False)
        print(f"[Backdoor] Validation Data Ready: {len(dataset)} images.")
        
    return TEST_LOADER

def evaluate_model(weights):
    """
    Loads weights into a temp model and tests against Coordinator's data.
    """
    device = torch.device("cpu") # Coordinator usually runs on CPU
    model = SimpleCNN().to(device)
    model.load_state_dict(weights)
    model.eval()
    
    test_loader = get_test_loader()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
    return correct / total

# ----------------------------
# 3. Main Detection Logic
# ----------------------------
import torch
import numpy as np
from collections import defaultdict

import torch
import numpy as np
from collections import defaultdict

# ----------------------------
# Persist across rounds
# ----------------------------
HISTORY = defaultdict(list)


def state_dict_to_vector(state_dict):
    """
    Converts a model state_dict to a single flattened tensor.
    """
    return torch.cat([
        param.view(-1)
        for param in state_dict.values()
        if torch.is_tensor(param)
    ])


def detect_backdoor(models, global_weights, round_id):
    """
    Robust Byzantine / Backdoor detector using:
    1. Real accuracy audit
    2. Median + MAD thresholding
    3. Update distance
    4. Cosine alignment
    5. Temporal consistency
    """

    print(f"[Coordinator] Auditing {len(models)} models (Round {round_id})")

    accs, dists, cosines = [], [], []

    # Convert global model ONCE
    w_g = state_dict_to_vector(global_weights)

    # ----------------------------
    # Step 1: Audit models
    # ----------------------------
    for m in models:
        node_id = m["node_id"]

        # ---- Accuracy (cannot be lied about)
        acc = evaluate_model(m["weights"])
        m["accuracy"] = acc
        accs.append(acc)

        # ---- Vectorize weights
        w_i = state_dict_to_vector(m["weights"])
        m["w_vec"] = w_i

        # ---- Update
        delta = w_i - w_g

        # ---- Distance
        dist = torch.norm(delta).item()
        dists.append(dist)

        # ---- Cosine similarity
        cos = torch.nn.functional.cosine_similarity(
            delta, w_g, dim=0
        ).item()
        cosines.append(cos)

        # ---- Temporal history
        HISTORY[node_id].append(acc)

        print(
            f" > Node {node_id} | "
            f"Acc={acc:.3f} | Dist={dist:.3f} | Cos={cos:.3f}"
        )

    accs = np.array(accs)
    dists = np.array(dists)
    cosines = np.array(cosines)

    # ----------------------------
    # Step 2: Robust thresholds
    # ----------------------------
    def robust_bounds(x, k=2.5, upper=True):
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-6
        return med + k * mad if upper else med - k * mad

    acc_th  = robust_bounds(accs, upper=False)
    dist_th = robust_bounds(dists, upper=True)
    cos_th  = robust_bounds(cosines, upper=False)

    print(
        f"[Thresholds] Acc<{acc_th:.3f} | "
        f"Dist>{dist_th:.3f} | Cos<{cos_th:.3f}"
    )

    # ----------------------------
    # Step 3: Detect Byzantine nodes
    # ----------------------------
    malicious = []

    for m in models:
        node_id = m["node_id"]
        acc = m["accuracy"]

        delta = m["w_vec"] - w_g
        dist = torch.norm(delta).item()
        cos = torch.nn.functional.cosine_similarity(
            delta, w_g, dim=0
        ).item()

        # ---- Temporal instability
        hist = HISTORY[node_id]
        unstable = len(hist) >= 3 and np.std(hist[-3:]) > 0.10

        if (
            acc < acc_th or
            dist > dist_th or
            cos < cos_th or
            unstable
        ):
            malicious.append(node_id)
            print(f"[DETECTED] Byzantine Node: {node_id}")

    return malicious