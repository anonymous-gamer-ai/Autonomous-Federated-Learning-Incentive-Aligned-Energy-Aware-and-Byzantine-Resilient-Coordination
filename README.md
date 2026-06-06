<p align="center">
  <h1 align="center"> Autonomous Federated Learning (AFL)</h1>
  <p align="center">
    <strong>Incentive-Aligned · Energy-Aware · Byzantine-Resilient</strong>
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#modules">Modules</a> •
    <a href="#deployment">Deployment</a> •
    <a href="#citation">Citation</a>
  </p>
</p>

---

## Overview

Federated Learning (FL) assumes clients are cooperative, honest, and energy-agnostic — assumptions that rarely hold in real deployments where participants own private data, operate under tight energy budgets, and face misaligned incentives.

**Autonomous Federated Learning (AFL)** is a research framework that closes this gap. Each client runs an autonomous agent that solves a local constrained decision problem to jointly optimize participation, update effort, and verification actions under explicit energy limits. The system introduces three core innovations:

| Pillar | Mechanism | What It Does |
|---|---|---|
| **Incentive Alignment** | Proof-of-Useful Learning (PoUL) | Rewards clients in proportion to *verifiable utility gain per unit energy*, assessed via committee-based validation on a shared reference set. |
| **Byzantine Resilience** | Multi-signal Backdoor Detector | Flags adversarial clients through accuracy audits, update-distance analysis, cosine alignment checks, and temporal consistency tracking. |
| **Energy Awareness** | Hardware-Aware Green Metrics | Monitors real CPU/GPU power draw via LibreHardwareMonitor and integrates energy cost into the trust and selection pipeline. |

A persistent **Trust Ledger** (backed by PostgreSQL) encodes each client's contribution history and efficiency profile, ensuring that only reliable, energy-efficient participants accumulate long-term influence.

### Key Results

> Across CIFAR-10, FMNIST, and MNIST under heterogeneous and adversarial settings with **30% Byzantine clients**:

| Metric | AFL | Best Baseline (Fed-NGA) |
|---|---|---|
| MNIST accuracy degradation | **3.93%** | 13.52% |
| CIFAR-10 accuracy degradation | **20.17%** | 46.48% |
| Adversarial confidence score | **≤ 0.15** | — |
| Benign-client mean confidence | **0.75** | — |
| Utility-per-Joule improvement | **15–35%** over random selection | — |

Methods such as CClip, MCA, and Geometric Median collapse to near-chance accuracy under structured attacks.

---

## Architecture

```
┌─────────────────┐      kafka:9092      ┌────────────┐
│   FL Clients    │◄────────────────────►│   Kafka    │◄── zookeeper:2181 ── Zookeeper
│ (N replicas)    │                      │            │
└───────┬─────────┘                      └─────┬──────┘
        │ :8003 metrics                        │
        │                                      │
        │                                ┌─────▼──────┐     postgres:5432    ┌──────────┐
        │                                │Coordinator │────────────────────►│ Postgres │
        │                                │            │                     │          │
        │                                └──────┬─────┘                     └──────────┘
        │                                       │ :8000 metrics
        ▼                                       ▼
   ┌─────────────────────────────────────────────────┐
   │           Prometheus  (:9090 / :30090)          │
   └────────────────────┬────────────────────────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Grafana (:3000 / :30000) │
              └───────────────────┘
```

**Communication flow:**

1. The **Coordinator** broadcasts a `START_ROUND` message (with optional global weights) to the `fl.control` Kafka topic.
2. Each **Client** agent trains locally, computes accuracy and energy cost, then publishes its model update to the `fl.updates` topic.
3. The Coordinator collects updates, runs the **Backdoor Detector** to flag Byzantine nodes, performs **Federated Averaging** over clean updates, distributes **PoUL rewards**, and updates the **Trust Ledger** in PostgreSQL.
4. **Prometheus** scrapes metrics from both coordinator and clients; **Grafana** visualizes them in real time.

---

## Modules

### Source (`src/`)

| File | Purpose |
|---|---|
| [`main.py`](src/main.py) | Simulation entry point — initializes clients (with configurable malicious agents), creates the coordinator, and runs FL rounds. |
| [`coordinator.py`](src/coordinator.py) | Core FL coordinator — listens on Kafka for client updates, broadcasts global model, manages round lifecycle, and exposes Prometheus metrics. Contains the `SimpleCNN` model definition (Conv2d → MaxPool → FC, 100-class output). |
| [`client.py`](src/client.py) | `AgenticClient` class — wraps a local model, simulates honest or malicious training (gradient ascent poisoning for adversarial agents), reports energy consumption and metadata for minimum participation threshold. |
| [`backdoor.py`](src/backdoor.py) | Multi-signal Byzantine/backdoor detector. Audits each client update using: **(1)** ground-truth accuracy on a validation set, **(2)** Median Absolute Deviation (MAD) thresholding, **(3)** L2 update distance from global model, **(4)** cosine alignment of weight deltas, and **(5)** temporal accuracy stability across rounds. |
| [`fedavg.py`](src/fedavg.py) | Federated Averaging — computes element-wise mean of client weight dictionaries and returns aggregated global weights as PyTorch tensors. |
| [`energy.py`](src/energy.py) | Real-time hardware energy monitor — queries LibreHardwareMonitor's JSON API for CPU/GPU power draw (W) and temperatures (°C). Supports Intel, AMD, and NVIDIA hardware auto-detection. |
| [`sim_green.py`](src/sim_green.py) | Standalone Green FL simulation — models client efficiency η = (D/E)·exp(−β(T−T_ref)), confidence scoring with sigmoid-gated trust, and iterative trust ledger updates over configurable rounds. |
| [`reward.py`](src/reward.py) | PoUL reward calculator — assigns coins inversely proportional to training loss, with zero reward for detected malicious clients. |
| [`db.py`](src/db.py) | PostgreSQL persistence layer — manages connection retries, initializes `global_rounds` and `ledger` tables, and provides helpers for logging round accuracy and distributing token rewards. |
| [`config.py`](src/config.py) | Central configuration — number of clients, total rounds, learning rate, energy constants, and reward scaling. |
| [`prom_metrics.py`](src/prom_metrics.py) | Prometheus metric definitions — global accuracy gauge, round counter, per-client energy, coin balance, malicious status, and contribution score. |
| [`requirement.txt`](src/requirement.txt) | Python dependencies. |

### Kubernetes (`k8s/`)

| File | Resources | Description |
|---|---|---|
| [`namespace.yaml`](k8s/namespace.yaml) | Namespace | Creates the `fl-sim` namespace for all resources. |
| [`db.yaml`](k8s/db.yaml) | Deployment + Service | PostgreSQL 13 instance (`postgres:5432`). Stores the trust ledger and round history. |
| [`kafka.yaml`](k8s/kafka.yaml) | 2× Deployment + 2× Service | Apache Kafka (`kafka:9092`) + Zookeeper (`zookeeper:2181`). Message bus for coordinator↔client communication. |
| [`coordinator.yaml`](k8s/coordinator.yaml) | Deployment + Service | Runs `fl-coordinator:v1` (1 replica). Connects to Kafka and Postgres, exposes Prometheus metrics on port 8000. |
| [`client.yaml`](k8s/client.yaml) | Deployment + Headless Service | Runs `fl-client:v1` (scalable replicas). Each pod gets a unique `NODE_ID` via the Downward API. Prometheus discovers all pods via DNS service discovery. |
| [`monitoring.yaml`](k8s/monitoring.yaml) | ConfigMap + 2× Deployment + 2× Service | Prometheus (NodePort `:30090`) scrapes coordinator and clients. Grafana (NodePort `:30000`) for dashboards. |

### Docker

| File | Builds |
|---|---|
| [`Dockerfile.coordinator`](Dockerfile.coordinator) | `fl-coordinator:v1` — Python 3.9 slim, installs dependencies, runs `coordinator.py`. |
| [`client.Dockerfile`](client.Dockerfile) | `fl-client:v1` — Python 3.9 slim, installs dependencies, runs `client.py`. |

---

## Quick Start

### Prerequisites

- Python 3.9+
- Docker & Kubernetes (Minikube, Kind, or a cloud cluster)
- Apache Kafka (or use the K8s manifest)
- PostgreSQL (or use the K8s manifest)
- *(Optional)* LibreHardwareMonitor for real energy telemetry

### Local Simulation (No Infrastructure)

```bash
# 1. Install dependencies
cd src
pip install -r requirement.txt

# 2. Run the standalone green FL simulation
python sim_green.py

# 3. Or run the full simulation (requires Kafka + Postgres)
python main.py
```

### Kubernetes Deployment

```bash
# 1. Build Docker images
docker build -f Dockerfile.coordinator -t fl-coordinator:v1 .
docker build -f client.Dockerfile -t fl-client:v1 .

# 2. Deploy infrastructure
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/db.yaml
kubectl apply -f k8s/kafka.yaml

# 3. Wait for Kafka and Postgres to become ready
kubectl -n fl-sim wait --for=condition=ready pod -l app=postgres --timeout=120s
kubectl -n fl-sim wait --for=condition=ready pod -l app=kafka --timeout=120s

# 4. Deploy FL components
kubectl apply -f k8s/coordinator.yaml
kubectl apply -f k8s/client.yaml

# 5. Deploy monitoring
kubectl apply -f k8s/monitoring.yaml

# 6. Scale clients as needed
kubectl -n fl-sim scale deployment fl-clients --replicas=5
```

### Accessing Dashboards

| Service | URL |
|---|---|
| Prometheus | `http://<node-ip>:30090` |
| Grafana | `http://<node-ip>:30000` |
| Coordinator Metrics | `http://<node-ip>:8000/metrics` |

---

## Configuration

All tunable parameters are centralized in [`config.py`](src/config.py):

```python
NUM_CLIENTS         = 4       # Number of FL clients
TOTAL_ROUNDS        = 5       # Communication rounds
MIN_CLIENTS_PER_ROUND = 3     # Minimum clients selected per round


GLOBAL_EPOCHS       = 1
LOCAL_EPOCHS         = 3
LEARNING_RATE        = 0.01

ALPHA_REWARD         = 10.0   # Reward scaling factor
POWER_CONSUMPTION_W  = 50.0   # Assumed power draw (Watts) for energy estimation
```

Environment variables for the database connection are defined in [`db.py`](src/db.py):

| Variable | Default | Description |
|---|---|---|
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_NAME` | `fl_simulation` | Database name |
| `DB_USER` | `postgres` | Database user |
| `DB_PASS` | `secret` | Database password |
| `DB_PORT` | `5432` | Database port |

---

## Prometheus Metrics

The following metrics are exported and scraped by Prometheus:

| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `global_model_accuracy` | Gauge | — | Current global model accuracy |
| `fl_rounds_total` | Counter | — | Total federated learning rounds completed |
| `client_energy_joules` | Gauge | `client_id` | Per-client energy consumption (Joules) |
| `fc_coin_balance` | Gauge | `client_id` | Per-client PoUL token balance |
| `client_malicious` | Gauge | `client_id` | Malicious flag (1 = adversarial, 0 = honest) |
| `client_contribution_score` | Gauge | `client_id` | Per-client contribution/confidence score |
| `client_accuracy` | Gauge | `client_id` | Per-client local model accuracy |

---

## Byzantine Detection Pipeline

The backdoor detector in [`backdoor.py`](src/backdoor.py) applies a multi-layered audit on every received client update:

```
┌──────────────┐     ┌─────────────────┐     ┌────────────────┐
│  1. Accuracy │────►│  2. L2 Distance │────►│  3. Cosine     │
│     Audit    │     │   from Global   │     │   Alignment    │
└──────────────┘     └─────────────────┘     └───────┬────────┘
                                                     │
                                                     ▼
                                             ┌───────────────┐
                                             │ 4. Temporal    │
                                             │  Consistency   │
                                             └───────┬───────┘
                                                     │
                                                     ▼
                                             ┌───────────────┐
                                             │  MAD-based    │
                                             │  Thresholding │
                                             └───────┬───────┘
                                                     │
                                              ┌──────▼──────┐
                                              │  FLAGGED /  │
                                              │   CLEAN     │
                                              └─────────────┘
```

Robust thresholds are computed using **Median Absolute Deviation (MAD)** with a configurable `k` factor (default 2.5), avoiding sensitivity to outlier-heavy distributions common in adversarial settings.

---

## Green Energy Simulation

The standalone simulator ([`sim_green.py`](src/sim_green.py)) models the full AFL decision loop without infrastructure dependencies:

**Client Efficiency:**

$$\eta_i = \frac{D_i}{E_i} \cdot \exp\!\bigl(-\beta\,(T_i - T_{\mathrm{ref}})\bigr)$$

**Confidence Score:**

$$C_i = \tau_i \cdot \sigma(\Delta\mathrm{acc}_i) \cdot \sigma(\eta_i) \cdot \exp(-\gamma \cdot m_i)$$

**Trust Update:**

$$\tau_i^{(t+1)} = (1 - \alpha)\,\tau_i^{(t)} + \alpha\bigl[\lambda_1 \sigma(\Delta\mathrm{acc}_i) + \lambda_2 \sigma(\eta_i) - \lambda_3 m_i\bigr]$$

Where σ is the sigmoid function, m_i is the malicious flag, and α controls trust memory decay.

---

## Project Structure

```
AFL/
├── Dockerfile.coordinator       # Coordinator container image
├── client.Dockerfile            # Client container image
├── k8s/
│   ├── namespace.yaml           # fl-sim namespace
│   ├── db.yaml                  # PostgreSQL deployment
│   ├── kafka.yaml               # Kafka + Zookeeper
│   ├── coordinator.yaml         # FL coordinator pod
│   ├── client.yaml              # FL client pods (scalable)
│   └── monitoring.yaml          # Prometheus + Grafana
└── src/
    ├── main.py                  # Simulation entry point
    ├── coordinator.py           # Round orchestration + CNN model
    ├── client.py                # Agentic client with attack sim
    ├── backdoor.py              # Byzantine detection pipeline
    ├── fedavg.py                # Federated Averaging aggregation
    ├── energy.py                # Hardware energy monitoring
    ├── sim_green.py             # Green FL math simulation
    ├── reward.py                # PoUL reward calculation
    ├── db.py                    # PostgreSQL persistence
    ├── config.py                # Configuration constants
    ├── prom_metrics.py          # Prometheus metric definitions
    └── requirement.txt          # Python dependencies
```

---


<p align="center">
  <sub>Built with PyTorch · Kafka · Kubernetes · Prometheus · Grafana</sub>
</p>
