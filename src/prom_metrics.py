# prom_metrics.py
from prometheus_client import Gauge, Counter, start_http_server

# Dashboard Metrics
PROM_ACCURACY = Gauge('global_model_accuracy', 'Current Global Model Accuracy')
PROM_ROUNDS   = Counter('fl_rounds_total', 'Total Federated Learning Rounds')
PROM_ENERGY   = Gauge('client_energy_joules', 'Client Energy Consumption', ['client_id'])
PROM_BALANCE  = Gauge('fc_coin_balance', 'Client FA Coin Balance', ['client_id'])
PROM_MALICIOUS = Gauge('client_malicious', 'Malicious Status (1=Bad, 0=Good)', ['client_id'])
PROM_CONTRIB  = Gauge('client_contribution_score', 'Client Contribution Score', ['client_id'])

def start_metrics_server(port=8000):
    try:
        start_http_server(port)
        print(f"✅ Prometheus Metrics Server running on port {port}")
    except Exception as e:
        print(f"⚠️  Metrics server already running or error: {e}")