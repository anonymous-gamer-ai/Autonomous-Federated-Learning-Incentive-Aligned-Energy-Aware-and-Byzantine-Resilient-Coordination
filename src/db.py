import os
import psycopg2
import time


DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "fl_simulation")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "secret")
DB_PORT = os.getenv("DB_PORT", "5432")

DSN = f"dbname={DB_NAME} user={DB_USER} password={DB_PASS} host={DB_HOST} port={DB_PORT}"

def get_connection():
    """Retries connection to DB until successful (waits for Postgres container)."""
    while True:
        try:
            conn = psycopg2.connect(DSN)
            return conn
        except psycopg2.OperationalError:
            print("[DB] Waiting for Postgres to start...")
            time.sleep(3)

def init_db():
    """Creates necessary tables in Postgres."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # 1. Training History
        cur.execute("""
            CREATE TABLE IF NOT EXISTS global_rounds (
                id SERIAL PRIMARY KEY,
                round_num INTEGER,
                global_accuracy REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # 2. Ledger (Replaces local blockchain.json for Cloud)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ledger (
                id SERIAL PRIMARY KEY,
                client_id TEXT,
                round_num INTEGER,
                coins REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        print("[DB] Tables initialized.")
    except Exception as e:
        print(f"[DB] Init Error: {e}")

def log_round(round_num, global_acc):
    """Logs Global Accuracy to DB."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO global_rounds (round_num, global_accuracy) VALUES (%s, %s)",
            (round_num, global_acc)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Log Error: {e}")

def add_reward(client_id, round_num, amount):
    """Adds coins to the DB ledger."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO ledger (client_id, round_num, coins) VALUES (%s, %s, %s)",
            (client_id, round_num, amount)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Reward Error: {e}")