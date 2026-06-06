# llm_brain.py
import openai
import json
from config import OPENAI_API_KEY, LLM_MODEL, MIN_CLIENTS_PER_ROUND

class LLMCoordinator:
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        if self.api_key:
            openai.api_key = self.api_key
            print("LLM Brain: ONLINE (OpenAI Connected)")
        else:
            print("LLM Brain: OFFLINE (Using Logic Fallback)")

    def select_optimal_clients(self, available_clients_metadata):
        """
        Analyzes client stats and returns the IDs of the best candidates.
        input: list of dicts [{'id': 1, 'energy': 50, 'malicious': False, 'reputation': 0.9}, ...]
        """
        print("\nLLM is thinking... Analyzing Client Registry.")
        
        # 1. First Pass: Hard Security Filter (The "Cognitive Layer")
        safe_clients = [c for c in available_clients_metadata if not c['malicious']]
        
        if not safe_clients:
            print("CRITICAL: No safe clients available!")
            return []

        # 2. Strategic Selection
        if self.api_key:
            return self._ask_gpt4(safe_clients)
        else:
            return self._heuristic_logic(safe_clients)

    def _heuristic_logic(self, safe_clients):
        """
        Fallback logic if no API key.
        Selects clients with highest reputation and lowest energy.
        """
        # Sort by: High Reputation (desc), Low Energy (asc)
        ranked = sorted(safe_clients, key=lambda x: (-x['reputation'], x['energy']))
        
        # Select top N (between MIN and total available)
        optimal_count = max(MIN_CLIENTS_PER_ROUND, len(ranked) // 2)
        selected = ranked[:optimal_count]
        
        selected_ids = [c['id'] for c in selected]
        print(f"Logic Decision: Selected {len(selected_ids)} clients based on efficiency.")
        return selected_ids

    def _ask_gpt4(self, safe_clients):
        """
        Uses OpenAI to decide purely based on context.
        """
        prompt = f"""
        You are the Coordinator of a Secure Federated Learning System.
        Goal: Select the optimal subset of clients to maximize training efficiency and security.
        
        Rules:
        1. Prioritize clients with HIGH reputation.
        2. Prioritize clients with LOW energy consumption.
        3. You must select at least {MIN_CLIENTS_PER_ROUND} clients.
        
        Available Safe Clients:
        {json.dumps(safe_clients, indent=2)}
        
        Return strictly a JSON list of selected client IDs. Example: [1, 4, 5]
        """
        
        try:
            response = openai.ChatCompletion.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a strategic AI system optimizer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            content = response.choices[0].message.content
            selected_ids = json.loads(content)
            print(f"GPT-4 Decision: Selected clients {selected_ids}")
            return selected_ids
        except Exception as e:
            print(f"LLM Error: {e}. Reverting to fallback.")
            return self._heuristic_logic(safe_clients)