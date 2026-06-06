def calculate_reward(loss, is_malicious):
    """
    Simple logic: High reward for low loss, 0 for malicious.
    """
    if is_malicious:
        return 0
    
    # Simple inverse loss scoring
    score = 100 / (loss + 0.1)
    return round(score, 2)