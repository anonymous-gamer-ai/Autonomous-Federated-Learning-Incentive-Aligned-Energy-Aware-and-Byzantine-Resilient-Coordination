import torch
import numpy as np

def aggregate_weights(weights_list):
    """
    Federated Averaging: Average the weights from all clients.
    Input: List of dictionaries (state_dicts with lists instead of tensors)
    Output: Dictionary of tensors
    """
    if not weights_list:
        return None

    # Take the first client's weights as the template
    avg_weights = {}
    first_weights = weights_list[0]

    for key in first_weights.keys():
        # Create a list of this specific layer's weights from all clients
        layer_updates = [np.array(w[key]) for w in weights_list]
        
        # Calculate mean
        mean_layer = np.mean(layer_updates, axis=0)
        
        # Convert back to Torch Tensor
        avg_weights[key] = torch.tensor(mean_layer, dtype=torch.float32)

    return avg_weights