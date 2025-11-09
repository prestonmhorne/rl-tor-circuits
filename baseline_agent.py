# baseline_agent.py

import numpy as np

class BaselineAgent:
    """Tor's Current Implementation: Bandwidth-Weighted Random Selection"""
    
    def __init__(self, action_dim, state_dim):
        self.action_dim = action_dim
        self.state_dim = state_dim

    def select_action(self, state, action_mask, relay_info):
        valid_actions = np.where(action_mask)[0]
        
        bandwidths = np.array([relay_info[i]['bandwidth'] for i in valid_actions])
        probs = bandwidths / bandwidths.sum()
        selected_idx = np.random.choice(len(valid_actions), p=probs)

        return valid_actions[selected_idx]