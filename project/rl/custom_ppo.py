from stable_baselines3 import PPO
import wandb
import numpy as np

class CustomPPO(PPO):
    def __init__(self, *args, **kwargs):
        super(CustomPPO, self).__init__(*args, **kwargs)    
        self.ep_info_list=[]
        
    def _log_wandb(self):
        wandb.log({
            'learning_rate': self.learning_rate,
            'buffer_size': self.rollout_buffer.size()
        })               