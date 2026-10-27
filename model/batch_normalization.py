import numpy as np
from typing import Tuple, List

class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        x = np.array(x)
        m = momentum
        running_mean = np.array(running_mean)
        running_var = np.array(running_var)

        if training == False:
            num = x - running_mean
            div = np.sqrt(running_var + eps)
            
            x_hat = num / div
            y = gamma * x_hat + beta

            return (np.round(y,4).tolist(),
                np.round(running_mean,4).tolist(),
                np.round(running_var,4).tolist())
        
        mu = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        
        div = np.sqrt(var + eps)
        x_hat = (x-mu)/div

        y = gamma * x_hat + beta

        running_mean = (1-m) * (running_mean) +m * (mu)
        running_var = (1-m) * (running_var) +m * (var)

        return (np.round(y,4).tolist(),
                np.round(running_mean,4).tolist(),
                np.round(running_var,4).tolist())