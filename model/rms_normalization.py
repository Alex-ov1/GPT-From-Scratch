import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        x = np.array(x)
        g = np.array(gamma)
        
        rms = np.sqrt(np.sum(x**2)/len(x) +eps)
        x_hat = x/rms
        y = g * x_hat
        
        return np.round(y, 4).tolist()