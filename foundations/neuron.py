import numpy as np
from numpy.typing import NDArray

class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        # return round(your_answer, 5)
        z = np.sum(x*w) + b

        if activation == "sigmoid":
            res = 1/(1+np.exp(-z))
        else:
            res = np.maximum(0, z)
        
        return np.round(res, 5)
