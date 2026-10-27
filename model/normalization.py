import numpy as np
from numpy.typing import NDArray

class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        # gamma: 1D scale parameter (same length as x)
        # beta: 1D shift parameter (same length as x)
        eps = 1e-5
        mu = np.mean(x)
        n = len(x)

        var = np.sum((x-mu)**2)/n
        div = np.sqrt(var + eps)

        res = ((x-mu)/div) * gamma + beta
        return np.round(res, 5)