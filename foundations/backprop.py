import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        z = np.sum(x*w) +b
        y_pred = 1/(1+np.exp(-z))

        n = len(w)
        mse = np.sum((y_pred - y_true)**2)/n

        dL_dw = (y_pred-y_true)*y_pred*(1-y_pred)*x
        dL_db = (y_pred-y_true)*y_pred*(1-y_pred)

        return ((np.round(dL_dw, 5)), (np.round(dL_db, 5)))