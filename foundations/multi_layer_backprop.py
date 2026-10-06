import numpy as np
from typing import List

class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        predictions = {}
        
        #z1 = x*W1 +b1
        z1 = []
        for i in range(len(W1)):
            res = 0
            for j in range(len(x)):
                res += W1[i][j] * x[j]
            z1.append(res+b1[i])

        #reLu
        a1 = []
        for i in range(len(z1)):
            if z1[i] < 0:
                a1.append(0)
            else:
                a1.append(z1[i])
        
        #ŷ
        z2 = []
        for i in range(len(W2)):
            res = 0
            for j in range(len(a1)):
                res += W2[i][j] * a1[j]
            z2.append(res+b2[i])

        #MSE = np.sum((z2 - y_true)**2)/len(y_true)
        loss = 0
        for i in range(len(y_true)):
            loss += (z2[i] - y_true[i])**2
        loss /= len(y_true)

        # dL_dW1
        dL_dW1 = []
        for i in range(len(W1)):
            l = []
            for j in range(len(x)):
                res = 0
                for k in range(len(z2)):
                    res += (2 * (z2[k] - y_true[k]) / len(y_true)) * W2[k][i]
                if z1[i] > 0:
                    l.append(res * x[j])
                else:
                    l.append(0)
            dL_dW1.append(l)

        # dL_db1
        dL_db1 = []
        for i in range(len(z1)):
            res = 0
            for k in range(len(z2)):
                res += (2 * (z2[k] - y_true[k]) / len(y_true)) * W2[k][i]
            if z1[i] > 0:
                dL_db1.append(res)
            else:
                dL_db1.append(0)

        #dL_dW2
        dL_dW2 = []
        for i in range(len(W2)):
            l = []
            for j in range(len(a1)):
                l.append(2* ((z2[i] - y_true[i])/len(y_true)) *a1[j])
            dL_dW2.append(l)
                
        #dL_db2
        dL_db2 = []
        for i in range(len(z2)):
            dL_db2.append(2* ((z2[i] - y_true[i])/len(y_true)))

        predictions["loss"] = np.round(loss, 4)
        predictions["dW1"] = np.round(dL_dW1, 4)
        predictions["db1"] = np.round(dL_db1, 4)
        predictions["dW2"] = np.round(dL_dW2, 4)
        predictions["db2"] = np.round(dL_db2, 4)

        return predictions