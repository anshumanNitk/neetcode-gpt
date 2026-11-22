import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        
        y_pred=np.squeeze(np.matmul(x,w))+b
        if activation == "sigmoid":
            y_pred_binary=np.where(
                y_pred>=0,
                1/(1+np.exp(-y_pred)),
                np.exp(y_pred)/(np.exp(y_pred)+1)
            )
            return np.round(y_pred_binary,5)
        if activation == "relu":
            y_pred_relu=np.maximum(0,y_pred)
            return np.round(y_pred_relu,5)
        
        return np.round(y_pred,5)
