import numpy as np
from typing import List

class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:

        # 1. Align arrays to match the platform's internal shape orientation
        x_arr = np.array(x).reshape(1, -1)              # Shape: (1, input_size)
        w1_arr = np.array(W1)                           # Shape: (hidden_size, input_size)
        b1_arr = np.array(b1).reshape(1, -1)            # Shape: (1, hidden_size)
        
        w2_arr = np.array(W2)                           # Shape: (output_size, hidden_size)
        b2_arr = np.array(b2).reshape(1, -1)            # Shape: (1, output_size)
        y_true_arr = np.array(y_true).reshape(1, -1)    # Shape: (1, output_size)

        # 2. Forward Pass
        y_pred = x_arr @ w1_arr.T + b1_arr              # Shape: (1, hidden_size)
        y_hat = np.maximum(0, y_pred)                   # Shape: (1, hidden_size)
        y_pred_2 = y_hat @ w2_arr.T + b2_arr            # Shape: (1, output_size)

        # Loss Calculation
        loss = np.power(y_pred_2 - y_true_arr, 2)
        mse = np.round(np.mean(loss), 4)

        # 3. Backward Pass
        N = len(y_true)
        
        # Calculate output error gradient
        DL = 2 * (y_pred_2 - y_true_arr) / N            # Shape: (1, output_size)

        # Gradients for Layer 2
        dw2 = DL.T @ y_hat                              # Shape: (output_size, hidden_size)
        db2 = np.sum(DL, axis=0)                        # Shape: (output_size,)
        
        # Travel Inward
        dy_hat = DL @ w2_arr                            # Shape: (1, hidden_size)
        dy_pred = dy_hat * (y_pred > 0)                 # Shape: (1, hidden_size)

        # Gradients for Layer 1
        dw1 = dy_pred.T @ x_arr                         # Shape: (hidden_size, input_size)
        db1 = np.sum(dy_pred, axis=0)                   # Shape: (hidden_size,)

        # Convert back to standard python lists with accurate convention alignments
        return {
          'loss':  float(mse),
          'dW1':   np.round(dw1, 4).tolist(),
          'db1':   np.round(db1, 4).tolist(),
          'dW2':   np.round(dw2, 4).tolist(),
          'db2':   np.round(db2, 4).tolist()
        }