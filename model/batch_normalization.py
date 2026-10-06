import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        x = np.array(x)
        gamma = np.array(gamma)
        beta = np.array(beta)
        running_mean = np.array(running_mean)
        running_var = np.array(running_var)

        if training:
            x_b_mean = np.mean(x, 0)
            x_b_var_sq = np.mean(np.square(x - x_b_mean), 0)

            x_hat = (x - x_b_mean) / np.sqrt(x_b_var_sq + eps)

            running_mean = (1-momentum) * running_mean + momentum * x_b_mean
            running_var = (1-momentum) * running_var + momentum * x_b_var_sq            
        else:
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        y = gamma * x_hat
        y = y + beta

        y = np.round(y, 4)
        y = [[v for v in y[i]] for i in range(y.shape[0])]
        return y, np.round(running_mean, 4).tolist(), np.round(running_var, 4).tolist()
        
        
