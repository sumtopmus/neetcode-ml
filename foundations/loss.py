import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        bce = round(-1.0/len(y_true) * np.sum(y_true * np.log(np.maximum(y_pred, 1e-7)) + (1.0-y_true)*np.log(np.maximum(1 - y_pred, 1e-7))), 4)
        return bce if bce != -0.0 else 0.0

    def categorical_cross_entropy(self, y_true: NDArray[np.np.float6464], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        cce = round(-1.0/len(y_true) * np.sum(np.sum(y_true * np.log(np.maximum(y_pred, 1e-7)))), 4)
        return cce if cce != -0.0 else 0.0
