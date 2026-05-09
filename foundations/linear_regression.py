import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        forward_pass = np.matmul(X, weights)
        return np.round(forward_pass, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        
        y = model_prediction
        y_hat = ground_truth
        loss = np.mean(np.square(y - y_hat))
        
        return round(loss, 5)
