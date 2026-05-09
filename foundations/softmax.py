import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        shifted = z - np.max(z)
        exp = np.exp(shifted)
        
        return np.round(exp / np.sum(exp), 4)
