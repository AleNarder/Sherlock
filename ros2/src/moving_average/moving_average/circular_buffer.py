import numpy as np


# Ordered circular buffer
class CircularBuffer ():

    def __init__(self, size: int, entry_size: int, fill_value: float = 0.0):
        self.size   = size
        self.buffer = np.empty((size, entry_size))
        self.buffer.fill(fill_value)

    def add(self, value: list[float]):
        self.buffer = np.roll(self.buffer, -1, axis = 0)
        self.buffer[-1] = value

    def get(self) -> np.ndarray:
        return self.buffer

    def get_average(self) -> np.ndarray:
        return np.mean(self.buffer, axis=0)