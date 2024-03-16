import math


class NeuronGaussian:

    def __init__(self, k):
        self.set = []
        self.k = k

    def add(self, value):
        self.set.append(value)

    def vote(self, x):
        # Calculate mean and variance
        mean = sum(self.set) / len(self.set)
        variance = sum((point - mean) ** 2 for point in self.set) / len(self.set)
        std_dev = math.sqrt(variance)

        if mean - self.k * std_dev <= x <= mean + self.k * std_dev:
            return True

        return False
