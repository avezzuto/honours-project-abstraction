import math
import numpy as np


class NeuronGaussian:

    def __init__(self, k):
        self.set = []
        self.k = k

    def add(self, value):
        self.set.append(value)

    def calculate_distribution(self):
        points = np.array(self.set)
        mean = np.mean(points, axis=0)
        covariance_matrix = np.cov(points, rowvar=False)

        return mean, covariance_matrix

    def vote(self, x):
        mean, cov = self.calculate_distribution()

        # Calculation of Mahalanobis distance
        if cov.shape == ():
            variance = sum((point - mean) ** 2 for point in self.set) / len(self.set)
            DM = math.sqrt(variance)
        else:
            delta = x - mean
            distance_squared = np.dot(np.dot(delta, np.linalg.inv(cov)), delta.T)
            DM = np.sqrt(distance_squared)

        if DM < self.k:
            return True

        return False
