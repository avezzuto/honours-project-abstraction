from scipy.spatial.distance import euclidean  # NOTE: changing this requires adapting PartitionBasedAbstraction as well
from math import inf

from scipy.stats import multivariate_normal

from .Abstraction import Abstraction
from .NeuronGaussian import NeuronGaussian
import numpy as np


class GaussianBasedAbstraction(Abstraction):

    def __init__(self, n, threshold):
        self.votes = 0
        self.neuron_list = [NeuronGaussian(k=2) for _ in range(n)]
        self.threshold = threshold

    def name(self):
        return "GaussianBasedAbstraction"

    def add(self, vector):
        for idx, val in enumerate(vector):
            self.neuron_list[idx].add(val)

    def evaluate(self, vector):
        score = 0
        for idx, val in enumerate(vector):
            if self.neuron_list[idx].vote(val):
                score += 1

        return score >= self.threshold

    def run_gaussian(self, data):
        # Estimating the mean vector.
        muVector = np.mean(data, axis=0)
        print(muVector)

        # Creating the estimated covariance matrix.
        cov = np.cov(data, rowvar=False)
        print(cov)

        # Create a multivariate normal distribution using the estimated parameters
        rv = multivariate_normal(mean=muVector, cov=cov)
