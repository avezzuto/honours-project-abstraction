from scipy.spatial.distance import euclidean  # NOTE: changing this requires adapting PartitionBasedAbstraction as well
from math import inf

from .Abstraction import Abstraction
from . import PointCollection
from utils import *
from .NeuronGaussian import NeuronGaussian
import scipy.stats as stats
import random

from data import *
from utils import *


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

    def run_gaussian(self, novelties):
        data = [[0.02, 0.33], [0.04, 0.3], [0., 0.27], [0., 0.3],
                [0., 0.39], [0.3, 0.45], [0.38, 0.51], [0.4, 0.48], [0.52, 0.48]]

        n = 2

        anomaly_point = np.array([0.2, 0.4])

        for data_point in data:
            self.add(data_point)

        for i in range(n):
            mean, cov_matrix = self.neuron_list[i].calculate_distribution()

            if cov_matrix.shape == ():
                DM = cov_matrix
            else:
                delta = anomaly_point - mean
                distance_squared = np.dot(np.dot(delta, np.linalg.inv(cov_matrix)), delta.T)
                DM = np.sqrt(distance_squared)

            x = np.linspace(mean - 3 * DM, mean + 3 * DM, 100)
            plt.plot(x, stats.norm.pdf(x, mean, DM))
            for j in range(len(novelties)):
                plt.plot(novelties[j][i], 0, marker='*', ls='none', ms=20)
            plt.show()
