from scipy.spatial.distance import euclidean  # NOTE: changing this requires adapting PartitionBasedAbstraction as well
from math import inf

from .Abstraction import Abstraction
from . import PointCollection
from utils import *
from .NeuronGaussian import NeuronGaussian


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
