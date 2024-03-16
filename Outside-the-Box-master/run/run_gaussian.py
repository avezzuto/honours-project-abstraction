import scipy.stats as stats

from data import *
from utils import *
from abstractions.GaussianBasedAbstraction import GaussianBasedAbstraction
from trainers import *
from monitoring import *

data = [[0.02, 0.33], [0.04, 0.3], [0., 0.27], [0., 0.3],
        [0., 0.39], [0.3, 0.45], [0.38, 0.51], [0.4, 0.48], [0.52, 0.48]]

n = 2
threshold = 3

anomaly_point = np.array([0.2, 0.4])

abstraction = GaussianBasedAbstraction(n, threshold)
for data_point in data:
    abstraction.add(data_point)

for i in range(anomaly_point.shape[0]):
    mean, cov_matrix = abstraction.neuron_list[i].calculate_distribution()

    if cov_matrix.shape == ():
        DM = cov_matrix
    else:
        delta = anomaly_point - mean
        distance_squared = np.dot(np.dot(delta, np.linalg.inv(cov_matrix)), delta.T)
        DM = np.sqrt(distance_squared)

    x = np.linspace(mean - 3 * DM, mean + 3 * DM, 100)
    plt.plot(x, stats.norm.pdf(x, mean, DM))
    print(anomaly_point[i])
    plt.plot(anomaly_point[i], 0, marker='*', ls='none', ms=20)
    plt.show()

print(abstraction.evaluate(anomaly_point))
