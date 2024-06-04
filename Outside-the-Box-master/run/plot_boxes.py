import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.tree import DecisionTreeClassifier

from abstractions.GaussianBasedAbstraction import GaussianBasedAbstraction
from data import *
from run.experiment_helper import *
from trainers import StandardTrainer


def run_script():
    model_name, data_name, stored_network_name, total_classes = instance_iris()
    classes = [0, 1]
    n_classes = 3
    model_path = "iris_neural_network.h5"
    data_train_model = DataSpec(randomize=False, classes=classes)
    data_test_model = DataSpec(randomize=False, classes=classes)
    data_train_monitor = DataSpec(randomize=False, classes=classes)
    data_test_monitor = DataSpec(randomize=False, classes=classes)
    data_run = DataSpec(randomize=False, classes=[0, 1, 2])

    all_classes_network, labels_network, all_classes_rest, labels_rest = get_data_loader(data_name)(
        data_train_model=data_train_model, data_test_model=data_test_model, data_train_monitor=data_train_monitor,
        data_test_monitor=data_test_monitor, data_run=data_run)

    model, _ = get_model(model_name=model_name, data_train=data_train_model, data_test=data_test_model,
                         n_classes=n_classes, model_trainer=StandardTrainer(), n_epochs=100, batch_size=1,
                         statistics=Statistics(), model_path=model_path)

    # create monitor
    layer2abstraction = {-2: OctagonAbstraction(euclidean_distance)}
    monitor = Monitor(layer2abstraction=layer2abstraction)
    monitor_manager = MonitorManager([monitor], n_clusters=3)

    # run instance
    monitor_manager.normalize_and_initialize(model, len(labels_rest))
    monitor_manager.train(model=model, data_train=data_train_monitor, data_test=data_test_monitor,
                          statistics=Statistics())

    # create plot
    history = History()
    history.set_ground_truths(data_run.ground_truths())
    layer = 2
    layer2values, _ = obtain_predictions(model=model, data=data_run, layers=[layer])
    history.set_layer2values(layer2values)
    plot_2d_projection(history=history, monitor=monitor, layer=layer, category_title=model_name, known_classes=classes,
                       novelty_marker="*", dimensions=[1, 2])

    pair = [1, 2]
    plot_colors = "ryb"

    X = np.array(history.layer2values[layer][:, pair])
    y = np.array(history.ground_truths)

    # Train
    clf = DecisionTreeClassifier().fit(X, y)

    predictions = clf.predict(X)
    class_2_points = X[predictions == 2]
    gt = y[predictions == 2]

    c2v = {1: class_2_points}

    data = DataSpec()
    data.set_data(x=class_2_points, y=gt)

    num_samples = data.y().shape[0]

    # Create a zero-filled array with shape (num_samples, 1) for one-hot encoding
    categorical = np.zeros((num_samples, 2), dtype='float32')

    # Set the value at index 1 of each row to 1 (representing class 2)
    categorical[:, 1] = 1

    data.set_y(categorical)

    layer2values = {2: class_2_points}
    monitor_manager.n_clusters = 1
    monitor_manager._layers = [2]
    monitor_manager.refine_clusters(data_train=data, layer2values=layer2values,
                                    statistics=Statistics(), class2values=c2v)

    history.set_ground_truths(gt)
    history.set_layer2values(layer2values)
    plot_2d_projection(history=history, monitor=monitor, layer=layer, category_title=model_name, known_classes=classes,
                       novelty_marker="*", dimensions=[0, 1])

    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Predict the class for each point in the mesh grid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(
            X[idx, 0],
            X[idx, 1],
            c=color,
            label=f"Class {i}",
            edgecolor="black",
            s=15,
        )

    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.title("Decision Boundary with Training Points")
    plt.show()

    save_all_figures()


if __name__ == "__main__":
    run_script()
