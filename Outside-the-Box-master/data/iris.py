import csv
import numpy as np

from data import loadData, loadLabels
from utils import DataSpec, load_data, filter_labels


def load_iris(data_train_model: DataSpec, data_test_model: DataSpec, data_train_monitor: DataSpec,
              data_test_monitor: DataSpec, data_run: DataSpec):
    # names of the data files
    data_train_filename = "C:/Users/andre/Downloads/Outside-the-Box-master/Outside-the-Box-master/data/Iris/training_iris.csv"
    data_test_filename = "C:/Users/andre/Downloads/Outside-the-Box-master/Outside-the-Box-master/data/Iris/testing_iris.csv"

    dataset_train = np.array(list(csv.reader(open(data_train_filename))))[1:]

    x_train = dataset_train[:, :4]
    y_train = dataset_train[:, 4:]

    dataset_test = np.array(list(csv.reader(open(data_test_filename))))[1:]

    x_test = dataset_test[:, :4]
    y_test = dataset_test[:, 4:]

    # Reshaping the array to 4-dims so that it can work with the Keras API
    """x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)"""

    data_train_model.set_data(x=x_train, y=y_train)
    data_train_monitor.set_data(x=x_train, y=y_train)
    data_test_model.set_data(x=x_test, y=y_test)
    data_test_monitor.set_data(x=x_test, y=y_test)
    data_run.set_data(x=x_test, y=y_test)
    pixel_depth = 255.0
    all_classes_network, all_classes_rest = load_data(
        data_train_model=data_train_model, data_test_model=data_test_model, data_train_monitor=data_train_monitor,
        data_test_monitor=data_test_monitor, data_run=data_run, pixel_depth=pixel_depth)
    # labels
    labels_all = ['label' + str(i) for i in range(3)]

    labels_network = filter_labels(labels_all, all_classes_network)
    labels_rest = filter_labels(labels_all, all_classes_rest)

    return all_classes_network, labels_network, all_classes_rest, labels_rest
