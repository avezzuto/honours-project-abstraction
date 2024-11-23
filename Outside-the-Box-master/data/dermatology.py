import csv
import numpy as np

from data import loadData, loadLabels
from utils import DataSpec, load_data, filter_labels


def load_dermatology(data_train_model: DataSpec, data_test_model: DataSpec, data_train_monitor: DataSpec,
                     data_test_monitor: DataSpec, data_run: DataSpec):
    # Names of the data files
    data_train_filename = "C:/Users/andre/Downloads/Outside-the-Box-master/Outside-the-Box-master/data/Dermatology/training_dermatology.csv"
    data_test_filename = "C:/Users/andre/Downloads/Outside-the-Box-master/Outside-the-Box-master/data/Dermatology/testing_dermatology.csv"

    # Load training data
    with open(data_train_filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        dataset_train = np.array([row for row in reader])

    x_train = dataset_train[:, :2].astype(float)
    y_train = dataset_train[:, 2]

    # Encode labels as integers
    _, y_train = np.unique(y_train, return_inverse=True)

    # Load testing data
    with open(data_test_filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        dataset_test = np.array([row for row in reader])

    x_test = dataset_test[:, :2].astype(float)
    y_test = dataset_test[:, 2]

    # Encode labels as integers
    _, y_test = np.unique(y_test, return_inverse=True)

    # Set data to DataSpec objects
    data_train_model.set_data(x=x_train, y=y_train)
    data_train_monitor.set_data(x=x_train, y=y_train)
    data_test_model.set_data(x=x_test, y=y_test)
    data_test_monitor.set_data(x=x_test, y=y_test)
    data_run.set_data(x=x_test, y=y_test)

    # Assuming pixel_depth is required for further processing
    pixel_depth = 255.0

    # Load data using the provided utility function
    all_classes_network, all_classes_rest = load_data(
        data_train_model=data_train_model, data_test_model=data_test_model, data_train_monitor=data_train_monitor,
        data_test_monitor=data_test_monitor, data_run=data_run, pixel_depth=pixel_depth
    )

    # Define labels
    labels_all = ['label' + str(i) for i in range(3)]

    # Filter labels
    labels_network = filter_labels(labels_all, all_classes_network)
    labels_rest = filter_labels(labels_all, all_classes_rest)

    return all_classes_network, labels_network, all_classes_rest, labels_rest
