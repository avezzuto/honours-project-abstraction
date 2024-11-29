from . import *
from .iris import load_iris
from .wine import load_wine
from .dermatology import load_dermatology
from .balance import load_balance


def get_data_loader(string):
    if string == "GTSRB":
        return load_GTSRB
    elif string == "iris":
        return load_iris
    elif string == "wine":
        return load_wine
    elif string == "dermatology":
        return load_dermatology
    elif string == "balance":
        return load_balance
    elif string == "CIFAR10":
        return load_CIFAR_10
    elif string == "MNIST":
        return load_MNIST
    elif string == "F_MNIST":
        return load_F_MNIST
    elif string == "ToyData":
        return load_ToyData
    else:
        raise(ValueError("Could not find data " + string + "!"))
