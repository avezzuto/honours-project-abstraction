from data import *
from utils import *
from abstractions import *
from trainers import *
from monitoring import *
from abstractions.GaussianBasedAbstraction import GaussianBasedAbstraction


def run_script():
    # options
    seed = 0
    data_name = "ToyData"
    classes = [0, 1]
    n_classes = 2
    data_train_model = DataSpec(classes=classes)
    data_test_model = DataSpec(classes=classes)
    data_train_monitor = DataSpec(classes=classes)
    data_test_monitor = DataSpec(classes=classes)
    data_run = DataSpec(classes=classes)
    model_name = "ToyModel"
    model_path = "Toy-model.h5"
    n_epochs = 0
    batch_size = 128
    model_trainer = StandardTrainer()

    all_classes_network, labels_network, all_classes_rest, labels_rest = get_data_loader(data_name)(
        data_train_model=data_train_model, data_test_model=data_test_model, data_train_monitor=data_train_monitor,
        data_test_monitor=data_test_monitor, data_run=data_run)
    model, _ = get_model(model_name=model_name, data_train=data_train_model, data_test=data_test_model,
                         n_classes=n_classes, model_trainer=model_trainer, n_epochs=n_epochs, batch_size=batch_size,
                         statistics=Statistics(), model_path=model_path)

    # create monitor
    layer2abstraction = {1: OctagonAbstraction(euclidean_distance)}
    monitor = Monitor(layer2abstraction=layer2abstraction)
    monitor_manager = MonitorManager([monitor], n_clusters=1)

    # run instance
    monitor_manager.normalize_and_initialize(model, len(labels_rest))
    monitor_manager.train(model=model, data_train=data_train_monitor, data_test=data_test_monitor,
                          statistics=Statistics(), ignore_misclassifications=False)

    novelty = []
    num_data = 20
    for i in range(num_data):
        x0 = random.uniform(0.3, 0.52)
        x1 = random.uniform(0.45, 0.51)
        novelty.append([x0, x1])

    for i in range(num_data):
        x0 = random.uniform(0., 0.03)
        x1 = random.uniform(0.27, 0.39)
        novelty.append([x0, x1])

    # create plots
    history = History()
    history.set_ground_truths(data_run.ground_truths())
    layer = 1
    layer2values, _ = obtain_predictions(model=model, data=data_run, layers=[layer])
    history.set_layer2values(layer2values)
    plot_2d_projection(history=history, monitor=None, layer=layer, category_title=model_name, known_classes=classes,
                       novelty_marker="*", dimensions=[0, 1], novelty=novelty)
    plot_2d_projection(history=history, monitor=monitor, layer=layer, category_title=model_name, known_classes=classes,
                       novelty_marker="*", dimensions=[0, 1], novelty=novelty)

    n = 2
    threshold = 3
    abstraction = GaussianBasedAbstraction(n, threshold)
    data = np.array(layer2values.get(layer))
    abstraction.run_gaussian(data)

    save_all_figures(close=True)


if __name__ == "__main__":
    run_script()
