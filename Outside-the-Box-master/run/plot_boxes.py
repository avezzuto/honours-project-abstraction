from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

from data import *
from run.experiment_helper import *
from trainers import StandardTrainer


def compute_confusion_matrix(novelty_X, known_X, monitor, layer):
    num_misclassified_novelties = 0
    for point in novelty_X:
        if monitor is not None:
            for _, ai in enumerate(monitor.abstraction(layer).abstractions()):
                if ai.isempty():
                    continue
                if ai.isknown(point[:2], skip_confidence=True)[0]:
                    num_misclassified_novelties += 1
                    break

    num_misclassified_known = 0
    for point in known_X:
        if monitor is not None:
            isKnown = False
            for _, ai in enumerate(monitor.abstraction(layer).abstractions()):
                if ai.isempty():
                    continue
                if ai.isknown(point[:2], skip_confidence=True)[0]:
                    isKnown = True
                    break
            if not isKnown:
                num_misclassified_known += 1

    print(f"Number of true positives: {len(novelty_X) - num_misclassified_novelties}")
    print(f"Number of false negatives: {num_misclassified_novelties}")
    print(f"Number of true negatives: {len(known_X) - num_misclassified_known}")
    print(f"Number of false positives: {num_misclassified_known}")


def run_script():
    model_name, data_name, stored_network_name, total_classes = instance_iris()
    known_classes = np.arange(total_classes - 1)
    all_classes = np.arange(total_classes)
    novelty_class = list(set(all_classes) - set(known_classes))[0]
    model_path = str(stored_network_name + ".h5")
    data_train_model = DataSpec(randomize=False, classes=known_classes)
    data_test_model = DataSpec(randomize=False, classes=known_classes)
    data_train_monitor = DataSpec(randomize=False, classes=known_classes)
    data_test_monitor = DataSpec(randomize=False, classes=known_classes)
    data_run = DataSpec(randomize=False, classes=all_classes)

    all_classes_network, labels_network, all_classes_rest, labels_rest = get_data_loader(data_name)(
        data_train_model=data_train_model, data_test_model=data_test_model, data_train_monitor=data_train_monitor,
        data_test_monitor=data_test_monitor, data_run=data_run)

    model, _ = get_model(model_name=model_name, data_train=data_train_model, data_test=data_test_model,
                         n_classes=total_classes, model_trainer=StandardTrainer(), n_epochs=100, batch_size=1,
                         statistics=Statistics(), model_path=model_path)

    layer = 2

    # create monitor
    layer2abstraction = {layer: OctagonAbstraction(euclidean_distance)}
    monitor = Monitor(layer2abstraction=layer2abstraction)
    monitor_manager = MonitorManager([monitor], n_clusters=len(known_classes))

    # run instance
    monitor_manager.normalize_and_initialize(model, len(labels_rest))
    monitor_manager.train(model=model, data_train=data_train_monitor, data_test=data_test_monitor,
                          statistics=Statistics())

    # create plot
    history = History()
    history.set_ground_truths(data_run.ground_truths())
    layer2values, _ = obtain_predictions(model=model, data=data_run, layers=[layer])
    history.set_layer2values(layer2values)

    plot_colors = "ryb"

    all_y = np.array(history.ground_truths)
    all_x = np.array(history.layer2values[layer])

    novelty_X = all_x[(all_y == novelty_class)]

    plot_2d_projection(history=history, monitor=monitor, layer=layer, category_title=model_name,
                       known_classes=known_classes, novelty_marker="*", dimensions=[0, 1], novelty=novelty_X)

    X = all_x[np.isin(all_y, known_classes)]

    print("CONFUSION MATRIX BEFORE DECISION TREES")
    compute_confusion_matrix(novelty_X=novelty_X, known_X=X, monitor=monitor, layer=layer)

    pca = PCA(n_components=2)

    training_data_x, _ = obtain_predictions(model=model, data=data_train_model, layers=[layer])
    training_data_x = training_data_x.get(layer)
    training_y = data_train_model.ground_truths()
    training_data_x_pca = pca.fit_transform(training_data_x)
    clf = DecisionTreeClassifier().fit(training_data_x_pca, training_y)
    tree = clf.tree_
    predictions_training = clf.predict(training_data_x_pca)

    # Print the threshold values for each node in the tree
    for i in range(tree.node_count):
        if tree.children_left[i] != tree.children_right[i]:  # only print non-leaf nodes
            print(f"Node {i}: feature {tree.feature[i]} <= {tree.threshold[i]}")

    all_x = pca.transform(all_x)
    X = all_x[np.isin(all_y, known_classes)]
    y = all_y[np.isin(all_y, known_classes)]
    novelty_X = all_x[(all_y == novelty_class)]

    predictions_testing = clf.predict(X)

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
    for i, color in zip(range(len(known_classes)), plot_colors):
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

    data_train = DataSpec()
    data_train.set_data(x=training_data_x_pca, y=predictions_training)
    data_train.set_y(to_categorical(data_train.y(), num_classes=number_of_classes(known_classes), dtype='float32'))

    data_test = DataSpec()
    data_test.set_data(x=X, y=predictions_testing)
    data_test.set_y(to_categorical(data_test.y(), num_classes=number_of_classes(known_classes), dtype='float32'))

    c2v = {}
    for i in range(number_of_classes(known_classes)):
        c2v[i] = training_data_x_pca[predictions_training == i].tolist()

    layer2values = {layer: training_data_x_pca}
    monitor_manager.n_clusters = 1
    monitor_manager._layers = [layer]
    monitor_manager.refine_clusters(data_train=data_train, layer2values=layer2values,
                                    statistics=Statistics(), class2values=c2v)

    # create plot
    history.set_ground_truths(all_y)
    history.set_layer2values({layer: all_x})
    plot_2d_projection(history=history, monitor=monitor, layer=layer, category_title=model_name,
                       known_classes=known_classes, novelty_marker="*", dimensions=[0, 1], novelty=novelty_X)

    print("CONFUSION MATRIX AFTER DECISION TREES")
    compute_confusion_matrix(novelty_X=novelty_X, known_X=X, monitor=monitor, layer=layer)
    save_all_figures()


if __name__ == "__main__":
    run_script()
