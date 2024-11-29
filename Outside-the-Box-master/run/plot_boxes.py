import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

from data import *
from run.experiment_helper import *
from trainers import StandardTrainer


def run_script():
    model_name, data_name, stored_network_name, total_classes = instance_balance()
    known_classes = np.arange(total_classes - 1)
    model_path = str(stored_network_name + ".h5")
    data_train_model = DataSpec(randomize=False, classes=known_classes)
    data_test_model = DataSpec(randomize=False, classes=known_classes)
    data_train_monitor = DataSpec(randomize=False, classes=known_classes)
    data_test_monitor = DataSpec(randomize=False, classes=known_classes)
    data_run = DataSpec(randomize=False, classes=np.arange(total_classes))

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

    novelty_X = all_x[(all_y == 2)]

    plot_2d_projection(history=history, monitor=monitor, layer=layer, category_title=model_name,
                       known_classes=known_classes, novelty_marker="*", dimensions=[0, 1], novelty=novelty_X)

    pca = PCA(n_components=2)
    all_x = pca.fit_transform(all_x)
    X = all_x[np.isin(all_y, known_classes)]
    y = all_y[np.isin(all_y, known_classes)]
    novelty_X = all_x[(all_y == 2)]

    # Train
    clf = DecisionTreeClassifier().fit(X, y)
    tree = clf.tree_

    # Print the threshold values for each node in the tree
    for i in range(tree.node_count):
        if tree.children_left[i] != tree.children_right[i]:  # only print non-leaf nodes
            print(f"Node {i}: feature {tree.feature[i]} <= {tree.threshold[i]}")
    predictions = clf.predict(X)

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

    data = DataSpec()
    data.set_data(x=X, y=predictions)
    data.set_y(to_categorical(data.y(), num_classes=number_of_classes(known_classes), dtype='float32'))

    c2v = {}
    for i in range(number_of_classes(known_classes)):
        c2v[i] = X[predictions == i].tolist()

    layer2values = {layer: X}
    monitor_manager.n_clusters = 1
    monitor_manager._layers = [layer]
    monitor_manager.refine_clusters(data_train=data, layer2values=layer2values,
                                    statistics=Statistics(), class2values=c2v)

    history.set_ground_truths(all_y)
    history.set_layer2values({layer: all_x})
    plot_2d_projection(history=history, monitor=monitor, layer=layer, category_title=model_name,
                       known_classes=known_classes, novelty_marker="*", dimensions=[0, 1], novelty=novelty_X)

    num_misclassified_novelties = 0
    for point in novelty_X:
        if monitor is not None:
            for i, ai in enumerate(monitor.abstraction(layer).abstractions()):
                if ai.isempty():
                    continue
                if ai.isknown(point, skip_confidence=True)[0]:
                    num_misclassified_novelties += 1
                    break

    num_misclassified_known = 0
    for i, point in enumerate(X):
        if monitor is not None:
            isKnown = False
            for _, ai in enumerate(monitor.abstraction(layer).abstractions()):
                if ai.isempty():
                    continue
                if ai.isknown(point, skip_confidence=True)[0]:
                    isKnown = True
                    break
            if not isKnown:
                num_misclassified_known += 1

    print(f"Number of true positives: {len(novelty_X) - num_misclassified_novelties}")
    print(f"Number of false negatives: {num_misclassified_novelties}")
    print(f"Number of true negatives: {len(X) - num_misclassified_known}")
    print(f"Number of false positives: {num_misclassified_known}")
    save_all_figures()


if __name__ == "__main__":
    run_script()
