import numpy as np
from tensorflow import keras

from .plot_utils import *


class DataSet:
    def __init__(self, X, Y, metadata=None, metadata_label="metadata"):
        self.X = X
        self.Y = Y
        if metadata is not None:
            setattr(self, metadata_label, metadata)


class Data:
    def __init__(
        self, raw_data_path, use_transpose=False, seed=8, valid_ratio=0, test_ratio=0,
    ):
        self.raw_data_path = raw_data_path
        self.use_transpose = use_transpose
        self.seed = seed
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.train_ratio = 1 - self.valid_ratio - self.test_ratio
        assert self.train_ratio + self.valid_ratio + self.test_ratio == 1
        self._preprocess_data()

    def _preprocess_data(self):
        # initial data load
        raw_data = np.load(self.raw_data_path)
        X = np.moveaxis(raw_data["X"], -1, 0).astype("float32")
        Y = keras.utils.to_categorical(raw_data["Y"].astype("float32"))
        p = raw_data["p"].astype("float32")

        # double the data because the transpose of a percolating lattice is also percolating
        if self.use_transpose:
            X = np.concatenate((X, X.transpose([0, 2, 1, 3])), axis=0)
            Y = np.concatenate((Y, Y), axis=0)
            p = np.concatenate((p, p), axis=0)

        # shuffle the data
        np.random.seed(self.seed)
        perm = np.random.permutation(len(X))
        X = X[perm]
        Y = Y[perm]
        p = p[perm]

        # set metadata
        self.input_shape = X.shape[1:]
        self.n_classes = Y.shape[1]

        # set the cutoffs
        valid_cutoff = int(self.train_ratio * len(X))
        test_cutoff = int((self.train_ratio + self.valid_ratio) * len(X))

        # create the train set
        self.train = DataSet(
            X[:valid_cutoff], Y[:valid_cutoff], p[:valid_cutoff], metadata_label="p"
        )

        # create the validation set
        self.valid = DataSet(
            X[valid_cutoff:test_cutoff],
            Y[valid_cutoff:test_cutoff],
            p[valid_cutoff:test_cutoff],
            metadata_label="p",
        )

        # create the test set
        self.test = DataSet(
            X[test_cutoff:], Y[test_cutoff:], p[test_cutoff:], metadata_label="p"
        )

        # print the statistics
        print("--------------------Data--------------------")
        print("X_train shape:", self.train.X.shape)
        print("Y_train shape:", self.train.Y.shape)
        print("X_valid shape:", self.valid.X.shape)
        print("Y_valid shape:", self.valid.Y.shape)
        print("X_test shape :", self.test.X.shape)
        print("Y_test shape :", self.test.Y.shape)
        print("--------------------------------------------")

    def plot_sample(self, set_label, sample_index, save_as=False):
        lattice = getattr(self, set_label).X[sample_index].squeeze(axis=-1)
        plot_lattice(lattice, title="Sample %s" % sample_index, save_as=save_as)

    def analyze_predictions(
        self, set_label, model, n_misclassified=5, plot_misclassified=False
    ):
        print("--------------------%s set results--------------------" % set_label)
        dataset = getattr(self, set_label)
        dataset.Y_hat = model.predict(dataset.X)

        misclassified = np.argwhere(
            np.argmax(dataset.Y_hat, axis=1) != np.argmax(dataset.Y, axis=1)
        ).reshape(-1)
        dataset.misclassified = misclassified

        accuracy = (len(dataset.Y) - len(misclassified)) / len(dataset.Y)
        dataset.accuracy = accuracy
        print("Accuracy: %.5f" % accuracy)
        plot_p_accuracies(
            dataset.Y_hat,
            dataset.Y,
            dataset.p,
            "accuracies_" + set_label + "_" + str(self.input_shape[0]) + ".png",
        )

        print("n samples misclassified: %i" % len(misclassified))
        for (i, sample_index) in enumerate(dataset.misclassified[:n_misclassified]):
            print("Sample ", sample_index)
            print("\tTruth     :", dataset.Y[sample_index])
            print("\tPrediction:", dataset.Y_hat[sample_index])
            print("\tp         :", dataset.p[sample_index])
            if plot_misclassified:
                self.plot_sample(
                    set_label,
                    sample_index,
                    save_as="misclassified_"
                    + set_label
                    + "_"
                    + str(self.input_shape[0])
                    + "_"
                    + str(sample_index)
                    + ".png",
                )
