"""This module represents the Machine Learning.
"""

import logging
import warnings
from abc import abstractmethod, ABCMeta

import numpy as np
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC

from program.model.datastructure.evaluation import Evaluation

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


class RecognitionModel(metaclass=ABCMeta):
    """Parent class for all classifiers."""
    def __init__(self):
        self.model = None
        self.parameters = list()

    def load(self, fn):
        self.model = joblib.load(fn)

    def save(self, fn):
        joblib.dump(self.model, fn)

    @abstractmethod
    def train(self, samples, labels):
        """ Train the model.

        :param samples: the samples
        :type samples: [float]
        :param labels: the corresponding labels
        :type labels: [str]
        """
        return

    @abstractmethod
    def evaluate_model(self, samples, labels):
        """ Evaluate the model using the given samples and labels.

        :param samples: the samples
        :type samples: [float]
        :param labels: the corresponding labels
        :type labels: [str]
        :return: The Evaluation of the model
        :rtype: Evaluation
        """
        return

    @abstractmethod
    def set_parameters(self, parameters: {str, float}):
        pass

    @abstractmethod
    def auto_config(self, samples, labels):
        pass

    @abstractmethod
    def auto_predict(self, samples):
        pass


class SVM(RecognitionModel):
    """This is the Sci-Kit version of SVM.

    .. seealso:: `sklearn.svm.SVC <http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_
    """

    def __init__(self, C=100, gamma=0.000001):
        """ Initialize a SVM classifier.
        The default kernel is RBF(Radial Basis Function).

        :param C: The **C** parameter trades off misclassification of training examples against simplicity
                of the decision surface
        :param gamma:  the gamma parameter defines how far the influence of a single training example reaches,
                with low values meaning ‘far’ and high values meaning ‘close’

        .. seealso:: see more about **C** and **gamma** :
                    `RBF SVM parameters <http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html>`_
        """
        super().__init__()
        self.model = SVC()
        self.parameters.append(('C', C))
        self.parameters.append(('gamma', gamma))
        self.model.C = C
        self.model.gamma = gamma
        self.model.probability = True
        self.model.cache_size = 1024

    @staticmethod
    def grid_search_c_gamma(samples, labels):
        """ Grid Search for best combination of **C** and **gamma**.

        :param samples: the samples
        :param labels: the corresponding labels
        :return: the best combination of **C** and **gamma**
        :rtype: Dictionary

        .. note:: This method uses a linear kernel, not the RBF kernel.
        """

        # GridSearch for gamma and C
        C_range = np.logspace(-2, 10, 13)
        gamma_range = np.logspace(-9, 3, 13)
        param_grid = dict(gamma=gamma_range, C=C_range)
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
        grid.fit(samples, labels)

        return grid.best_params_

    @staticmethod
    def reduce_class_3_4(samples, labels):

        new_samples = list()
        new_labels = list()

        for index in range(len(labels)):
            if labels[index] < 3:
                new_labels.append(labels[index])
                new_samples.append(samples[index])

            if labels[index] == 5:
                new_labels.append(3)
                new_samples.append(samples[index])

        return new_samples, new_labels

    @staticmethod
    def reduce_class_1_5(samples, labels):

        new_samples = list()
        new_labels = list()

        for index in range(len(labels)):
            if labels[index] == 0:
                new_labels.append(0)
                new_samples.append(samples[index])

            if 1 < labels[index] < 5:
                new_labels.append(labels[index] - 1)
                new_samples.append(samples[index])

        return new_samples, new_labels

    @staticmethod
    def mixed_demo(samples, labels):
        """ This is a demo trying to combine Feature Selection and Grid Search.
        But it's too slow to produce a good result.
        To be improved.

        :param samples: the samples
        :param labels: the corresponding labels
        """

        X, y = samples, labels

        # This dataset is way too high-dimensional. Better do PCA:
        pca = PCA(n_components=2)

        # Maybe some original features where good, too?
        selection = SelectKBest(k=1)

        # Build estimator from PCA and Univariate selection:

        combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

        # Use combined features to transform dataset:
        X_features = combined_features.fit(X, y).transform(X)

        svm = SVC(kernel="linear")

        # Do grid search over k, n_components and C:

        pipeline = Pipeline([("features", combined_features), ("svm", svm)])

        param_grid = dict(features__pca__n_components=[1, 2, 3],
                          features__univ_select__k=[1, 2],
                          svm__C=[0.1, 1, 10])

        grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
        # grid_search.fit(X, y)
        # print(grid_search.best_estimator_)
        grid_search.fit(X_features, y)
        logging.info(grid_search.best_estimator_)

    def train(self, samples, labels):
        """ Train the model.

        :param samples: the samples
        :type samples: [float]
        :param labels: the corresponding labels
        :type labels: [str]
        """
        self.model.fit(samples, labels)

    def predict(self, samples):
        """ Predict the samples.

        :param samples: the samples
        :return: a list of label-result
        """
        return self.model.predict(samples)

    def predict_proba(self, samples):
        # Seems unused
        """ Predict the probabilities of each given sample to each label.
        :param samples: the samples
        :return: a list of probas-result
        """
        return self.model.predict_proba(samples)

    def evaluate_model(self, samples, labels):
        """ Evaluate the model using the given samples and labels.

        :param samples: the samples
        :type samples: [float]
        :param labels: the corresponding labels
        :type labels: [str]
        :return: The Evaluation of the model
        :rtype: Evaluation
        """
        resp = self.predict(samples)
        reco = (labels == resp).mean()

        class_names = sorted(list(set(labels)))
        nb_classes = len(class_names)
        confusion = np.zeros((nb_classes, nb_classes), np.int32)

        # the vertical i is the correct answer,
        # while the horizon j is the prediction result
        for i, j in zip(labels, resp):
            confusion[class_names.index(i), class_names.index(j)] += 1

        return Evaluation(reco, confusion)

    def train_and_evaluate(self, samples_train, labels_train, samples_test, labels_test):
        """ A combination of train() and evaluate_model()

        :param samples_train: the samples to train the model
        :param labels_train:  the corresponding labels to train samples
        :param samples_test: the samples to evaluate the model
        :param labels_test: the corresponding labels to evaluate samples
        :return: the error rate
        """
        self.model.fit(samples_train, labels_train)
        return self.evaluate_model(samples_test, labels_test)

    def auto_predict(self, samples):
        """This function will use the model to predict the samples, generate probability map and the final result. """
        probs = self.model.predict_proba(samples)
        labels = list()
        labels.append(self.model.predict(samples)[0])

        return probs, labels

    def set_parameters(self, parameters: {str, float}):
        self.parameters = list()
        self.model.C = parameters['C']
        self.model.gamma = parameters['gamma']
        self.parameters.append(('C', self.model.C))
        self.parameters.append(('gamma', self.model.gamma))

    def auto_config(self, samples, labels):
        best_parameters = self.grid_search_c_gamma(samples, labels)
        self.set_parameters(best_parameters)

# class RandomForest(RecognitionModel):
#     """ A simple Random Forest classifier.
#
#     """
#
#     def __init__(self):
#         self.model = RandomForestClassifier(n_estimators=25, max_depth=None, max_features=0.4, random_state=11)
#
#     def train(self, samples, labels):
#         """ Train the model.
#
#         :param samples: the samples
#         :type samples: [float]
#         :param labels: the corresponding labels
#         :type labels: [str]
#         """
#         self.model.fit(samples, labels)
#
#     def predict(self, samples):
#         return self.model.predict(samples)
#
#     def predict_prob(self, samples):
#         return self.model.predict_proba(samples)
#
#     def evaluate_model(self, samples, labels):
#         """ Evaluate the model using the given samples and labels.
#
#         :param samples: the samples
#         :type samples: [float]
#         :param labels: the corresponding labels
#         :type labels: [str]
#         :return: The Evaluation of the model
#         :rtype: Evaluation
#         """
#         resp = self.predict(samples)
#         err = (labels != resp).mean()
#
#         logging.info('error: %.2f %%' % (err * 100))
#
#         confusion = np.zeros((6, 6), np.int32)
#         # the vertical i is the correct answer,
#         # while the horizon j is the prediction result
#         for i, j in zip(labels, resp):
#             confusion[i, j] += 1
#
#         result = Evaluation(err, confusion)
#         return result
#
#     def train_and_evaluate(self, samples_train, labels_train, samples_test, labels_test):
#         self.model.fit(samples_train, labels_train)
#         return self.evaluate_model(samples_test, labels_test)
