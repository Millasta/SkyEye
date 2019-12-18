"""This module contains all the algorithm to search the best features to use in our recognition model"""
import traceback
from abc import ABCMeta, abstractmethod

from PyQt5 import QtCore

from program.model.datastructure.evaluation import Evaluation
from program.model.featureextraction.feature import DefinitionFeature
from program.model.featureextraction.image_features_extractor import ImageFeaturesExtractor
from program.model.machinelearning.machinelearning import RecognitionModel


class FeatureSelector(metaclass=ABCMeta):
    """This abstract class represents a research algorithm. It contains just a unique name

    """

    def __init__(self, name: str, log: QtCore.pyqtSignal) -> None:
        """Create a FeatureSelector object

        :param name: The name of the feature selector
        :type name: str
        """
        super().__init__()
        self.name = name
        self.log = log

    @abstractmethod
    def search_best_features(self, model: RecognitionModel, csv_extracted_path: str, training_image_names: [str],
                             nb_classes: int,
                             possible_features: [DefinitionFeature], nb_samples_per_classes: int = 100,
                             nb_samples_for_evaluation: int = 10) -> [DefinitionFeature]:
        """Search the best features to use for a recognition model and a population of samples.

        :param model: The recognition model we will use
        :type model: RecognitionModel
        :param csv_extracted_path: The path to the CSV files containing all the features computed
        :type csv_extracted_path: str
        :param training_image_names: The list of training image names
        :type training_image_names: str
        :param nb_classes: The number of classes to recognize
        :type nb_classes: int
        :param possible_features: The list of possible features at the beginning
        :type possible_features: [DefinitionFeature]
        :param nb_samples_per_classes: The number of samples we should use per classes to train the model
        :type nb_samples_per_classes: int
        :param nb_samples_for_evaluation: The number of samples we should use per classes to evaluate the model
        :type nb_samples_for_evaluation: int
        :return: The list of features we should use with our model
        :rtype: [DefinitionFeature]
        """
        return


class SequentialForwardSelection(FeatureSelector):
    """This Feature Selector add the best feature until it stops increase the recognition rate

    """

    def __init__(self) -> None:
        """Create a SequentialForwardSelection object

        """
        super().__init__('Sequential Floating Selection', None)

    def search_best_features(self, model: RecognitionModel, csv_extracted_path: str, training_image_names: [str],
                             nb_classes: int,
                             possible_features: [DefinitionFeature], nb_samples_per_classes: int = 100,
                             nb_samples_for_evaluation: int = 10) -> [DefinitionFeature]:
        """Search the best features to use for a recognition model and a population of samples.

        :param model: The recognition model we will use
        :type model: RecognitionModel
        :param csv_extracted_path: The path to the CSV files containing all the features computed
        :type csv_extracted_path: str
        :param training_image_names: The list of training image names
        :type training_image_names: str
        :param nb_classes: The number of classes to recognize
        :type nb_classes: int
        :param possible_features: The list of possible features at the beginning
        :type possible_features: [DefinitionFeature]
        :param nb_samples_per_classes: The number of samples we should use per classes to train the model
        :type nb_samples_per_classes: int
        :param nb_samples_for_evaluation: The number of samples we should use per classes to evaluate the model
        :type nb_samples_for_evaluation: int
        :return: The list of features we should use with our model
        :rtype: [DefinitionFeature]

        .. seealso:: For more details on the algorithm, go to `this site
                     <https://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/>`_
        """
        result = set()
        old_recognition_rate = Evaluation(0, None)
        recognition_rate = Evaluation(0.01, None)
        while recognition_rate.recognition_rate - old_recognition_rate.recognition_rate > 0:
            old_recognition_rate = recognition_rate
            recognition_rate = Evaluation(0.01, None)
            best_feature = None
            for feature in possible_features:
                list_feature = set(result)
                list_feature.add(feature)
                nb_samples_per_classes_to_extract = nb_samples_per_classes + nb_samples_for_evaluation
                learning_data_set = ImageFeaturesExtractor.generate_training_base(csv_extracted_path,
                                                                                  training_image_names,
                                                                                  nb_samples_per_classes_to_extract,
                                                                                  nb_classes,
                                                                                  list_feature)
                evaluate_data_set = ImageFeaturesExtractor.extract_validation_data_set_from_learning_data_set(
                    nb_samples_for_evaluation, nb_classes, learning_data_set)
                labels, samples = learning_data_set.get_labels_and_samples()
                model.train(samples, labels)
                labels, samples = evaluate_data_set.get_labels_and_samples()
                add_gain = model.evaluate_model(samples, labels)
                if add_gain.recognition_rate > recognition_rate.recognition_rate:
                    recognition_rate = add_gain
                    best_feature = feature
            if recognition_rate.recognition_rate > old_recognition_rate.recognition_rate:
                if best_feature is not None:
                    result.add(best_feature)
                    possible_features.remove(best_feature)

        return result


class SequentialForwardFloatingSelection(FeatureSelector):
    """This Feature Selector add the best feature and remove the less useful feature
    until it stops increase the recognition rate

    """

    def __init__(self, log: QtCore.pyqtSignal = None) -> None:
        """Create a SequentialForwardFloatingSelection object

        """
        super().__init__('Sequential Forward Floating Selection', log)

    def search_best_features(self, model: RecognitionModel, csv_extracted_path: str, training_image_names: [str],
                             nb_classes: int,
                             possible_features: [DefinitionFeature], nb_samples_per_classes: int = 100,
                             nb_samples_for_evaluation: int = 10) -> [DefinitionFeature]:
        """Search the best features to use for a recognition model and a population of samples.

        :param model: The recognition model we will use
        :type model: RecognitionModel
        :param csv_extracted_path: The path to the CSV files containing all the features computed
        :type csv_extracted_path: str
        :param training_image_names: The list of training image names
        :type training_image_names: str
        :param nb_classes: The number of classes to recognize
        :type nb_classes: int
        :param possible_features: The list of possible features at the beginning
        :type possible_features: [DefinitionFeature]
        :param nb_samples_per_classes: The number of samples we should use per classes to train the model
        :type nb_samples_per_classes: int
        :param nb_samples_for_evaluation: The number of samples we should use per classes to evaluate the model
        :type nb_samples_for_evaluation: int
        :return: The list of features we should use with our model
        :rtype: [DefinitionFeature]

        .. seealso:: For more details on the algorithm, go to `this site
                     <https://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/>`_
        """
        try:
            result = set()
            old_recognition_rate = Evaluation(0, None)
            new_recognition_rate = Evaluation(0.01, None)
            continue_to_add = True
            while continue_to_add or len(result) == 1:
                continue_to_add = False
                continue_to_remove = True
                best_feature = None
                for feature in possible_features:
                    list_feature = set(result)
                    list_feature.add(feature)
                    nb_samples_to_extract = nb_samples_per_classes + nb_samples_for_evaluation
                    learning_data_set = ImageFeaturesExtractor.generate_training_base(csv_extracted_path,
                                                                                      training_image_names,
                                                                                      nb_samples_to_extract,
                                                                                      nb_classes,
                                                                                      list_feature)
                    evaluate_data_set = ImageFeaturesExtractor.extract_validation_data_set_from_learning_data_set(
                        nb_samples_for_evaluation, nb_classes, learning_data_set)
                    labels, samples = learning_data_set.get_labels_and_samples()
                    model.train(samples, labels)
                    labels, samples = evaluate_data_set.get_labels_and_samples()
                    add_gain = model.evaluate_model(samples, labels)
                    self.log.emit('Try with ' + feature.full_name + ' added : ' + str(add_gain.recognition_rate))
                    if add_gain.recognition_rate > new_recognition_rate.recognition_rate:
                        new_recognition_rate = add_gain
                        best_feature = feature
                if new_recognition_rate.recognition_rate > old_recognition_rate.recognition_rate and best_feature is not None:
                    result.add(best_feature)
                    self.log.emit(
                        best_feature.full_name + ' added : ' + str(new_recognition_rate.recognition_rate * 100) + '%')
                    possible_features.remove(best_feature)
                    continue_to_add = True
                    old_recognition_rate = new_recognition_rate

                while continue_to_remove:
                    continue_to_remove = False
                    if len(result) != 1:
                        old_recognition_rate = new_recognition_rate
                        best_feature = None
                        for feature in result:
                            list_feature = set(result)
                            list_feature.remove(feature)
                            nb_samples_to_extract = nb_samples_per_classes + nb_samples_for_evaluation
                            learning_data_set = ImageFeaturesExtractor.generate_training_base(csv_extracted_path,
                                                                                              training_image_names,
                                                                                              nb_samples_to_extract,
                                                                                              nb_classes,
                                                                                              list_feature)
                            evaluate_data_set = ImageFeaturesExtractor.extract_validation_data_set_from_learning_data_set(
                                nb_samples_for_evaluation, nb_classes, learning_data_set)
                            labels, samples = learning_data_set.get_labels_and_samples()
                            model.train(samples, labels)
                            labels, samples = evaluate_data_set.get_labels_and_samples()
                            remove_gain = model.evaluate_model(samples, labels)
                            self.log.emit(
                                'Try with ' + feature.full_name + ' removed : ' + str(remove_gain.recognition_rate))
                            if remove_gain.recognition_rate > new_recognition_rate.recognition_rate:
                                new_recognition_rate = remove_gain
                                best_feature = feature
                        if new_recognition_rate.recognition_rate > old_recognition_rate.recognition_rate:
                            if best_feature is not None:
                                result.remove(best_feature)
                                self.log.emit(best_feature.full_name + ' removed : ' + str(
                                    new_recognition_rate.recognition_rate * 100) + '%')
                                possible_features.append(best_feature)
                                continue_to_remove = True
                                old_recognition_rate = new_recognition_rate

            return result
        except:
            traceback.print_exc()

# class FeatureSelection(object):
#     """ This class contains a group of functions of selecting features.
#
#         The features can be selected in the following ways:
#
#         * manual: given a list of feature names
#         * by algorithm :
#             1. choose an algorithm to determiner the best features.
#             2. choose an algorithm and set a number k to determiner the K-best features
#
#         .. note::
#             Most algorithms used here are based on the
#             `sklearn.feature_selection <http://scikit-learn.org/stable/modules/feature_selection.html>`_ module
#     """
#
#     @staticmethod
#     def select_features(samples, selected_feature_names):
#         """ Select features manually.
#
#         :param samples: the original all-feature samples
#         :param selected_feature_names: a list a feature names, must in order
#         :return: the new samples with the selected features
#         """
#         feature_indexes = list()
#         for feature_name in selected_feature_names:
#             feature_indexes.append(FEATURE_NAMES.index(feature_name))
#
#         samples_new = list()
#         for sample in samples:
#             sample_new = list()
#             for index in feature_indexes:
#                 sample_new.append(sample[index])
#             samples_new.append(sample_new)
#
#         return samples_new
#
#     @staticmethod
#     def auto_select_k_best_features(samples_train, labels_train, samples_test, k=2):
#         """ Auto select k best features.
#
#         :param samples_train: the original all-feature training samples
#         :param labels_train: the corresponding training labels
#         :param samples_test: the original all-feature testing samples
#         :param k: the number of wanted features, default value is 2.
#         :return: the new training samples and the new testing samples with the k-best features
#         """
#         chooser = SelectKBest(chi2, k)
#         chooser.fit(samples_train, labels_train)
#         logging.info(chooser.get_support())
#         return chooser.transform(samples_train), chooser.transform(samples_test)
#
#     @staticmethod
#     def auto_select_best_features(samples_train, labels_train, samples_test, method='TREE'):
#         """ Auto select best features, the number of the features is determined by the algorithm.
#
#         :param samples_train: the original all-feature training samples
#         :param labels_train: the corresponding training labels
#         :param samples_test: the original all-feature testing samples
#         :param method: the algorithm, default value is 'TREE'
#         :return: the new training samples and the new testing samples with the best features
#
#         ..notes:: For the moment, two methods are available.
#
#             * 'TREE' : Tree-based feature selection
#             * 'RFECV' : Recursive feature elimination with cross-validation
#         """
#
#         if method == 'RFECV':
#             logging.info('Recursive feature elimination with cross-validation....')
#             # This is too slow, unfit to transform dataset
#             svc = SVC(kernel="linear", cache_size=1024)
#             # The "accuracy" scoring is proportional to the number of correct
#             # classifications
#             rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
#                           scoring='accuracy')
#             rfecv.fit(samples_train, labels_train)
#
#             logging.info("Optimal number of features : %d " % rfecv.n_features_)
#             logging.info("with accuracy: ", rfecv.grid_scores_)
#
#             logging.info('L1-based feature selection...')
#             lsvc = LinearSVC(C=0.001, penalty="l1", dual=False).fit(samples_train, labels_train)
#             model = SelectFromModel(lsvc, prefit=True)
#             samples_train_new = model.transform(samples_train)
#             logging.info(len(samples_train_new), len(samples_train_new[0]))
#
#             samples_test_new = model.transform(samples_test)
#             logging.info(len(samples_test_new), len(samples_test_new[0]))
#             logging.info('feature selection is done!')
#             return samples_train_new, samples_test_new
#
#         else:
#             logging.info('Tree-based feature selection....')
#             logging.info(len(samples_train), len(samples_train[0]))
#
#             # Tree-based feature selection
#             clf = ExtraTreesClassifier()
#             clf = clf.fit(samples_train, labels_train)
#             weights = clf.feature_importances_
#
#             logging.info("ALl feature weights : \n", weights)
#
#             model = SelectFromModel(clf, prefit=True)
#             samples_train_new = model.transform(samples_train)
#             # print(len(samples_train_new), len(samples_train_new[0]))
#
#             samples_test_new = model.transform(samples_test)
#
#             selected_feature_num = len(samples_test_new[0])
#
#             logging.info('{} features are selected: '.format(selected_feature_num))
#
#             weights_ordered = sorted(weights, reverse=True)
#
#             ordered_features = list()
#
#             for x in weights_ordered:
#                 ordered_features.append(FEATURE_NAMES[weights.tolist().index(x)])
#
#             logging.info(ordered_features[:selected_feature_num])
#
#             logging.info('Tree-based feature selection is done!')
#
#             return samples_train_new, samples_test_new
#
#     @staticmethod
#     def feature_view(samples, labels, feature_name_1="feature 1", feature_name_2="feature 2"):
#         """ View two features with samples.
#
#         :param samples: the 2-feature samples
#         :param labels: the corresponding labels
#
#         .. warning:: the samples must be composed by exactly 2 features.
#         """
#         if len(samples[0]) != 2:
#             logging.info('To view features, there must be 2 features. Not {} !!'.format(len(samples[0])))
#
#         class_num = len(REAL_CLASS_NAMES) + 1
#
#         f1 = list()
#         f2 = list()
#
#         for i in range(class_num):
#             f1.append(list())
#             f2.append(list())
#
#         for label_index in range(len(labels)):
#             label = labels[label_index]
#             sample = samples[label_index]
#             f1[label].append(sample[0])
#             f2[label].append(sample[1])
#
#         fig = plt.figure()
#         ax = plt.subplot(111)
#
#         colors = ['black', 'blue', 'green', 'red', 'cyan', 'yellow']
#
#         ax.scatter(f1[0], f2[0], marker='o', c=colors[0], label='C0', edgecolors='white')
#         for i in range(len(CLASS_COLORS)):
#             ax.scatter(f1[i + 1], f2[i + 1], marker='o', c=colors[i + 1],
#               label=REAL_CLASS_NAMES[i], edgecolors='white')
#
#         plt.title('Feature Viewer')
#         plt.xlabel(feature_name_1)
#         plt.ylabel(feature_name_2)
#         plt.grid(True)
#
#         box = ax.get_position()
#         ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#
#         # Put a legend to the right of the current axis
#         ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#
#         plt.show()
