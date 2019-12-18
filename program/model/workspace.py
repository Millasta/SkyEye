"""This module contains the definition of a workspace"""
import csv
import random
from os import makedirs, walk, path
from os.path import join, splitext

import numpy as np
from cv2.cv2 import imwrite
from treelib import Tree, Node

from program.model.datastructure.evaluation import Evaluation
from program.model.datastructure.feature_table import FeatureTable
from program.model.datastructure.image import Image
from program.model.datastructure.labeled_pixel import LabeledPixel
from program.model.datastructure.prediction import Prediction, PredictionPixel
from program.model.featureextraction.feature import DefinitionFeature
from program.model.featureextraction.image_features_extractor import all_definition_features, ImageFeaturesExtractor
from program.model.machinelearning.machinelearning import RecognitionModel, SVM
from program.model.resultbuilder.resultbuilder import ResultBuilder


def list_files_into_directory(directory_path: str) -> [str]:
    """Method to retrieve the list of direct files of a directory

        :param directory_path: The directory to parse
        :type directory_path: str
        :return: The list of files into the directory
        :rtype [str]
    """
    for root, directory_names, file_names in walk(directory_path):
        return file_names


def list_folders_into_directory(directory_path: str) -> [str]:
    """Method to retrieve the list of direct subfolders of a directory

        :param directory_path: The directory to parse
        :type directory_path: str
        :return: The list of subfolders into the directory
        :rtype [str]
    """
    for root, directory_names, file_names in walk(directory_path):
        return directory_names


# Definition of our required repository structure, this object will be used in the following method to verify
# if the workspace is faithful or not.
#
# Root
# ├── features
# │   ├── predict
# │   └── train_features
# ├── machinelearning
# │   ├── evaluate
# │   ├── results
# │   ├── svm
# │   └── train_machinelearning
# ├── predict-images
# └── train-images
#     └── img
required_tree = Tree()
required_tree.create_node("Root", "Root")
required_tree.create_node("features", "features", parent="Root")
required_tree.create_node("predict", "predict", parent="features")
required_tree.create_node("train_features", "train_features", parent="features")
required_tree.create_node("machinelearning", "machinelearning", parent="Root")
required_tree.create_node("evaluate", "evaluate", parent="machinelearning")
required_tree.create_node("results", "results", parent="machinelearning")
required_tree.create_node("svm", "svm", parent="machinelearning")
required_tree.create_node("train_machinelearning", "train_machinelearning", parent="machinelearning")
required_tree.create_node("predict-images", "predict-images", parent="Root")
required_tree.create_node("train-images", "train-images", parent="Root")
required_tree.create_node("img", "img", parent="train-images")


class Workspace(object):
    """This class represents the workspace where we will work during the process. It contains :

        * The names of the class we will classify
        * The list of the faithful training images
        * The path to the repository
        * A boolean to storage if the workspace is faithful

        This is the required structure for our workspace :

        Root

        ├── features

        │   ├── predict

        │   └── train_features

        ├── machinelearning

        │   ├── evaluate

        │   ├── results

        │   ├── svm

        │   └── train_machinelearning

        ├── predict-images

        └── train-images

            └── img

        .. warning:: You need to call the __init() function after each creation of a workspace object

        """

    def __init__(self, directory_path: str = None, recognition_model: RecognitionModel = None) -> None:
        """Create a workspace object

        :param directory_path: the path to the directory where we will work during the process.
        :type directory_path: string or None

        :except If the structure is not empty but improper, an exception will be thrown.
        """
        super().__init__()
        self.directory_path = directory_path
        self.class_names = list()
        self.training_image_names = list()
        self.learning_data_set = None
        self.evaluate_data_set = None

        self.extracted_features = list()
        self.selected_features = all_definition_features

        if recognition_model is not None:
            self.recognition_model = recognition_model
        else:
            self.recognition_model = SVM()
        if self.directory_path is not None:
            self.faithful = self.is_faithful()

    @property
    def train_image_path(self) -> str:
        """Returns the directory path of the train-images repository into the workspace

        :return: The directory path of train-images repository into the workspace
        :rtype: str
        """
        return join(self.directory_path, 'train-images')

    @property
    def predict_image_path(self) -> str:
        """Returns the directory path of the predict_images repository into the workspace

        :return: The directory path of predict_images repository into the workspace
        :rtype: str
        """
        return join(self.directory_path, 'predict-images')

    @property
    def feature_path(self) -> str:
        """Returns the directory path of the features repository into the workspace

        :return: The directory path of the features repository into the workspace
        :rtype: str
        """
        return join(self.directory_path, 'features')

    @property
    def train_features_path(self) -> str:
        """Returns the directory path of the train_features repository into the workspace

        :return: The directory path of the train_features repository into the workspace
        :rtype: str
        """
        return join(self.feature_path, 'train_features')

    @property
    def predict_features_path(self) -> str:
        """Returns the directory path of the train_features repository into the workspace

        :return: The directory path of the train_features repository into the workspace
        :rtype: str
        """
        return join(self.feature_path, 'predict')

    @property
    def machine_learning_path(self) -> str:
        """Returns the directory path of the machinelearning repository into the workspace

        :return: The directory path of the machinelearning repository into the workspace
        :rtype: str
        """
        return join(self.directory_path, 'machinelearning')

    @property
    def ml_train_samples_path(self) -> str:
        """Returns the directory path of the train_machinelearning repository into the workspace

        :return: The directory path of the train_machinelearning repository into the workspace
        :rtype: str
        """
        return join(self.machine_learning_path, 'train_machinelearning')

    @property
    def ml_evaluate_samples_path(self) -> str:
        """Returns the directory path of the evaluate repository into the workspace

        :return: The directory path of the evaluate repository into the workspace
        :rtype: str
        """
        return join(self.machine_learning_path, 'evaluate')

    @property
    def ml_predict_results_path(self) -> str:
        """Returns the directory path of the results repository into the workspace

        :return: The directory path of the results repository into the workspace
        :rtype: str
        """
        return join(self.machine_learning_path, 'results')

    @property
    def ml_svm_path(self) -> str:
        """Returns the directory path of the svm repository into the workspace

        :return: The directory path of the svm repository into the workspace
        :rtype: str
        """
        return join(self.machine_learning_path, 'svm')

    def init(self) -> None:
        """Initialize the workspace using the path to the repository, the workspace will parse the repository structure.

            *   If the structure matches with the required structure,
                the workspace will find the class names the user want to classify (by retrieving the directory names in
                ./train-images.
                After that, it will create a CSV file containing all the labeled pixels from all the training images.
            *   Otherwise, if the folder is empty, the structure will be created and the user will need to
                update the workspace when he will have stored his training images in the right folders.

            :raise AttributeError: If the structure is not empty but improper, an exception will be thrown.
        """
        self.faithful = self.is_faithful()
        if not self.faithful:
            # We retrieve a list of each element into the directory_path
            files_into_directory = list_files_into_directory(self.directory_path)
            if files_into_directory.__len__() == 0:
                self.create_required_structure()
                self.faithful = True
            else:
                raise AttributeError("The directory seems improper but not empty")
        self.class_names = self.find_class_names()
        self.training_image_names = self.find_training_image_names()
        self.extracted_features = self.find_features_already_extracted()
        if len(self.extracted_features) != 0:
            self.selected_features = set(self.extracted_features)
        self.generate_csv_dictionary()

    def is_faithful(self, required_folders: Node = None, directory_path: str = None) -> bool:
        """Recursive method to know if a directory contains all the structure required.
        The method will check if the repository contains all the sub folders into required_folders.
        Then it will check the sub folders into each sub folders from the initial list recursively.

        :param required_folders: The list of the first sub folders
        :type required_folders: Node
        :param directory_path: The directory to test
        :type directory_path: str
        :return: If the directory contains all the required folders
        :rtype: boolean

        .. seealso:: The tree structure documentation (`link <http://treelib.readthedocs.io/en/latest/>`_)
        """
        if required_folders is None:
            required_folders = required_tree.children("Root")
        if directory_path is None:
            directory_path = self.directory_path
        if required_folders.__len__() == 0:
            return True
        result = True
        list_folders = list_folders_into_directory(directory_path)
        for required_folder in required_folders:
            required_folder_tag = required_folder.tag
            required_folders_children = required_tree.children(required_folder_tag)
            new_directory_path = join(directory_path, required_folder_tag)
            if result and required_folder_tag in list_folders and required_folders_children.__len__() != 0:
                result = result and self.is_faithful(required_folders_children, new_directory_path)
            if required_folder_tag not in list_folders:
                return False
        return result

    def create_required_structure(self, required_folders: Node = None, directory_path: str = None) -> None:
        """Recursive method to create the missing folders by comparing with the required structure.
        The method will check if the repository contains all the sub folders into required_folders and create
        the missing ones.
        Then it will make the same for each sub folders from the initial list recursively.

        :param required_folders: The list of the first sub folders
        :type required_folders: Node
        :param directory_path: The directory path to update
        :type directory_path: str

        .. seealso:: The tree structure documentation (`link <http://treelib.readthedocs.io/en/latest/>`_)
        """
        if required_folders is None:
            required_folders = required_tree.children("Root")
        if directory_path is None:
            directory_path = self.directory_path
        list_folders = list_folders_into_directory(directory_path)
        for required_folder in required_folders:
            required_folder_tag = required_folder.tag
            required_folders_children = required_tree.children(required_folder_tag)
            new_directory_path = join(directory_path, required_folder_tag)
            if required_folder_tag not in list_folders:
                makedirs(new_directory_path)
            if required_folders_children.__len__() != 0:
                self.create_required_structure(required_folders_children, new_directory_path)

    def find_class_names(self) -> [str]:
        """This method will retrieve all the classes we want to classify.
        It will retrieve the directory names in ./Root/train-images

        :return: The list of classes to classify
        :rtype: [str]

        :raise AttributeError: If the directory isn't faithful
        """
        if not self.faithful:
            raise AttributeError('The workspace is not faithful')

        result = list()

        list_possible_class = list_folders_into_directory(join(self.directory_path, "train-images"))

        for possible_class in list_possible_class:
            if possible_class != "img":
                result.append(possible_class)

        self.class_names = result
        return result

    def find_training_image_names(self) -> [str]:
        """This method will retrieve all the training images.
        It will retrieve the images which have a binary image in each class folders/

        :return: The list of training images
        :rtype: [str]

        :raise AttributeError: If the directory isn't faithful
        """
        if not self.faithful:
            raise AttributeError

        list_possible_images_img = list_files_into_directory(join(join(self.directory_path, "train-images"), "img"))
        dict_possible_images_class = []
        result = []

        if self.class_names is None:
            self.class_names = self.find_class_names()

        # We retrieve all the files from each class folders
        for class_name in self.class_names:
            for file in list_files_into_directory(
                    join(join(self.directory_path, "train-images"), str(class_name))):
                dict_possible_images_class.append((class_name, file))

        # We verify for each training images if we have a binary image in each class directory
        for possible_images_img in list_possible_images_img:
            image = Image(join(join(self.train_image_path, "img"), possible_images_img))
            width = image.width
            height = image.height
            for class_name in self.class_names:
                if not (class_name, possible_images_img) in dict_possible_images_class:
                    image_matrix = np.zeros((height + 1, width + 1), np.uint8)
                    image_matrix.fill(255)
                    imwrite(join(join(self.train_image_path, class_name), possible_images_img), image_matrix)
            # If the extension is .tif, we add it to the training image list
            filename, file_extension = splitext(possible_images_img)
            if file_extension == '.tif':
                result.append(filename)

        self.training_image_names = result
        return result

    def update(self) -> None:
        """Method to update the attribute is_faithful, class_names and training_image_names.
        If something have changed, a new CSV will be generated.
        """
        self.faithful = self.is_faithful()
        if self.faithful:
            old_class_names = self.class_names
            old_training_image_names = self.training_image_names
            self.class_names = self.find_class_names()
            self.training_image_names = self.find_training_image_names()
            self.extracted_features = list()
            if old_class_names != self.class_names or old_training_image_names != self.training_image_names:
                self.generate_csv_dictionary()
        return

    def generate_csv_dictionary(self) -> None:
        """Method which generate a CSV file in ./Root/dict_pixels_labels.csv.
        It will contain all the labelled pixels from all the training images.

        Example :

        x;y;img_name;label

        1;46;z01;C4

        2;44;z01;C4

        23;76;z02;C3

        23;77;z02;C3

        23;78;z02;C3

        23;79;z02;C3

        """
        with open(join(self.directory_path, 'dict_pixels_labels.csv'), 'w', newline='') as csv_file:
            directory_to_training_images = join(self.directory_path, 'train-images')

            field_names = ['x', 'y', 'img_name', 'label']
            writer = csv.DictWriter(csv_file, fieldnames=field_names, delimiter=';')
            writer.writeheader()

            for img_name in self.training_image_names:
                for class_name in self.class_names:
                    image = Image(join(join(directory_to_training_images, class_name), img_name + '.tif'))
                    for x in list(range(image.width)):
                        for y in list(range(image.height)):
                            if image.get_pixel_gray_value((x, y)) == 0:
                                writer.writerow({'x': str(x), 'y': str(y), 'img_name': img_name, 'label': class_name})

        return

    def extract_training_images_features(self, image_name: str = None) -> None:
        if image_name is None:
            training_image_names = self.training_image_names
            self.extracted_features = list(self.selected_features)
        else:
            training_image_names = [image_name]

        if not self.selected_features.issubset(self.extracted_features):

            self.learning_data_set = None
            self.evaluate_data_set = None

            feature_names = list()
            for feature in self.selected_features:
                feature_names.append(feature.full_name)

            with open(join(self.directory_path, 'dict_pixels_labels.csv'), newline='') as csv_file:
                reader = csv.DictReader(csv_file, delimiter=';')
                row = next(reader)
                for image_training_name in training_image_names:
                    labeled_pixels = set()

                    # Retrieve labeled pixels
                    while row['img_name'] != 'EOF':
                        if row['img_name'] == image_training_name:
                            labeled_pixels.add(LabeledPixel((int(row['x']), int(row['y'])), row['label']))
                        row = next(reader, {'img_name': 'EOF'})

                    # Extract FeatureTable
                    feature_table = ImageFeaturesExtractor.extract_features(
                        join(join(self.train_image_path, 'img'), image_training_name) + '.tif',
                        self.selected_features, labeled_pixels, int(len(labeled_pixels) / len(self.class_names)))

                    # Save it as csv
                    feature_table.save_to_csv(self.train_features_path, image_training_name,
                                              feature_names)
        return

    def train(self, nb_samples_per_classes: int = 100, nb_samples_for_evaluation: int = 10) -> None:
        not_found = False
        for feature in self.selected_features:
            if feature not in self.extracted_features:
                not_found = True
        if not_found:
            raise EnvironmentError('Features are not extracted yet')

        nb_samples_per_classes_to_extract = nb_samples_per_classes + nb_samples_for_evaluation
        self.learning_data_set = ImageFeaturesExtractor.generate_training_base(self.train_features_path,
                                                                               self.training_image_names,
                                                                               nb_samples_per_classes_to_extract,
                                                                               len(self.class_names),
                                                                               self.selected_features)
        self.evaluate_data_set = ImageFeaturesExtractor.extract_validation_data_set_from_learning_data_set(
            nb_samples_for_evaluation, len(self.class_names), self.learning_data_set)
        labels, samples = self.learning_data_set.get_labels_and_samples()
        self.recognition_model.train(samples, labels)
        return

    def evaluate(self) -> Evaluation:
        labels, samples = self.evaluate_data_set.get_labels_and_samples()
        return self.recognition_model.evaluate_model(samples, labels)

    def predict(self, path_image: str, image_names: [str]) -> None:
        feature_names = list()
        for feature in self.selected_features:
            feature_names.append(feature.full_name)
        for image_name in image_names:
            filename, file_extension = splitext(image_name)
            if file_extension == '.tif':
                predict_feature_table = None
                try:
                    predict_feature_table = FeatureTable(join(self.predict_features_path, filename + '.csv'),
                                                         self.selected_features)
                except:
                    pass
                if predict_feature_table is None:
                    predict_feature_table = ImageFeaturesExtractor.extract_features(join(path_image, image_name),
                                                                                    self.selected_features)
                    predict_feature_table.save_to_csv(self.predict_features_path, filename, feature_names)
                prediction = Prediction(filename)
                for individual in predict_feature_table.individuals:
                    probabilities, labels = self.recognition_model.auto_predict([individual.sample])
                    prediction.add_prediction(PredictionPixel(individual.pixel, probabilities[0], labels[0]))
                if not path.exists(join(self.ml_predict_results_path, filename)):
                    makedirs(join(self.ml_predict_results_path, filename))
                prediction.save_to_csv(join(self.ml_predict_results_path, filename), filename + '.csv',
                                       self.class_names)
                self.generate_results(filename)
        return

    def generate_results(self, image_name: str) -> None:
        result_builders = [cls() for cls in ResultBuilder.__subclasses__()]
        for result_builder in result_builders:
            result_builder.generate_results(join(self.ml_predict_results_path, image_name), image_name,
                                            self.class_names, self.predict_image_path)
        return

    def extract_validation_data_set_from_learning_data_set(self, nb_samples_per_classes: int) -> FeatureTable:
        result = FeatureTable()
        class_names_found = list()
        nb_samples_per_class_found = list()
        nb_samples_required = (1 + len(self.class_names)) * nb_samples_per_classes

        while nb_samples_required != 0:
            individual_challenger = self.learning_data_set.pop_individual(
                random.randint(0, self.learning_data_set.size - 1))
            try:
                index = class_names_found.index(individual_challenger.label)
            except ValueError:
                class_names_found.append(individual_challenger.label)
                nb_samples_per_class_found.append(0)
                index = class_names_found.index(individual_challenger.label)

            if nb_samples_per_class_found[index] != nb_samples_per_classes:
                nb_samples_per_class_found[index] = nb_samples_per_class_found[index] + 1
                nb_samples_required = nb_samples_required - 1
                result.add_individual(individual_challenger)
            else:
                self.learning_data_set.add_individual(individual_challenger)
        return result

    def set_directory_path(self, directory_path: str) -> None:
        """Replace the directory path of the workspace and reinitialize the workspace.

        :param directory_path: The new directory path
        :type directory_path: str
        """
        self.directory_path = directory_path
        self.init()

    def set_selected_features(self, selected_feature: [DefinitionFeature]) -> None:
        self.selected_features = selected_feature
        return

    def find_features_already_extracted(self) -> [DefinitionFeature]:
        feature_names_found_per_csv_file = list()
        result = list()
        if len(self.training_image_names) != 0:
            try:

                for img_name in self.training_image_names:
                    with open(join(self.train_features_path, img_name + '.csv')) as csv_file:
                        reader = csv.DictReader(csv_file, delimiter=';')
                        feature_names_found_per_csv_file.append(reader.fieldnames[3:])

                extracted_feature_names = set(feature_names_found_per_csv_file[0]).intersection(
                    *feature_names_found_per_csv_file)
                for extracted_feature_name in extracted_feature_names:
                    result.append(DefinitionFeature(None, None, extracted_feature_name))
            except FileNotFoundError:
                return list()

        return result

    def auto_config(self, nb_samples_per_classes: int = 100, nb_samples_for_evaluation: int = 10) -> None:
        not_found = False
        for feature in self.selected_features:
            if feature not in self.extracted_features:
                not_found = True
        if not_found:
            raise EnvironmentError('Features are not extracted yet')

        nb_samples_per_classes_to_extract = nb_samples_per_classes + nb_samples_for_evaluation
        self.learning_data_set = ImageFeaturesExtractor.generate_training_base(self.train_features_path,
                                                                               self.training_image_names,
                                                                               nb_samples_per_classes_to_extract,
                                                                               len(self.class_names),
                                                                               self.selected_features)
        labels, samples = self.learning_data_set.get_labels_and_samples()
        if len(self.class_names) == 1:
            result = list()
            value_to_write = 0.1
            for value in labels:
                if value == 'C0':
                    result.append('C' + str(value_to_write))
                    value_to_write = 0.1 - value_to_write
                else:
                    result.append(value)
            labels = result
        self.recognition_model.auto_config(samples, labels)
        return
