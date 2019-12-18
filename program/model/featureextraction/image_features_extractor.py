''' This module includes classes used for extracting features of an image or a folder of image.

'''
import random
from os.path import join

from program.model.datastructure.feature_table import FeatureTable, Individual
from program.model.datastructure.image import Image
from program.model.datastructure.labeled_pixel import LabeledPixel
from program.model.featureextraction.feature import DefinitionFeature, DefinitionFeatureEnum

all_mask_size = [3, 5, 9, 15, 19, 25, 35]
all_definition_features = set()
for definition_feature in DefinitionFeatureEnum:
    if definition_feature.need_mask_size:
        for mask_size in all_mask_size:
            all_definition_features.add(DefinitionFeature(definition_feature, mask_size))
    else:
        all_definition_features.add(DefinitionFeature(definition_feature))


class ImageFeaturesExtractor(object):
    """ This class contains functions to extract features of a given image.
    """
    @staticmethod
    def extract_features(image_path: str, features: [DefinitionFeature], labeled_pixels: [LabeledPixel] = None,
                         nb_random_pixel_to_add: int = 0) -> FeatureTable:
        """Extract some features from a given image.

        :param image_path: The path to the image
        :type image_path: str
        :param features: The list of Feature to compute
        :type features: [DefinitionFeature]
        :param labeled_pixels: The list of labeled pixels to use
        :type labeled_pixels: [LabeledPixel]
        :param nb_random_pixel_to_add: The number of random pixel we will add as trash class
        :type nb_random_pixel_to_add: int
        :return: The FeatureTable containing all the labeled pixels and trash pixels with their features extracted
        :rtype: FeatureTable
        """
        image = Image(image_path)
        # If there are not labeled pixel, we are predicting the image,
        # so we will extract all the pixels, so we retrieve them
        if labeled_pixels is None:
            labeled_pixels = set()
            for x in list(range(image.width)):
                for y in list(range(image.height)):
                    labeled_pixels.add(LabeledPixel((x, y), 'Unknown'))

        if nb_random_pixel_to_add > image.width * image.height - len(labeled_pixels):
            raise AttributeError(
                "There are not enough pixels to add " + str(nb_random_pixel_to_add) + " C0 pixels into the image.")

        # We retrieve the list of features to use
        feature_to_compute = set()
        features = sorted(list(features))
        for feature in features:
            feature_to_compute.add((feature.required_feature, feature.mask_size))

        # Add some random pixels
        i = 0
        while i < nb_random_pixel_to_add:
            x, y = random.randint(0, image.width - 1), random.randint(0, image.height - 1)
            old_size = len(labeled_pixels)
            labeled_pixels.add(LabeledPixel((x, y), 'C0'))
            i = i + len(labeled_pixels) - old_size
        result = FeatureTable()
        labeled_pixels = sorted(labeled_pixels)

        # For each labeled pixel, we extract the features and add an individual to the result
        for labeled_pixel in labeled_pixels:
            dictionary = dict()
            for feature in feature_to_compute:
                dictionary.update(feature[0].compute(labeled_pixel.pixel, image, feature[1]))
            sample = list()
            for feature in features:
                sample.append(dictionary[feature.full_name])
            result.add_individual(
                Individual(labeled_pixel.label, sample, (labeled_pixel.pixel[0], labeled_pixel.pixel[1])))
        return result

    @staticmethod
    def extract_all_features(image_path: str, labeled_pixels: [LabeledPixel],
                             nb_random_pixel_to_add: int) -> FeatureTable:
        """Extract all the possible features from a given image.

        :param image_path: The path to the image
        :type image_path: str
        :param labeled_pixels: The list of labeled pixels to use
        :type labeled_pixels: [LabeledPixel]
        :param nb_random_pixel_to_add: The number of random pixel we will add as trash class
        :type nb_random_pixel_to_add: int
        :return: The FeatureTable containing all the labeled pixels and trash pixels with their features extracted
        :rtype: FeatureTable
        """
        return ImageFeaturesExtractor.extract_features(image_path, all_definition_features, labeled_pixels,
                                                       nb_random_pixel_to_add)

    @staticmethod
    def generate_training_base(individuals_csv_files_path: str, file_names: [str], nb_samples_per_classes: int,
                               nb_classes: int, list_features: [DefinitionFeature]) -> FeatureTable:
        """Merge all the CSV generated from a feature extraction and create a data set from them.
        Each class will have the same amount of individuals chosen randomly.

        :param individuals_csv_files_path: The path to the CSV generated
        :type individuals_csv_files_path: str
        :param file_names: The names of CSV files
        :type file_names: [str]
        :param nb_samples_per_classes: The number of sample to extract for each class
        :type nb_samples_per_classes: int
        :param nb_classes: The number of classes
        :type nb_classes: int
        :param list_features: The list of features to extract
        :type list_features: [DefinitionFeature]
        :return: A data set which could be use for the training of our recognition model
        :rtype: FeatureTable
        """
        list_features = sorted(list(list_features))
        feature_table = FeatureTable()
        result = FeatureTable()
        class_names_found = list()
        nb_samples_per_class_found = list()
        nb_samples_required = (1 + nb_classes) * nb_samples_per_classes

        # We merge all the CSV file into a FeatureTable
        for file_name in file_names:
            feature_table_image = FeatureTable(join(individuals_csv_files_path, file_name + '.csv'), list_features)
            for individual in feature_table_image.individuals:
                feature_table.add_individual(individual)

        # Then, we pop some individuals until we have the right amount for each class
        while nb_samples_required != 0 and feature_table.size > 0:
            individual_challenger = feature_table.pop_individual(random.randint(0, feature_table.size - 1))
            try:
                index = class_names_found.index(individual_challenger.label)
            except ValueError:
                # If index not found, that means it's the first sample of this class,
                # so we create another element into our table
                class_names_found.append(individual_challenger.label)
                nb_samples_per_class_found.append(0)
                index = class_names_found.index(individual_challenger.label)

            # If we haven't the right number of samples in this class,
            # we add it and remove one to the number of elements to add
            if nb_samples_per_class_found[index] != nb_samples_per_classes:
                nb_samples_per_class_found[index] = nb_samples_per_class_found[index] + 1
                nb_samples_required = nb_samples_required - 1
                result.add_individual(individual_challenger)

        return result

    @staticmethod
    def extract_validation_data_set_from_learning_data_set(nb_samples_per_classes: int, nb_class_names: int,
                                                           learning_data_set: FeatureTable) -> FeatureTable:
        """Extract the validation data set from the learning data set

        :param nb_samples_per_classes: The number of sample to extract for each class
        :type nb_samples_per_classes: int
        :param nb_class_names: The number of classes
        :type nb_class_names: int
        :param learning_data_set: The learning data set
        :type learning_data_set: FeatureTable
        :return: A data set which could be use for the evaluation of our recognition model
        :rtype: FeatureTable
        """

        # Same algorithm than generate_training_base
        result = FeatureTable()
        class_names_found = list()
        nb_samples_per_class_found = list()
        nb_samples_required = (1 + nb_class_names) * nb_samples_per_classes

        while nb_samples_required != 0:
            individual_challenger = learning_data_set.pop_individual(
                random.randint(0, learning_data_set.size - 1))
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
                learning_data_set.add_individual(individual_challenger)
        return result
