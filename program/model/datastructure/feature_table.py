"""This module contains the required classes to define the result of a feature extraction."""
import csv
from os.path import join

from program.model.featureextraction.feature import DefinitionFeature


class Individual(object):
    """This class represents an object which we will use to train the model or predict it. It contains :

        * The label of the individual (for the training)
        * The list of all the values extracted
        * The individual's coordinates

    """

    def __init__(self, label: str, sample: [float], pixel: (int, int)) -> None:
        """Create an individual object.

        :param label: The label of the individual (for the training)
        :type label: str
        :param sample: The list of all the values extracted
        :type sample: [float]
        :param pixel: The individual's coordinates
        :type pixel: (int, int)
        """
        super().__init__()
        self.label = label
        self.sample = sample
        self.pixel = pixel


class FeatureTable(object):
    """This class represents an object which we will store in our CSV files. It contains a list of Individuals.
    """

    def __init__(self, csv_path: str = None, list_features: [DefinitionFeature] = None) -> None:
        """Create a FeatureTable object from a CSV file or an empty FeatureTable.

        :param csv_path: The path to the CSV file.
        :type csv_path: str or None
        :param list_features: A list of DefinitionFeature in order to extract from the CSV file only the required features
        :type list_features: [DefinitionFeature]
        """
        super().__init__()
        # Initialization
        self.individuals = list()

        # Test if we parse a CSV file or we just create an empty FeatureTable
        if csv_path is not None and list_features is not None:
            with open(csv_path) as csv_file:
                reader = csv.DictReader(csv_file, delimiter=';')
                nb_columns = len(reader.fieldnames)
                feature_need = len(list_features)
                feature_index = list()
                # We already know that the first three columns are Class, x and y. We want to loop on the other columns
                # in order to find the index of all the feature we will extract
                for i in range(3, nb_columns):
                    # If the current feature is required, we add it to the index list
                    if reader.fieldnames[i] in [feature.full_name for feature in list_features]:
                        feature_index.append(i)
                        feature_need = feature_need - 1
                # For each row, we create an individual with the right features and add it to the FeatureTable
                if feature_need != 0:
                    raise EnvironmentError()
                for row in reader:
                    sample = list()
                    for i in feature_index:
                        sample.append(float(row[reader.fieldnames[i]]))
                    self.add_individual(Individual(row['Class'], sample, (int(row['x']), int(row['y']))))

    @property
    def size(self) -> int:
        """Returns the number of Individuals in the FeatureTable.

        :return: The number of Individuals in the FeatureTable
        :rtype: int
        """
        return self.individuals.__len__()

    def add_individual(self, individual: Individual) -> None:
        """Add an individual to the FeatureTable's list.

        :param individual: The individual to add.
        :type individual: Individual
        :return: Nothing, the individual has been added to the FeatureTable's individual list.
        """
        self.individuals.append(individual)

    def pop_individual(self, position: int) -> Individual:
        """Delete and return an individual from the individuals list. This method will be used to generate
        a random data set by the image feature extractor.

        :param position: The index of the individual to pop.
        :type position: int
        :return: The right individual
        :rtype: Individual

        :except If the list is empty or the position is out of range, an IndexError will be thrown.
        """
        try:
            result = self.individuals.pop(position)
        except IndexError:
            raise IndexError("The index is out of range")
        return result

    def get_labels_and_samples(self):
        """Return the labels and samples in another format just to be used by our recognition models.

        :return: The labels and the samples in two separate lists.
        :rtype: ([str], [float])
        """
        labels = []
        samples = []

        for individual in self.individuals:
            labels.append(individual.label)
            samples.append(individual.sample)

        return labels, samples

    def save_to_csv(self, directory_path: str, file_name: str, feature_names: [str], nb_decimals: int = 2) -> None:
        """Save the FeatureTable into a CSV file with this structure :

        Class;x;y;MinD_9x9;NumD_9x9;NumW_9x9

        C0;1;4;1;144.81;9.44

        Talus-fosse;1;46;0;94.44;15.74

        C0;3;2;1;144.81;9.44

        :param directory_path: The directory where to store the CSV file.
        :type directory_path: str
        :param file_name: The name of the CSV file.
        :type file_name: str
        :param feature_names: The list of the name of the features in order to write it as headers.
        :type feature_names: [str]
        :param nb_decimals: The number of digits
        :type nb_decimals: int
        :return: Nothing, a CSV file has been stored in /directory_path/file_name.csv
        """
        # We construct the headers
        field_names = ['Class', 'x', 'y']
        feature_names.sort()
        for feature_name in feature_names:
            field_names.insert(len(field_names), feature_name)

        with open(join(directory_path, file_name + '.csv'), 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=field_names, delimiter=';')
            writer.writeheader()

            # We write on the CSV row by row
            for individual in self.individuals:
                row_to_write = {'Class': individual.label, 'x': individual.pixel[0], 'y': individual.pixel[1]}
                # We loop only on the features, so we remove 3 for Class, x and y.
                for i in range(len(field_names) - 3):
                    row_to_write[field_names[i + 3]] = round(individual.sample[i], nb_decimals)
                writer.writerow(row_to_write)

        return
