"""This module contains all the tools to exploit the model predictions

"""

import colorsys
import csv
from abc import ABCMeta, abstractmethod
from os import makedirs
from os.path import join, exists

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from cv2.cv2 import imwrite

from program.model.datastructure.image import Image


class ResultBuilder(metaclass=ABCMeta):
    """This abstract class will define all our result builders.
    It contains a name and a prediction.

    """

    def __init__(self, name: str) -> None:
        """Create a result builder object.

        :param name: The name of the result builder
        :type name: str
        """
        super().__init__()
        self.name = name
        self.prediction = None

    @abstractmethod
    def generate_results(self, result_path: str, image_name: str, class_names: [str], image_path: str) -> None:
        """Generate a result using the prediction.

        :param result_path: The path where to store the result
        :param image_name: The image name
        :param class_names: The list of class names
        :param image_path: The path to the image
        :return: Nothing, a result has been saved into the result_path. It name is (image_name + '_' + self.name)
        """
        return


class BinaryImageBuilder(ResultBuilder):
    def __init__(self) -> None:
        super().__init__('Bin')

    def generate_results(self, result_path: str, image_name: str, class_names: [str], image_path: str) -> None:
        """Generate a binary image for each class. The pixel will be black if the pixel was defined as
        an individual of this class.

        :param result_path: The path where to store the result
        :param image_name: The image name
        :param class_names: The list of class names
        :param image_path: The path to the image
        :return: Nothing, a result has been saved into the result_path. It name is (image_name+_+self.name+_+class_name)
        """
        super().generate_results(result_path, image_name, class_names, image_path)
        if not exists(join(result_path, self.name)):
            makedirs(join(result_path, self.name))
        image_source = Image(join(image_path, image_name + '.tif'))
        my_matrix = {}
        for class_name in class_names:
            my_matrix[class_name] = np.zeros((image_source.height + 1, image_source.width + 1), np.uint8)
            my_matrix[class_name].fill(255)
        with open(join(result_path, image_name + '.csv')) as csv_file:
            reader = csv.DictReader(csv_file, delimiter=';')
            for row in reader:
                pixel = (int(row['x']), int(row['y']))
                predicted_class = row['Class predicted']
                if predicted_class != 'C0':
                    my_matrix[predicted_class][pixel[1], pixel[0]] = 0

        for class_name in class_names:
            file_name = image_name + '_' + self.name + '_' + class_name
            imwrite(join(join(result_path, self.name), file_name + ".tif"), my_matrix[class_name])


class GrayImageBuilder(ResultBuilder):
    def __init__(self) -> None:
        super().__init__('Gray')

    def generate_results(self, result_path: str, image_name: str, class_names: [str], image_path: str) -> None:
        """Generate a gray scale image for each class. The pixel will darker if the pixel has a high probability to be
        an individual of this class.

        :param result_path: The path where to store the result
        :param image_name: The image name
        :param class_names: The list of class names
        :param image_path: The path to the image
        :return: Nothing, a result has been saved into the result_path. It name is (image_name+_+self.name+_+class_name)
        """
        super().generate_results(result_path, image_name, class_names, image_path)
        if not exists(join(result_path, self.name)):
            makedirs(join(result_path, self.name))
        image_source = Image(join(image_path, image_name + '.tif'))
        my_matrix = {}
        for class_name in class_names:
            my_matrix[class_name] = np.zeros((image_source.height + 1, image_source.width + 1), np.uint8)
            my_matrix[class_name].fill(255)
        with open(join(result_path, image_name + '.csv')) as csv_file:
            reader = csv.DictReader(csv_file, delimiter=';')
            nb_columns = len(reader.fieldnames)
            for row in reader:
                probabilities = list()
                pixel = (int(row['x']), int(row['y']))
                # We already know the first two columns are x and y
                for i in range(2, nb_columns - 1):
                    # We won't use a dictionary because we already know the probabilities
                    # are sorted by the class names
                    probabilities.append(float(row[reader.fieldnames[i]]))
                for class_name in class_names:
                    value = 255 - (probabilities[class_names.index(class_name) + 1] * 255)
                    my_matrix[class_name][pixel[1], pixel[0]] = value

        for class_name in class_names:
            file_name = image_name + '_' + self.name + '_' + class_name
            imwrite(join(join(result_path, self.name), file_name + ".tif"), my_matrix[class_name])


class ClassImageBuilder(ResultBuilder):
    def __init__(self) -> None:
        super().__init__('ClassImage')

    def generate_results(self, result_path: str, image_name: str, class_names: [str], image_path: str) -> None:
        """Generate a colored image with one color for each class.

        :param result_path: The path where to store the result
        :param image_name: The image name
        :param class_names: The list of class names
        :param image_path: The path to the image
        :return: Nothing, a result has been saved into the result_path. It name is (image_name+_+self.name)
        """
        super().generate_results(result_path, image_name, class_names, image_path)

        original_image = Image(join(image_path, image_name + '.tif'))
        image_matrix = np.zeros((original_image.height + 1, original_image.width + 1, 3), np.uint8)
        image_matrix.fill(0)

        nb_colors = len(class_names)
        hsv_tuples = [(x * 1.0 / nb_colors, 0.75, 0.75) for x in range(nb_colors)]
        rgb_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)
        colors = []
        rgb_colors = []
        for rgb_tuple in rgb_tuples:
            colors.append((int(rgb_tuple[2] * 255), int(rgb_tuple[1] * 255), int(rgb_tuple[0] * 255)))
            rgb_colors.append(
                (int(rgb_tuple[0] * 255) / 255, int(rgb_tuple[1] * 255) / 255, int(rgb_tuple[2] * 255) / 255))

        with open(join(result_path, image_name + '.csv')) as csv_file:
            reader = csv.DictReader(csv_file, delimiter=';')
            for row in reader:
                pixel = (int(row['x']), int(row['y']))
                predicted_class = row['Class predicted']
                if predicted_class != 'C0':
                    try:
                        index = class_names.index(predicted_class)
                        image_matrix[pixel[1], pixel[0]] = colors[index]
                    except ValueError:
                        pass

        file_name = image_name + '_' + self.name
        imwrite(join(result_path, file_name + ".tif"), image_matrix)


class ClassImageCompareBuilder(ResultBuilder):
    def __init__(self) -> None:
        super().__init__('ClassImageCompare')

    def generate_results(self, result_path: str, image_name: str, class_names: [str], image_path: str) -> None:
        """Generate a colored image with one color for each class.

        :param result_path: The path where to store the result
        :param image_name: The image name
        :param class_names: The list of class names
        :param image_path: The path to the image
        :return: Nothing, a result has been saved into the result_path. It name is (image_name+_+self.name)
        """
        super().generate_results(result_path, image_name, class_names, image_path)

        original_image = Image(join(image_path, image_name + '.tif'))
        image_matrix = np.zeros((original_image.height + 1, original_image.width + 1, 3), np.uint8)
        image_matrix.fill(0)

        nb_colors = len(class_names)
        hsv_tuples = [(x * 1.0 / nb_colors, 0.75, 0.75) for x in range(nb_colors)]
        rgb_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)
        colors = []
        rgb_colors = []
        for rgb_tuple in rgb_tuples:
            colors.append((int(rgb_tuple[0] * 255), int(rgb_tuple[1] * 255), int(rgb_tuple[2] * 255)))
            rgb_colors.append(
                (int(rgb_tuple[0] * 255) / 255, int(rgb_tuple[1] * 255) / 255, int(rgb_tuple[2] * 255) / 255))

        with open(join(result_path, image_name + '.csv')) as csv_file:
            reader = csv.DictReader(csv_file, delimiter=';')
            for row in reader:
                pixel = (int(row['x']), int(row['y']))
                predicted_class = row['Class predicted']
                if predicted_class != 'C0':
                    try:
                        index = class_names.index(predicted_class)
                        image_matrix[pixel[1], pixel[0]] = colors[index]
                    except ValueError:
                        pass
        plt.switch_backend('Agg')
        fig = plt.figure()

        fig.add_subplot(1, 3, 1)

        plt.imshow(original_image.img, cmap='gray')

        fig.add_subplot(1, 3, 2)
        plt.imshow(image_matrix)

        patch_list = []
        for i in range(len(class_names)):
            patch_list.append(mpatches.Patch(color=rgb_colors[i], label=class_names[i]))
        plt.legend(handles=patch_list, bbox_to_anchor=(1.3, 0.5), loc=6, borderaxespad=0.)

        file_name = image_name + '_' + self.name
        plt.savefig(join(result_path, file_name + ".tif"))
        plt.close()
