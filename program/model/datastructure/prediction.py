"""This module contains the definition of classes which will be used as the result of a prediction from
our recognition models"""
import csv
from os.path import join


class PredictionPixel(object):
    """This class will define a predicted pixel (after a prediction from the recognition model). It contains :

        * The coordinates of the pixel
        * A list of probabilities (0-1) to belong to each classes
        * The label of the predicted class max(probability)

    """
    def __init__(self, pixel: (int, int), probabilities: [float], label: str) -> None:
        """Create a PredictionPixel object

        :param pixel: The coordinates of the pixel
        :type pixel: (int, int)
        :param probabilities: The list of probabilities
        :type probabilities: [float]
        :param label: The class name predicted
        :type label: str
        """
        super().__init__()
        self.pixel = pixel
        self.probabilities = probabilities
        self.label = label


class Prediction(object):
    """This class represent the result of a prediction from the recognition model. It contains :

        * The image name
        * The path to the csv
        * The image's height
        * The image's width
        * A list of PredictedPixel

    """
    def __init__(self, image_name: str, csv_file_path: str = None) -> None:
        """Create a Prediction without any predicted pixels or parse the CSV file (used for result builders)

        :param image_name: The image name
        :type: str
        :param csv_file_path: The path to the CSV file
        :type csv_file_path: str
        """
        super().__init__()
        self.image_name = image_name
        self.prediction_pixels = set()
        self.height = 0
        self.width = 0

        # If a path is defined, we will parse it
        if csv_file_path is not None:
            with open(join(csv_file_path, image_name + '.csv')) as csv_file:
                reader = csv.DictReader(csv_file, delimiter=';')
                nb_columns = len(reader.fieldnames)
                for row in reader:
                    probabilities = list()
                    pixel = (int(row['x']), int(row['y']))
                    predicted_class = row['Class predicted']
                    # We already know the first two columns are x and y
                    for i in range(2, nb_columns - 1):
                        # We won't use a dictionary because we already know the probabilities
                        # are sorted by the class names
                        probabilities.append(float(row[reader.fieldnames[i]]))
                    self.prediction_pixels.add(PredictionPixel(pixel, probabilities, predicted_class))

    def add_prediction(self, prediction_pixel: PredictionPixel):
        """Add a predicted pixel to the list, we also use this function to retrieve the image size

        :param prediction_pixel: The predicted pixel to add
        :type prediction_pixel: PredictionPixel
        :return: Nothing, a predicted pixel has been added to the list
        """
        self.prediction_pixels.add(prediction_pixel)

        # We retrieve the max of all the predicted pixel into the list
        # in order to find the image's height and the image's width
        if prediction_pixel.pixel[0] > self.height:
            self.height = prediction_pixel.pixel[0]
        if prediction_pixel.pixel[1] > self.width:
            self.width = prediction_pixel.pixel[1]

    def save_to_csv(self, directory_path: str, file_name: str, class_names: [str], nb_decimals: int = 4) -> None:
        """Save the object into a csv file with this structure :

        x;y;C0;Chemin_creux;Fosse;Talus;Talus-fosse;Trame_actuelle;Class predicted

        81;46;0.0486;0.299;0.0036;0.485;0.1069;0.0569;Talus

        111;27;0.1639;0.3234;0.0007;0.4923;0.0115;0.0081;Talus

        111;26;0.1238;0.4041;0.006;0.428;0.0135;0.0246;Talus

        .. warning:: The predicted pixels will not be sorted

        :param directory_path: The path where to store the CSV file
        :type directory_path: str
        :param file_name: The file name to use
        :type file_name: str
        :param class_names: The class names to define the headers
        :type class_names: [str]
        :param nb_decimals: The number of digits
        :type nb_decimals: int
        :return: Nothing, the CSV file has been stored in the right directory
        """
        # We construct the headers
        field_names = ['x', 'y']
        my_class_names = list(class_names)
        my_class_names.append('C0')
        my_class_names = sorted(my_class_names)
        for class_name in my_class_names:
            field_names.insert(len(field_names), class_name)
        field_names.insert(len(field_names), 'Class predicted')

        with open(join(directory_path, file_name), 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=field_names, delimiter=';')
            writer.writeheader()

            # We write row by row into the CSV file
            for prediction_pixel in self.prediction_pixels:
                row_to_write = {'x': prediction_pixel.pixel[0], 'y': prediction_pixel.pixel[1]}
                for i in range(len(field_names) - 3):
                    try:
                        row_to_write[field_names[i + 2]] = round(prediction_pixel.probabilities[i], nb_decimals)
                    except IndexError:
                        raise IndexError('Index out of range')
                row_to_write['Class predicted'] = prediction_pixel.label
                writer.writerow(row_to_write)

        return
