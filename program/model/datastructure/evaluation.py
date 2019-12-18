"""This module represents the result of an evaluation for our recognition models"""
import numpy


class Evaluation(object):
    """This class represents the result of an evaluation for our recognition models. It contains :

         * The recognition rate of the trained model
         * The recognition matrix with the correct answers vertically and the predicted results horizontally
    """
    def __init__(self, recognition_rate: float, confusion_matrix) -> None:
        """Create an evaluation object

        :param recognition_rate: The recognition rate of the trained model (between 0 and 1)
        :type recognition_rate: float
        :param confusion_matrix: The matrix with the correct answers vertically and the predicted results horizontally
        :type confusion_matrix: A numpy matrix
        """
        super().__init__()
        self.recognition_rate = recognition_rate
        self.confusion_matrix = confusion_matrix

    def recognition_matrix_to_string(self) -> str:
        """Returns a string which described the matrix in this format :

            [[1 2 3 4 5]

            [6 7 8 9 10]

            [11 12 13 14 15]

            [16 17 18 19 20]

            [21 22 23 24 25]]

        :return: The string which described the matrix
        :rtype: str
        """
        numpy.set_printoptions(formatter={'float': '{: 0.0f}'.format})
        return self.confusion_matrix.__str__()
