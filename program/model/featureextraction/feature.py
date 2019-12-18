"""
This module contains all the classes to define our features.

It's a little tricky to define abstract base classes (ABCs) in Python,
see more details from https://docs.python.org/3.5/library/abc.html
"""
from abc import ABCMeta, abstractmethod
from enum import Enum

import cv2
import numpy as np

from copy import deepcopy


class Feature(metaclass=ABCMeta):
    """ This class represents an abstract base class for all features,
        in which common operations are defined, such as

        * construct a fullname as an id using its class and its mask size
        * compute the feature value of a given pixel
    """

    def __init__(self, name):
        """ A feature must be initialed with a name.

        :param name: name of the feature,  obligatory
        :type name: str
        """
        self.name = name

    @staticmethod
    def construct_full_name(name: str, mask_size: int):
        """Construct a unique full name using the subclass name and the mask_size like this : (name)_(size)x(size)

        :param name: The name of the Feature
        :type name: str
        :param mask_size: The mask_size of the feature
        :type mask_size: int
        :return: The fullname of the feature
        :rtype: str
        """
        if mask_size is None:
            return name
        else:
            return name + '_' + str(mask_size) + 'x' + str(mask_size)

    def __repr__(self) -> str:
        """Return a description of the object.

        Feature(name)

        :return: A description of the object
        :rtype: str
        """
        return "Feature(%s)" % self.name

    def __hash__(self) -> int:
        """Hash our object in order to obtain an id from our object.

        :return: The id of our object
        :rtype: int
        """
        return hash(self.__repr__())

    def __eq__(self, other: object) -> bool:
        """Return if the object (self) is equals to the object other.

        :param other: The other object
        :return: If the objects are equals or not
        :rtype: bool
        """
        if isinstance(other, Feature):
            return self.name == other.name
        else:
            return False

    def __str__(self) -> str:
        """ Return str(self).

        :return: A description of the object
        :rtype: str
        """
        return self.name

    @abstractmethod
    def compute(self, pixel: (int, int), img, mask_size: int = None) -> {str, float}:
        """ The core function to compute the the feature value of a given pixel.

        :param pixel: the pixel, obligatory
        :type pixel: (int, int)
        :param img: the image, obligatory
        :type img: OpenCV Image
        :param mask_size: the size of neighborhood of the pixel to compute feature value, optional
        :type mask_size: int
        :return: A dictionary containing the features computed.
        :rtype: {str, float}
        """
        pass


class GrayValue(Feature):
    """ This class represents a feature: the gray scale of the pixel.
    """

    def compute(self, pixel: (int, int), img, mask_size: int = None):
        """ The core function to compute the the feature value of a given pixel.

        :param pixel: the pixel, obligatory
        :type pixel: (int, int)
        :param img: the image, obligatory
        :type img: OpenCV Image
        :param mask_size: the size of neighborhood of the pixel to compute feature value, optional
        :type mask_size: int
        :return: A dictionary containing the features computed.
        :rtype: {str, float}
        """
        return dict([(self.name, img.get_pixel_gray_value(pixel))])


class MeanGrayValue(Feature):
    """ This class represents a feature : the mean gray value of the NxN neighbors around the pixel
    """

    def compute(self, pixel: (int, int), img, mask_size: int = None):
        """ The core function to compute the the feature value of a given pixel.

        :param pixel: the pixel, obligatory
        :type pixel: (int, int)
        :param img: the image, obligatory
        :type img: OpenCV Image
        :param mask_size: the size of neighborhood of the pixel to compute feature value, optional
        :type mask_size: int
        :return: A dictionary containing the features computed.
        :rtype: {str, float}

        :raise AttributeError: If the msk_size isn't an odd number
        """
        if mask_size is None or mask_size % 2 == 0:
            raise AttributeError('The mask_size must be an odd number')

        roi = img.get_region_of_image(pixel, mask_size)
        return dict([(self.construct_full_name(self.name, mask_size), cv2.mean(roi)[0])])


class CalcDifference(Feature):
    """ This feature returns the minimum of the differences between central value
        and neighborhood values in n * n mask.
    """

    def compute(self, pixel: (int, int), img, mask_size: int = None):
        """ The core function to compute the the feature value of a given pixel.

        :param pixel: the pixel, obligatory
        :type pixel: (int, int)
        :param img: the image, obligatory
        :type img: OpenCV Image
        :param mask_size: the size of neighborhood of the pixel to compute feature value, optional
        :type mask_size: int
        :return: A dictionary containing the features computed.
        :rtype: {str, float}

        :raise AttributeError: If the msk_size isn't an odd number
        """
        if mask_size is None or mask_size % 2 == 0:
            raise AttributeError('The mask_size must be an odd number')

        roi = img.get_region_of_image(pixel, mask_size)

        result = dict()
        if not roi.size == mask_size * mask_size:
            # maybe this is not the best solution to process corner pixel
            result[self.construct_full_name(MIN_CALC_DIFF_NAME, mask_size)] = 0
            result[self.construct_full_name(MAX_CALC_DIFF_NAME, mask_size)] = 0
            result[self.construct_full_name(AVG_CALC_DIFF_NAME, mask_size)] = 0
        else:
            central_value = roi[(mask_size - 1) // 2][(mask_size - 1) // 2]
            diff = list()
            for v in np.nditer(roi):
                diff.append(abs(int(central_value) - int(v)))
            # remove the central_value - central_value
            del diff[(mask_size * mask_size - 1) // 2]
            # return the min/max/avg of the differences.
            result[self.construct_full_name(MIN_CALC_DIFF_NAME, mask_size)] = min(diff)
            result[self.construct_full_name(MAX_CALC_DIFF_NAME, mask_size)] = max(diff)
            result[self.construct_full_name(AVG_CALC_DIFF_NAME, mask_size)] = sum(diff) / len(diff)
        return result


# the value of pixel
#
# [0, 64) : BLACK
# [64, 128) : DARK
# [128, 192) : FAIR
# [192, 255] : WHITE
DARK_LEVEL = 64
FAIR_LEVEL = 128
WHITE_LEVEL = 192


class GrayLevel(Feature):
    """ This feature returns the numbers of pixels in each level.

    """

    def compute(self, pixel: (int, int), img, mask_size: int = None):
        """ The core function to compute the the feature value of a given pixel.

        :param pixel: the pixel, obligatory
        :type pixel: (int, int)
        :param img: the image, obligatory
        :type img: OpenCV Image
        :param mask_size: the size of neighborhood of the pixel to compute feature value, optional
        :type mask_size: int
        :return: A dictionary containing the features computed.
        :rtype: {str, float}

        :raise AttributeError: If the msk_size isn't an odd number
        """
        if mask_size is None or mask_size % 2 == 0:
            raise AttributeError('The mask_size must be an odd number')

        roi = img.get_region_of_image(pixel, mask_size)

        num_black = 0
        num_dark = 0
        num_fair = 0
        num_white = 0

        for v in np.nditer(roi):
            if v < DARK_LEVEL:
                num_black += 1
            elif v < FAIR_LEVEL:
                num_dark += 1
            elif v < WHITE_LEVEL:
                num_fair += 1
            else:
                num_white += 1
        # return the number of pixels in black / dark / fair /  white levels.
        # the value is between [0, NxN]which should be normed to [0, 255]
        base = 255 / (mask_size * mask_size)
        # return num_black, num_dark, num_fair, num_white
        result = dict()
        result[self.construct_full_name(NUM_BLACK_NAME, mask_size)] = num_black * base
        result[self.construct_full_name(NUM_DARK_NAME, mask_size)] = num_dark * base
        result[self.construct_full_name(NUM_FAIR_NAME, mask_size)] = num_fair * base
        result[self.construct_full_name(NUM_WHITE_NAME, mask_size)] = num_white * base
        return result


class LocalBinaryPatterns(Feature):
    """ This feature represents the LBP.
    """

    def compute(self, pixel: (int, int), img, mask_size: int = None):

        """ The core function to compute the the feature value of a given pixel.

        :param pixel: the pixel, obligatory
        :type pixel: (int, int)
        :param img: the image, obligatory
        :type img: OpenCV Image
        :param mask_size: the size of neighborhood of the pixel to compute feature value, optional
        :type mask_size: int
        :return: A dictionary containing the features computed.
        :rtype: {str, float}

        :raise AttributeError: If the msk_size isn't an odd number
        """
        if mask_size is None or mask_size % 2 == 0:
            raise AttributeError('The mask_size must be an odd number')

        # We compute our number of points we will use depending the mask_size
        radius = (mask_size - 1) / 2
        nb_points = int(4 * (2 + (radius - 1) / 2))

        base_gray = img.get_pixel_gray_value(pixel)

        result_value = 0

        for i in range(nb_points):
            neighborhood_point = (pixel[0] + int(round(radius * np.cos(2 * np.pi * i / nb_points))),
                                  pixel[1] + int(round(radius * np.sin(2 * np.pi * i / nb_points))))
            # We try if the point exists, if not, we will just pass, maybe is not the best solution
            try:
                gray_value_neighborhood = img.get_pixel_gray_value(neighborhood_point)
                if gray_value_neighborhood is not None and gray_value_neighborhood >= base_gray:
                    result_value = result_value + pow(2, i)
            except AttributeError:
                pass
        result = dict()
        result[self.construct_full_name(self.name, mask_size)] = result_value
        return result


GRAY_NAME = 'Gray'
AVG_NAME = 'Avg'
MIN_CALC_DIFF_NAME = 'MinD'
MAX_CALC_DIFF_NAME = 'MaxD'
AVG_CALC_DIFF_NAME = 'AvgD'
NUM_BLACK_NAME = 'NumB'
NUM_DARK_NAME = 'NumD'
NUM_FAIR_NAME = 'NumF'
NUM_WHITE_NAME = 'NumW'
LOCAL_BINARY_PATTERN_NAME = 'LBP'

CALC_DIFF_NAME = 'CalcD'
GRAY_LEVEL_NAME = 'GrayL'


class DefinitionFeatureEnum(Enum):
    """This enumeration is used to retrieve all the possible features. A definition feature enumeration contains :

        * A name which must be unique
        * The feature to compute to retrieve his value
        * A boolean to say if you need a size or not

    """
    GRAY_NAME_ENUM = (GRAY_NAME, GrayValue(GRAY_NAME), False)
    AVG_NAME_ENUM = (AVG_NAME, MeanGrayValue(AVG_NAME), True)
    MIN_CALC_DIFF_NAME_ENUM = (MIN_CALC_DIFF_NAME, CalcDifference(CALC_DIFF_NAME), True)
    MAX_CALC_DIFF_NAME_ENUM = (MAX_CALC_DIFF_NAME, CalcDifference(CALC_DIFF_NAME), True)
    AVG_CALC_DIFF_NAME_ENUM = (AVG_CALC_DIFF_NAME, CalcDifference(CALC_DIFF_NAME), True)
    NUM_BLACK_NAME_ENUM = (NUM_BLACK_NAME, GrayLevel(GRAY_LEVEL_NAME), True)
    NUM_DARK_NAME_ENUM = (NUM_DARK_NAME, GrayLevel(GRAY_LEVEL_NAME), True)
    NUM_FAIR_NAME_ENUM = (NUM_FAIR_NAME, GrayLevel(GRAY_LEVEL_NAME), True)
    NUM_WHITE_NAME_ENUM = (NUM_WHITE_NAME, GrayLevel(GRAY_LEVEL_NAME), True)
    LOCAL_BINARY_PATTERN_NAME_ENUM = (LOCAL_BINARY_PATTERN_NAME, LocalBinaryPatterns(LOCAL_BINARY_PATTERN_NAME), True)

    @property
    def my_name(self):
        """Return the name of the feature

        :return: The name of the feature
        :rtype: str
        """
        return self.value[0]

    @property
    def required_feature(self):
        """Return the feature to use

        :return: The feature to use
        :rtype: Feature
        """
        return self.value[1]

    @property
    def need_mask_size(self):
        """Return if the feature needs a mask size or not

        :return: If the feature needs a mask size or not
        :rtype: bool
        """
        return self.value[2]


class DefinitionFeature(object):
    """This class represents a calculable feature. Indeed, some feature use the same class to compute itself
    (for example, te number of white pixels and dark pixels use the GrayLevel Feature)

    A definition feature contains :

        * A unique name
        * The feature to use to compute it
        * A mask_size
        * A full_name, compute from the name and the mask_size

    """
    def __init__(self, definition_enum: DefinitionFeatureEnum = None, mask_size: int = None,
                 feature_name: str = None) -> None:
        """Create a DefinitionFeature object. It can be created with a DefinitionFeatureEnum or just a feature_name.

        :param definition_enum: The definitionFeatureEnum to use to create it
        :type definition_enum: DefinitionFeatureEnum
        :param mask_size: The mask size
        :type mask_size: int
        :param feature_name: The name of the feature
        :type feature_name: str
        """
        super().__init__()
        if definition_enum is None and feature_name is None:
            raise AttributeError('You must a DefinitionFeatureEnum or a feature_name in the constructor')
        if feature_name is not None:
            parsed_name = feature_name.split('_')
            if len(parsed_name) == 2:
                mask_size = int((parsed_name[1].split('x'))[0])
            for definition_enum_loop in DefinitionFeatureEnum:
                if parsed_name[0] == definition_enum_loop.my_name:
                    definition_enum = deepcopy(definition_enum_loop)
        if mask_size is not None and not (mask_size % 2) == 1 and feature_name is None:
            raise AttributeError("The mask_size is not an odd number")
        self.name = definition_enum.my_name
        self.mask_size = mask_size
        self.required_feature = definition_enum.required_feature

    def __eq__(self, other: object) -> bool:
        """Return if the object (self) is equals to the object other.

        :param other: The other object
        :return: If the objects are equals or not
        :rtype: bool
        """
        if isinstance(other, DefinitionFeature):
            return self.full_name == other.full_name
        else:
            return False

    def __hash__(self) -> int:
        """Hash our object in order to obtain an id from our object.

        :return: The id of our object
        :rtype: int
        """
        return hash(self.full_name)

    def __lt__(self, other):
        return self.full_name < other.full_name

    @property
    def full_name(self) -> str:
        """Construct a unique full name using the name and the mask_size like this : (name)_(size)x(size)

        :return: The fullname of the feature
        :rtype: str
        """
        return Feature.construct_full_name(self.name, self.mask_size)
