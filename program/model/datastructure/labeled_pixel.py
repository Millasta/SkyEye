"""This module contains the class LabeledPixel which define a labeled pixel from a training image.
"""


class LabeledPixel(object):
    """This class represent a row of our dictionary. This class will be used into sets,
    so we need to override all the methods to compare two LabeledPixel. It contains :

        * The coordinates of the pixel
        * The label of the pixel

    """

    def __init__(self, pixel: (int, int), label: str) -> None:
        """Create a labeled pixel object

        :param pixel: The coordinates of the pixel
        :type pixel: (int, int)
        :param label: The label of the pixel
        :type label: str
        """
        super().__init__()
        self.pixel = pixel
        self.label = label

    def __repr__(self) -> str:
        """Return a description of the object.

        (x y), label

        :return: A description of the object
        :rtype: str
        """
        return "(%s %s), %s" % (self.pixel[0], self.pixel[1], self.label)

    def __hash__(self) -> int:
        """Hash our object in order to obtain an id from our object.

        :return: The id of our object
        :rtype: int
        """
        return hash(self.__repr__())

    def __eq__(self, other):
        """Return if the object (self) is equals to the object other.

        :param other: The other object
        :return: If the objects are equals or not
        :rtype: bool
        """
        if isinstance(other, LabeledPixel):
            return self.pixel == other.pixel
        else:
            return False

    def __ne__(self, other):
        """Return if the object (self) is not equals to the object other.

        :param other: The other object
        :return: !self.__eq__(other)
        :rtype: bool
        """
        return self.pixel != other.pixel

    def __lt__(self, other):
        """Return if the object (self) is inferior to the object other.

        :param other: The other object
        :return: If the object (self) is inferior to the object other
        :rtype: bool
        """
        result = (self.pixel[0], self.pixel[1]) < (other.pixel[0], other.pixel[1])
        return result

    def __le__(self, other):
        """Return if the object (self) is inferior or equals to the object other.

        :param other: The other object
        :return: If the object (self) is inferior to the object other
        :rtype: bool
        """
        return (self.pixel[0], self.pixel[1]) <= (other.pixel[0], other.pixel[1])

    def __gt__(self, other):
        """Return if the object (self) is superior to the object other.

        :param other: The other object
        :return: If the object (self) is superior to the object other
        :rtype: bool
        """
        return (self.pixel[0], self.pixel[1]) > (other.pixel[0], other.pixel[1])

    def __ge__(self, other):
        """Return if the object (self) is superior or equals to the object other.

        :param other: The other object
        :return: If the object (self) is superior to the object other
        :rtype: bool
        """
        return (self.pixel[0], self.pixel[1]) >= (other.pixel[0], other.pixel[1])
