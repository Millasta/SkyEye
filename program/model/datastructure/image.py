"""This module is a wrapper of image processing.

"""
from cv2.cv2 import imread, IMREAD_GRAYSCALE
import matplotlib.pyplot as plt


class Image:
    """ This class represents the **image** processed in the system which contains

        * file path for loading or saving the image
        * OpenCV image to manipulate
        * image information such as height, weight, size, etc

    """

    def __init__(self, image_file_path: str):
        """
        :param image_file_path: path to the image file
        :type image_file_path: str

        .. note:: img.shape returns

            * a tuple of 3 values if img is color : img.height(y), img.width(x) and img.channels .
            * a tuple of 2 values if img is gray scale : img.height(y), img.width(x) and img.channels .


        .. seealso:: `Basic Operations on OpenCV Images`_
        .. _Basic Operations on OpenCV Images : http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_core/py_basic_ops/py_basic_ops.html#accessing-image-properties

        """
        self.img = imread(image_file_path, IMREAD_GRAYSCALE)
        if self.img is None:
            raise AttributeError("Could not open or find the image : " + image_file_path)
        self.height, self.width = self.img.shape[0:2]
        self.size = self.img.size

    def get_pixel_gray_value(self, pixel: (int, int)):
        """ Return the gray value of a pixel
    
        :param pixel: the pixel
        :type pixel: (int, int)
        :return: the gray value of the pixel
        :rtype: int
    
        """
        (x, y) = pixel
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            assert AttributeError('Pixel Position is out of range!')
        else:
            return self.img.item(y, x)

    def get_rect(self, x: int, y: int, width: int, height: int):
        """ Return a rectangle area of image.
    
        :param x: horizon position of up-left starting point
        :type x: int
        :param y: vertical position of up-left starting point
        :type y: int
        :param width: the width of the rectangle
        :type width: int
        :param height: the height of the rectangle
        :type height: int
        :return: the rectangle area
        :rtype: OpenCV
    
        .. note:: If the Rect detects an edge, it will auto-reduce the size to reach the edge.
    
        .. todo:: Verify x, y, w, h are all valid numbers.
        """
        x_end = min(x + width, self.width)
        y_end = min(y + height, self.height)
        return self.img[y:y_end, x:x_end]

    def get_region_of_image(self, pixel: (int, int), mask_size: int):
        """ Return the NxN roi around the pixel.
    
        :param pixel: the pixel
        :type pixel: (int, int)
        :param mask_size: the size N
        :type mask_size: int
        :return: the Region Of Image
        :rtype: OpenCV Image
    
        .. note:: Keep the mark_size an odd number.
        """
        (x, y) = pixel

        x_start = max(x - (mask_size - 1) // 2, 0)
        y_start = max(y - (mask_size - 1) // 2, 0)

        return self.get_rect(x_start, y_start, mask_size, mask_size)

    def show(self) -> None:
        """ Show the image in a individual window powered by matplotlib.

        .. note::

            * Pointing at a pixel of the shown image, you can see its position and color values.

        .. seealso:: While displaying an image using **matplot**, multi-parameter is supported.
                     For more specific parameters, see `image tutorial of matplotlib
                     <http://matplotlib.org/users/image_tutorial.html>`_
        """
        if len(self.img.shape) == 2:
            plt.imshow(self.img, cmap='gray')
        else:
            plt.imshow(self.img)

    @staticmethod
    def show_single_image(image) -> None:
        """Shows the image into a window.

        :param image: The OpenCV image to show.
        :type image: OpenCV Image
        :return: Nothing, a window has been opened containing the image.
        """
        # If the image is a gray scale or a color image
        if len(image.shape) == 2:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)

    @staticmethod
    def compare_images(image_1, image_2) -> None:
        """Show two images in order to compare them.

        :param image_1: The first image to show
        :type image_1: OpenCV Image
        :param image_2: The second image to show
        :type image_2: OpenCV Image
        :return: Nothing, a new window has been opened containing the two images.
        """
        fig = plt.figure()

        fig.add_subplot(1, 2, 1)
        if len(image_1.shape) == 2:  # gray image
            plt.imshow(image_1, cmap='gray')
        else:  # color image
            plt.imshow(image_1)

        fig.add_subplot(1, 2, 2)
        if len(image_2.shape) == 2:
            plt.imshow(image_2, cmap='gray')
        else:
            plt.imshow(image_2)

        plt.show()
