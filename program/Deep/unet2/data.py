from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import cv2
import warnings
from PIL import Image

warnings.filterwarnings("ignore")

BackGround = [255, 255, 255]
road = [0, 0, 0]
# COLOR_DICT = np.array([BackGround, road])
one = [128, 128, 128]
two = [128, 0, 0]
three = [192, 192, 128]
COLOR_DICT = np.array([one, two,three])

lab1 = [1.,1.,1.]
lab2 = [11, 11, 11]
lab3 = [21, 21, 21]
LABEL_DICT = np.array([lab1, lab2, lab3])

CLASS_WEIGHT = {0: 1.,
                1: 30.,
                2: 20.}

# Fusion of binary masks into a colored unique one
def maskFusion():
    for file in os.listdir("data/img"):

        # Masque charbonniere
        charboMaskArray = io.imread("data/charbonniere/" + file);

        # Masque talus
        talusMaskArray = io.imread("data/talus/" + file);

        # Nouveau masque
        newMask = Image.new('RGB', (400, 400), color="white") #
        newMaskArray = np.array(newMask)
        print(charboMaskArray.shape)

        DefautScale = [1,1,1]
        TalusScale = [11,11,11]
        CharbonniereScale = [22,22,22]

        # talus
        for x in range(charboMaskArray.shape[0]):  # Width
            if x >= 400:
                continue
            for y in range(charboMaskArray.shape[1]):  # Height
                if y >= 400:
                    continue
                if talusMaskArray[x, y] == False:  # Pixel noir
                    newMaskArray[x, y] = TalusScale
                else:
                    newMaskArray[x, y] = DefautScale

        # Charbonniere
        for x in range(charboMaskArray.shape[0]): # Width
            if x >= 400:
                continue
            for y in range(charboMaskArray.shape[1]): # Height
                if y >= 400:
                    continue
                if charboMaskArray[x,y] == False: # Pixel noir
                    newMaskArray[x,y] = CharbonniereScale

        # Création de l'image
        newImage = "data/classes/" + file.split(".")[0] + ".png"
        io.imsave(newImage, newMaskArray)
        print(newImage)
    print("Done !")

class data_preprocess:
    def __init__(self, train_path=None, image_folder=None, label_folder=None,
                 valid_path=None,valid_image_folder =None,valid_label_folder = None,
                 test_path=None, save_path=None,
                 img_rows=256, img_cols=256,
                 flag_multi_class=False,
                 num_classes = 2):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.train_path = train_path
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.valid_path = valid_path
        self.valid_image_folder = valid_image_folder
        self.valid_label_folder = valid_label_folder
        self.test_path = test_path
        self.save_path = save_path
        self.data_gen_args = dict(rotation_range=0.2,
                                  width_shift_range=0.05,
                                  height_shift_range=0.05,
                                  shear_range=0.05,
                                  zoom_range=0.05,
                                  vertical_flip=True,
                                  horizontal_flip=True,
                                  fill_mode='nearest')
        self.image_color_mode = "rgb"
        self.label_color_mode = "rgb"

        self.flag_multi_class = flag_multi_class
        self.num_class = num_classes
        self.target_size = (256, 256)
        self.img_type = 'png'

    def adjustData(self, img, label):
        weights = np.zeros(label[:,:,:,0].shape + (1,))
        if (self.flag_multi_class):
            img = img / 255.
            #label = label[:, :, :, 0] if (len(label.shape) == 4) else label[:, :, 0]
            #new_label = np.zeros(label.shape + (self.num_class,))
            #for i in range(self.num_class):
            #    new_label[label == i, i] = 1

            # new_mask = np.zeros(mask.shape + (num_class,))
            new_label = np.zeros(label.shape)

            ### for one pixel in the image, find the class in mask and convert it into one-hot vector
            for img_i in range(img.shape[0]):  # For each image of the batch
                for x_i in range(img.shape[1]):  # X
                    for y_i in range(img.shape[2]):  # Y
                        for i in range(len(COLOR_DICT)):
                            if (LABEL_DICT[i] == tuple(label[img_i][x_i][y_i])).all():
                                new_label[img_i][x_i][y_i][i] = 1
                                #weights[img_i][x_i][y_i][0] = CLASS_WEIGHT[i]
                                break
                        if np.count_nonzero(new_label[img_i][x_i][y_i]) != 1: # In case pixel is not recognized
                            new_label[img_i][x_i][y_i] = [1.,0.,0.]
                            #weights[img_i][x_i][y_i][0] = CLASS_WEIGHT[0]

            # Calc weights
            #for img in range(len(new_label)):
            #    for x in range(len(new_label[img])):
            #        for y in range(len(new_label[img][x])):
            #            for class_i in range(self.num_class):
            #                if new_label[img][x][y][0] == 1:
            #                    weights[img][x][y] = CLASS_WEIGHT[class_i]
            #                    print("Poids : " + weights[img][x][y])
            #                    break

            label = new_label
        elif (np.max(img) > 1):
            img = img / 255.
            label = label / 255.
            label[label > 0.5] = 1
            label[label <= 0.5] = 0
        return (img, label)

    def trainGenerator(self, batch_size, image_save_prefix="image", label_save_prefix="label",
                       save_to_dir=None, seed=7):
        '''
        can generate image and label at the same time
        use the same seed for image_datagen and label_datagen to ensure the transformation for image and label is the same
        if you want to visualize the results of generator, set save_to_dir = "your path"
        '''
        image_datagen = ImageDataGenerator(**self.data_gen_args)
        label_datagen = ImageDataGenerator(**self.data_gen_args)
        image_generator = image_datagen.flow_from_directory(
            self.train_path,
            classes=[self.image_folder],
            class_mode=None,
            color_mode=self.image_color_mode,
            target_size=self.target_size,
            batch_size=batch_size,
            save_to_dir=save_to_dir,
            save_prefix=image_save_prefix,
            seed=seed)
        label_generator = label_datagen.flow_from_directory(
            self.train_path,
            classes=[self.label_folder],
            class_mode=None,
            color_mode=self.label_color_mode,
            target_size=self.target_size,
            batch_size=batch_size,
            save_to_dir=save_to_dir,
            save_prefix=label_save_prefix,
            seed=seed)
        train_generator = zip(image_generator, label_generator)
        for (img, label) in train_generator:
            img, label = self.adjustData(img, label)
            yield (img, label)

    def testGenerator(self):
        filenames = os.listdir(self.test_path)
        for filename in filenames:
            img = io.imread(os.path.join(self.test_path, filename), as_gray=False)
            img = img / 255.
            img = trans.resize(img, self.target_size, mode='constant')
            img = np.reshape(img, img.shape + (1,)) if (not self.flag_multi_class) else img
            img = np.reshape(img, (1,) + img.shape)
            yield img

    def validLoad(self, batch_size,seed=7):
        image_datagen = ImageDataGenerator(**self.data_gen_args)
        label_datagen = ImageDataGenerator(**self.data_gen_args)
        image_generator = image_datagen.flow_from_directory(
            self.valid_path,
            classes=[self.valid_image_folder],
            class_mode=None,
            color_mode=self.image_color_mode,
            target_size=self.target_size,
            batch_size=batch_size,
            seed=seed)
        label_generator = label_datagen.flow_from_directory(
            self.valid_path,
            classes=[self.valid_label_folder],
            class_mode=None,
            color_mode=self.label_color_mode,
            target_size=self.target_size,
            batch_size=batch_size,
            seed=seed)
        train_generator = zip(image_generator, label_generator)
        for (img, label) in train_generator:
            img, label = self.adjustData(img, label)
            yield (img, label)
        # return imgs,labels

    def saveResult(self, npyfile, size, name,threshold=127):
        for i, item in enumerate(npyfile):
            img = item
            img_std = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            if self.flag_multi_class:
                for row in range(len(img)):
                    for col in range(len(img[row])):
                        num = np.argmax(img[row][col])
                        img_std[row][col] = COLOR_DICT[num]
            else:
                for k in range(len(img)):
                    for j in range(len(img[k])):
                        num = img[k][j]
                        if num < (threshold/255.0):
                            img_std[k][j] = road
                        else:
                            img_std[k][j] = BackGround
            img_std = cv2.resize(img_std, size, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(self.save_path, ("%s_predict." + self.img_type) % (name)), img_std)
