"""This module contains the Controller class."""
from abc import ABCMeta, abstractmethod

from PyQt5.QtWidgets import QMessageBox

from program.model.datastructure.evaluation import Evaluation
from program.model.featureextraction.feature import DefinitionFeature


class Controller(metaclass=ABCMeta):
    """This class represent the object which will make the link between the view and the model"""
    @abstractmethod
    def root_directory_validate(self, directory_path: str):
        """The action when the workspace directory is defined. The workspace will be created and initialized.

        :param directory_path: The path to the workspace.
        :type directory_path: str
        """
        pass

    @abstractmethod
    def change_directory_path(self, directory_path: str):
        """The action when the workspace directory is changed. The workspace will be recreated.

        :param directory_path: The new workspace directory.
        :type directory_path: str
        """
        pass

    @abstractmethod
    def update_workspace(self):
        """The action when the update button is clicked. The workspace will be updated."""
        pass

    @abstractmethod
    def extract_features(self, features: [DefinitionFeature]):
        """The action to extract the features from the training images."""
        pass

    @abstractmethod
    def train(self, nb_samples_per_classes: int, nb_samples_for_evaluation: int):
        """Action to train the model."""
        pass

    @abstractmethod
    def predict(self, path_files: [str]):
        """The action to predict a list of images.

        :param path_files: The list of path to the images to predict.
        :type path_files: [str]
        """
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def open(self, selected_file: str):
        pass

    @abstractmethod
    def save(self, file_name: str):
        pass

    @abstractmethod
    def save_evaluation(self, file_name: str, evaluation: Evaluation):
        pass

    @abstractmethod
    def auto_config(self):
        pass

    @abstractmethod
    def auto_select(self, nb_samples_per_classes: int, nb_samples_for_evaluation: int):
        pass

    @abstractmethod
    def display_view_feature(self):
        pass

    @abstractmethod
    def view(self, features: [DefinitionFeature]):
        pass

    @abstractmethod
    def change_workspace_selected_features(self, result: [DefinitionFeature]):
        pass

    @staticmethod
    def show_error(message: str, advice: str = '', title: str = 'Error', details: str = ''):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(message)
        msg.setWindowTitle(title)

        if len(advice) != 0:
            msg.setInformativeText(advice)
        if len(details) != 0:
            msg.setDetailedText(details)

        msg.setStandardButtons(QMessageBox.Ok)
        msg.buttonClicked.connect(lambda: msg.close())
        msg.show()
