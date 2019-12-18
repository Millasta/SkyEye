# Inherit from QThread
from os.path import split

from PyQt5 import QtCore

from program.controller.controller import Controller
from program.model.featureextraction.image_features_extractor import all_definition_features
from program.model.featureextraction.feature_selector import SequentialForwardFloatingSelection
from program.model.workspace import Workspace


class ExtractWorker(QtCore.QThread):
    # This is the signal that will be emitted during the processing.
    # By including int as an argument, it lets the signal know to expect
    # an integer argument when emitting.
    updateProgress = QtCore.pyqtSignal(int)

    # You can do any extra things in this init you need, but for this example
    # nothing else needs to be done expect call the super's init
    def __init__(self, workspace: Workspace):
        QtCore.QThread.__init__(self)
        self.workspace = workspace

    # A QThread is run by calling it's start() function, which calls this run()
    # function in it's own "thread".
    def run(self):
        value_to_add = 99 / float(len(self.workspace.training_image_names))
        current = 0
        self.updateProgress.emit(current)
        for image_name in self.workspace.training_image_names:
            self.workspace.extract_training_images_features(image_name)
            current = current + value_to_add
            self.updateProgress.emit(current)
        self.workspace.extracted_features = list(self.workspace.selected_features)
        self.updateProgress.emit(100)
        self.exit()


class PredictWorker(QtCore.QThread):
    # This is the signal that will be emitted during the processing.
    # By including int as an argument, it lets the signal know to expect
    # an integer argument when emitting.
    updateProgress = QtCore.pyqtSignal(int)

    # You can do any extra things in this init you need, but for this example
    # nothing else needs to be done expect call the super's init
    def __init__(self, workspace: Workspace):
        QtCore.QThread.__init__(self)
        self.workspace = workspace
        self.path_files = None

    # A QThread is run by calling it's start() function, which calls this run()
    # function in it's own "thread".
    def run(self):
        value_to_add = 99 / len(self.path_files)
        current = 0
        self.updateProgress.emit(current)
        for image_path in self.path_files:
            path, image_name = split(image_path)
            self.workspace.predict(path, [image_name])
            current = current + value_to_add
            self.updateProgress.emit(current)
        self.updateProgress.emit(100)
        self.exit()


class TrainWorker(QtCore.QThread):
    # This is the signal that will be emitted during the processing.
    # By including int as an argument, it lets the signal know to expect
    # an integer argument when emitting.
    job_done = QtCore.pyqtSignal()

    # You can do any extra things in this init you need, but for this example
    # nothing else needs to be done expect call the super's init
    def __init__(self, workspace: Workspace):
        QtCore.QThread.__init__(self)
        self.workspace = workspace
        self.nb_samples_per_classes = None
        self.nb_samples_for_evaluation = None

    # A QThread is run by calling it's start() function, which calls this run()
    # function in it's own "thread".
    def run(self):
        self.workspace.train(self.nb_samples_per_classes, self.nb_samples_for_evaluation)
        self.job_done.emit()
        self.exit()


class AutoConfigWorker(QtCore.QThread):
    # This is the signal that will be emitted during the processing.
    # By including int as an argument, it lets the signal know to expect
    # an integer argument when emitting.
    job_done = QtCore.pyqtSignal()

    # You can do any extra things in this init you need, but for this example
    # nothing else needs to be done expect call the super's init
    def __init__(self, workspace: Workspace):
        QtCore.QThread.__init__(self)
        self.workspace = workspace

    # A QThread is run by calling it's start() function, which calls this run()
    # function in it's own "thread".
    def run(self):
        self.workspace.auto_config()
        self.job_done.emit()
        self.exit()


class AutoSelectWorker(QtCore.QThread):
    # This is the signal that will be emitted during the processing.
    # By including int as an argument, it lets the signal know to expect
    # an integer argument when emitting.
    job_done = QtCore.pyqtSignal(list)
    log = QtCore.pyqtSignal(str)

    # You can do any extra things in this init you need, but for this example
    # nothing else needs to be done expect call the super's init
    def __init__(self, workspace: Workspace):
        QtCore.QThread.__init__(self)
        self.workspace = workspace
        self.nb_samples_per_classes = None
        self.nb_samples_for_evaluation = None

    # A QThread is run by calling it's start() function, which calls this run()
    # function in it's own "thread".
    def run(self):
        selector = SequentialForwardFloatingSelection(self.log)
        result = selector.search_best_features(self.workspace.recognition_model, self.workspace.train_features_path,
                                               self.workspace.training_image_names, len(self.workspace.class_names),
                                               set(all_definition_features),
                                               self.nb_samples_per_classes, self.nb_samples_for_evaluation)
        self.job_done.emit(list(result))
        self.exit()
