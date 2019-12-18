import colorsys
import sys
import traceback

import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow

from program.model.featureextraction.image_features_extractor import ImageFeaturesExtractor
from program.controller.controller import Controller
from program.controller.worker import ExtractWorker, PredictWorker, TrainWorker, AutoConfigWorker, AutoSelectWorker
from program.model.datastructure.evaluation import Evaluation
from program.model.featureextraction.feature import DefinitionFeature
from program.model.workspace import Workspace
from program.view.evaluationView import EvaluationView
from program.view.featureView import UiFeatureView
from program.view.mainView import UiMainWindow
from program.view.setRootDirect import UiRootWindow


class SkyEye(Controller):
    def __init__(self) -> None:
        super().__init__()
        self.workspace = Workspace()
        self.main_view = None
        self.evaluate_view = None
        self.view_feature = None
        self.extract_worker = ExtractWorker(self.workspace)
        self.predict_worker = PredictWorker(self.workspace)
        self.train_worker = TrainWorker(self.workspace)
        self.auto_config_worker = AutoConfigWorker(self.workspace)
        self.auto_worker = AutoSelectWorker(self.workspace)

        app = QApplication(sys.argv)
        self.main_window = QMainWindow()
        self.set_root_view = UiRootWindow(self.main_window, self)
        self.set_root_view.setup_ui(self.main_window)
        self.main_window.show()
        sys.exit(app.exec_())

    def root_directory_validate(self, directory_path: str):
        if directory_path is None or len(directory_path) == 0:
            self.show_error('You forgot to choose a directory.',
                            'Please click on \"Choose\" to choose where you will work')
        else:
            try:
                self.workspace = Workspace(directory_path)
                self.workspace.init()
            except:
                self.show_error('Error during the workspace creation', details=traceback.format_exc())
            try:
                self.set_root_view.close_it()
                self.main_view = UiMainWindow(self.main_window, self, self.workspace)
                self.main_view.setup_ui(self.main_window, self.extract_worker, self.predict_worker, self.train_worker,
                                        self.auto_config_worker, self.auto_worker)
                self.main_window.show()
            except:
                self.show_error('Error during MainView creation', details=traceback.format_exc())

    def change_directory_path(self, directory_path: str):
        try:
            self.workspace = Workspace(directory_path)
            self.workspace.init()
        except:
            self.show_error('Error during the workspace creation', details=traceback.format_exc())
        try:
            self.main_view.update(self.workspace)
        except:
            self.show_error('Error during MainView update', details=traceback.format_exc())

    def update_workspace(self):
        try:
            self.workspace.update()
        except:
            self.show_error('Error during the workspace update', details=traceback.format_exc())
        try:
            self.main_view.update(self.workspace)
        except:
            self.show_error('Error during MainView update', details=traceback.format_exc())

    def extract_features(self, features: [DefinitionFeature]):
        self.workspace.set_selected_features(features)
        self.extract_worker.workspace = self.workspace
        try:
            self.extract_worker.start()
        except:
            self.show_error('Error during features extraction', details=traceback.format_exc())

    def train(self, nb_samples_per_classes: int, nb_samples_for_evaluation: int):
        try:
            self.workspace.recognition_model.set_parameters(self.main_view.get_parameters())
            self.train_worker.workspace = self.workspace
            self.train_worker.nb_samples_per_classes = nb_samples_per_classes
            self.train_worker.nb_samples_for_evaluation = nb_samples_for_evaluation
            self.train_worker.start()
        except EnvironmentError:
            self.show_error('The features are not extracted yet',
                            advice='Please go to the Feature tab and click on extract features')
        except:
            self.show_error('Error during model training', details=traceback.format_exc())

    def predict(self, path_files: [str]):
        if path_files is None or len(path_files) == 0:
            self.show_error('You forgot to choose images to predict.',
                            'Please click on \"Predict\" to choose the images to predict')

        else:
            self.predict_worker.workspace = self.workspace
            self.predict_worker.path_files = path_files
            try:
                self.workspace.recognition_model.set_parameters(self.main_view.get_parameters())
                self.predict_worker.start()
            except:
                self.show_error('Error during model training', details=traceback.format_exc())

    def evaluate(self):
        evaluation = None
        try:
            self.workspace.recognition_model.set_parameters(self.main_view.get_parameters())
            evaluation = self.workspace.evaluate()
        except:
            self.show_error('Error during evaluation', details=traceback.format_exc())
        try:
            evaluation_window = QMainWindow()
            self.evaluate_view = EvaluationView(evaluation_window, self)
            self.evaluate_view.setup_ui(evaluation_window, evaluation)
            evaluation_window.show()
        except:
            self.show_error('Error during EvaluationView creation', details=traceback.format_exc())

    def open(self, selected_file: str):
        try:
            self.workspace.recognition_model.load(selected_file)
            self.main_view.workspace = self.workspace
        except:
            self.show_error('Error during Model opening', details=traceback.format_exc())

    def save(self, file_name: str):
        try:
            self.workspace.recognition_model.save(file_name)
        except:
            self.show_error('Error during Model saving', details=traceback.format_exc())

    def save_evaluation(self, file_name: str, evaluation: Evaluation):
        evaluate_result_str = 'Recognition Rate: ' + str(
            evaluation.recognition_rate) + '\n\n' + evaluation.recognition_matrix_to_string()
        with open(file_name, 'w') as save_file:
            save_file.write(evaluate_result_str)

    def auto_config(self):
        try:
            self.workspace.recognition_model.set_parameters(self.main_view.get_parameters())
            self.auto_config_worker.workspace = self.workspace
            self.auto_config_worker.start()
        except EnvironmentError:
            self.show_error('The features are not extracted yet',
                            advice='Please go to the Feature tab and click on extract features')
        except:
            self.show_error('Error during model training', details=traceback.format_exc())

    def auto_select(self, nb_samples_per_classes: int, nb_samples_for_evaluation: int):
        try:
            self.workspace.recognition_model.set_parameters(self.main_view.get_parameters())
            self.auto_worker.workspace = self.workspace
            self.auto_worker.nb_samples_per_classes = nb_samples_per_classes
            self.auto_worker.nb_samples_for_evaluation = nb_samples_for_evaluation
            self.auto_worker.start()
        except EnvironmentError:
            self.show_error('The features are not extracted yet',
                            advice='Please go to the Feature tab and click on extract features')
        except:
            self.show_error('Error during auto selection', details=traceback.format_exc())

    def display_view_feature(self):
        try:
            feature_window = QMainWindow()
            self.view_feature = UiFeatureView(feature_window, self)
            self.view_feature.setup_ui(feature_window, self.workspace.extracted_features)
            feature_window.show()
        except:
            self.show_error('Error during EvaluationView creation', details=traceback.format_exc())

    def view(self, features: [DefinitionFeature]):
        if len(features) != 2:
            self.show_error('You have selected more than two features',
                            advice='Please select only two features into the list above and click on View')
        else:
            nb_classes = len(self.workspace.class_names)
            learning_data_set = ImageFeaturesExtractor.generate_training_base(self.workspace.train_features_path,
                                                                              self.workspace.training_image_names,
                                                                              100,
                                                                              nb_classes,
                                                                              features)

            labels, samples = learning_data_set.get_labels_and_samples()

            hsv_tuples = [(x * 1.0 / (nb_classes + 1), 0.75, 0.75) for x in range(nb_classes + 1)]
            rgb_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)
            colors = []
            for rgb_tuple in rgb_tuples:
                colors.append((rgb_tuple[0], rgb_tuple[1], rgb_tuple[2]))

            f1 = dict()
            f2 = dict()

            f1['C0'] = list()
            f2['C0'] = list()

            for i in range(nb_classes):
                f1[self.workspace.class_names[i]] = list()
                f2[self.workspace.class_names[i]] = list()

            for label_index in range(len(labels)):
                label = labels[label_index]
                sample = samples[label_index]
                f1[label].append(sample[0])
                f2[label].append(sample[1])

            plt.switch_backend('Qt5Agg')
            fig = plt.figure()
            ax = plt.subplot(111)

            try:
                ax.scatter(f1['C0'], f2['C0'], marker='o', c=colors[0], label='C0', edgecolors='white')
                for i in range(nb_classes):
                    ax.scatter(f1[self.workspace.class_names[i]], f2[self.workspace.class_names[i]], marker='o',
                               c=colors[i + 1], label=self.workspace.class_names[i],
                               edgecolors='white')
            except:
                self.show_error('Error during Graph creation', details=traceback.format_exc())

            plt.title('Feature Viewer')
            plt.xlabel(features[0].full_name)
            plt.ylabel(features[1].full_name)
            plt.grid(True)

            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            plt.show()

    def change_workspace_selected_features(self, result: [DefinitionFeature]):
        self.workspace.set_selected_features(result)


if __name__ == "__main__":
    sky_eye = SkyEye()
