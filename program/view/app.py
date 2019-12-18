# import sys
#
# from PyQt5.QtWidgets import (QApplication, QMainWindow)
#
# from program.model.machinelearning.machinelearning import SVM
# from program.view.functions import *
# from program.view.ui import CentralWindow
#
#
# class SkyEye(QMainWindow):
#     def __init__(self):
#         super(SkyEye, self).__init__()
#         self.central_widget = CentralWindow()
#         self.setCentralWidget(self.central_widget)
#         self.central_widget.classifieur = SVM()
#
#         self.setWindowTitle("SkyEye")
#         self.set_default_values()
#         self.bind_actions()
#         self.show()
#
#     def set_default_values(self):
#         self.central_widget.svm_c_param.setText(str(self.central_widget.classifieur.model.C))
#         self.central_widget.svm_gamma_param.setText(str(self.central_widget.classifieur.model.gamma))
#         self.central_widget.ml_train_samples_number_le.setText("100")
#         self.central_widget.svm_auto_samples_number_le.setText("100")
#         self.central_widget.ml_evaluate_recognition_rate_le.setText("0.00%")
#         self.central_widget.ml_evaluate_confusion_matrix_pte.insertPlainText("Confusion Matrix: ")
#         self.statusBar().showMessage("Warning: Root Dir not set yet!")
#
#     def bind_actions(self):
#         # set root dir
#         self.central_widget.choose_root_dir.clicked.connect(lambda: setting_root_dir(self.central_widget.root_dir_line,
#                                                                                      self.statusBar()))
#         # process images
#         self.central_widget.clean_base_btn.clicked.connect(lambda: process_images(self.statusBar()))
#
#         # extract features
#         self.central_widget.feature_train_btn.clicked.connect(lambda: extract_train_base_features(self.statusBar()))
#         self.central_widget.feature_predict_btn.clicked.connect(lambda: extract_predict_base_features(self.statusBar()))
#         self.central_widget.split_train_evaluate_btn.clicked.connect(lambda: init_train_evaluate_features_choosepanel(self.central_widget.train_evaluate_features_choosepanel, self.statusBar()))
#         self.central_widget.train_evaluate_features_choosepanel.btn_submit.clicked.connect(lambda: split_train_evaluate_features(self.central_widget.train_evaluate_features_choosepanel, self.statusBar()))
#         self.central_widget.view_features_choose_feature_btn.clicked.connect(lambda: init_feature_names_choosepanel(self.central_widget.view_features_choosepanel, self.statusBar()))
#         self.central_widget.view_features_choosepanel.submit_btn.clicked.connect(lambda: choose_features_to_view(self.central_widget.view_features_choosepanel,
#                                                                                                                         self.central_widget.view_features_feature1_le,
#                                                                                                                         self.central_widget.view_features_feature2_le,
#                                                                                                                         self.statusBar()))
#         self.central_widget.view_features_btn.clicked.connect(lambda: view_features(self.central_widget.view_features_sample_number_le, self.central_widget.view_features_feature1_le,
#                                                                                                                         self.central_widget.view_features_feature2_le, self.statusBar()))
#
#         # machine train-images
#         ## SVM
#         self.central_widget.svm_new_btn.clicked.connect(lambda: svm_new(self.central_widget.classifieur, self.central_widget.svm_c_param.text(),
#                                                          self.central_widget.svm_gamma_param.text()))
#         self.central_widget.svm_auto_config_btn.clicked.connect(lambda: svm_auto_config(self.central_widget.classifieur, self.central_widget.svm_c_param,
#                                                 self.central_widget.svm_gamma_param, self.statusBar(), int(self.central_widget.svm_auto_samples_number_le.text())))
#
#         self.central_widget.svm_load_btn.clicked.connect(lambda: svm_load(self.central_widget.classifieur, self.central_widget.svm_c_param,
#                                                          self.central_widget.svm_gamma_param))
#         self.central_widget.svm_save_btn.clicked.connect(lambda: svm_save(self.central_widget.classifieur))
#
#         ## ml
#         self.central_widget.ml_train_btn.clicked.connect(lambda: ml_train(self.central_widget.classifieur, self.statusBar(), int(self.central_widget.ml_train_samples_number_le.text())))
#         self.central_widget.ml_evaluate_btn.clicked.connect(lambda: ml_evaluate(self.central_widget.classifieur, self.statusBar(), self.central_widget.ml_evaluate_recognition_rate_le, self.central_widget.ml_evaluate_confusion_matrix_pte))
#         self.central_widget.ml_evaluate_result_save_btn.clicked.connect(lambda: ml_evaluate_save_result(self.central_widget.ml_evaluate_recognition_rate_le, self.central_widget.ml_evaluate_confusion_matrix_pte))
#
#         self.central_widget.ml_predict_btn.clicked.connect(lambda: ml_predict(self.central_widget.classifieur, self.statusBar()))
#
#         ## analyse results
#         # self.central_widget.btn_generate_class_image.clicked.connect(generate_class_image)
#
# def main():
#     # Just comment this line to not see anymore the debug into the terminal
#     logging.basicConfig(level=logging.INFO)
#
#     app = QApplication(sys.argv)
#     mywindow = SkyEye()
#
#     sys.exit(app.exec_())
#
# if __name__ == '__main__':
#     main()
