# import sys
#
# from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QHBoxLayout, QPushButton, QGridLayout, QWidget,
#                              QCheckBox, QGroupBox, QLabel, QLineEdit, QFrame, QMainWindow, QPlainTextEdit)
#
#
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super(MainWindow, self).__init__()
#         self.setCentralWidget(CentralWindow())
#
#
# class CentralWindow(QWidget):
#     def __init__(self):
#         super(CentralWindow, self).__init__()
#
#         # set root dir
#         self.root_dir_line = QLineEdit("Not set yet ...")
#         self.choose_root_dir = QPushButton("Choose")
#
#         # process images
#         self.clean_base_btn = QPushButton("Clean Base")
#         # extract features
#         self.feature_train_btn = QPushButton("Train Base Images")
#         self.feature_evaluate_btn = QPushButton("Evaluate Base Images")
#         self.feature_predict_btn = QPushButton("Predict Base Images")
#         self.split_train_evaluate_btn = QPushButton("Split Train || Evaluate Features")
#         self.train_evaluate_features_choosepanel = TrainEvaluateFeatureChoosePanel()
#         ## view features
#         self.view_features_sample_number_le = QLineEdit("100")
#         self.view_features_choose_feature_btn = QPushButton("Choose 2 features")
#         self.view_features_feature1_le = QLineEdit("feature1")
#         self.view_features_feature2_le = QLineEdit("feature2")
#         self.view_features_btn = QPushButton("View \n Features")
#         self.view_features_choosepanel = FeatureChoosePanel("Choose 2 features")
#
#         # machine train-images
#         self.svm_c_param = QLineEdit()
#         self.svm_gamma_param = QLineEdit()
#
#         self.svm_new_btn = QPushButton("New")
#
#         self.svm_auto_samples_number_le = QLineEdit()
#         self.svm_auto_config_btn = QPushButton("Auto Config")
#
#         self.svm_load_btn = QPushButton("Load")
#         self.svm_save_btn = QPushButton("Save")
#
#         self.ml_train_btn = QPushButton("    Train   ")
#         self.ml_train_samples_number_le = QLineEdit()
#         self.ml_evaluate_btn = QPushButton("Evaluate")
#         self.ml_evaluate_recognition_rate_le = QLineEdit()
#         self.ml_evaluate_confusion_matrix_pte = QPlainTextEdit()
#         self.ml_evaluate_result_save_btn = QPushButton("Save Evaluate Result")
#         self.ml_predict_btn = QPushButton("Predict")
#
#         # analyse results
#         self.btn_generate_class_image = QPushButton("Generate Class Image")
#
#         self.init_ui()
#         self.set_init_value()
#
#     def init_ui(self):
#         # group 0 setting root dir
#         groupbox0 = QGroupBox()
#         groupbox0.setTitle("Set Root Directory")
#
#         h_layout0 = QHBoxLayout()
#         h_layout0.addWidget(QLabel("Root Dir"))
#         h_layout0.addWidget(self.root_dir_line)
#         h_layout0.addWidget(self.choose_root_dir)
#
#         groupbox0.setLayout(h_layout0)
#
#         # group 1
#         groupbox1 = QGroupBox()
#         groupbox1.setTitle("Process Images")
#
#         v_layout1 = QVBoxLayout()
#         groupbox1.setLayout(v_layout1)
#         v_layout1.addWidget(self.clean_base_btn)
#
#         # group 2
#         groupbox2 = QGroupBox()
#         groupbox2.setTitle("Extract Features")
#
#         v_layout2 = QVBoxLayout()
#         groupbox2.setLayout(v_layout2)
#
#         v_layout2.addWidget(self.feature_train_btn)
#         # v_layout2.addWidget(self.feature_evaluate_btn)
#         v_layout2.addWidget(self.feature_predict_btn)
#         v_layout2.addWidget(self.split_train_evaluate_btn)
#         v_layout2.addWidget(QHline())
#
#         ## view features
#         h_layout_view_feature_sample_number = QHBoxLayout()
#         h_layout_view_feature_sample_number.addWidget(QLabel("Samples:"))
#         h_layout_view_feature_sample_number.addWidget(self.view_features_sample_number_le)
#         h_layout_view_feature_sample_number.addWidget(QLabel("per class"))
#
#         h_layout_view_feature_feature_chooser = QHBoxLayout()
#         h_layout_view_feature_feature_chooser.addWidget(self.view_features_choose_feature_btn)
#         h_layout_view_feature_feature_chooser.addWidget(self.view_features_feature1_le)
#         h_layout_view_feature_feature_chooser.addWidget(self.view_features_feature2_le)
#
#         v_layout_view_feature_parametre = QVBoxLayout()
#         v_layout_view_feature_parametre.addLayout(h_layout_view_feature_sample_number)
#         v_layout_view_feature_parametre.addLayout(h_layout_view_feature_feature_chooser)
#
#         h_layout_view_features = QHBoxLayout()
#         h_layout_view_features.addLayout(v_layout_view_feature_parametre)
#         h_layout_view_features.addWidget(self.view_features_btn)
#
#         v_layout2.addLayout(h_layout_view_features)
#
#         # group 3
#         groupbox3 = QGroupBox()
#         groupbox3.setTitle("Machine Learning")
#
#         svm_c_label = QLabel("C:")
#         svm_gamma_label = QLabel("gamma:")
#
#         h_layout_svm_param = QHBoxLayout()
#         h_layout_svm_param.addWidget(svm_c_label)
#         h_layout_svm_param.addWidget(self.svm_c_param)
#         h_layout_svm_param.addWidget(svm_gamma_label)
#         h_layout_svm_param.addWidget(self.svm_gamma_param)
#         h_layout_svm_param.addWidget(self.svm_new_btn)
#
#         h_layout_svm_auto_config = QHBoxLayout()
#         h_layout_svm_auto_config.addWidget(QLabel("Samples:"))
#         h_layout_svm_auto_config.addWidget(self.svm_auto_samples_number_le)
#         h_layout_svm_auto_config.addWidget(QLabel(" per class          "))
#         h_layout_svm_auto_config.addWidget(self.svm_auto_config_btn)
#
#         h_layout_svm_load_save = QHBoxLayout()
#         h_layout_svm_load_save.addWidget(self.svm_load_btn)
#         h_layout_svm_load_save.addWidget(self.svm_save_btn)
#
#         v_layout_svm_actions = QVBoxLayout()
#         v_layout_svm_actions.addLayout(h_layout_svm_param)
#         v_layout_svm_actions.addWidget(QHline())
#         v_layout_svm_actions.addLayout(h_layout_svm_auto_config)
#         v_layout_svm_actions.addWidget(QHline())
#         v_layout_svm_actions.addLayout(h_layout_svm_load_save)
#
#         h_layout_svm = QHBoxLayout()
#         svm_label = QLabel("  SVM  ")
#         h_layout_svm.addWidget(svm_label)
#         h_layout_svm.addLayout(v_layout_svm_actions)
#
#         v_layout3 = QVBoxLayout()
#         v_layout3.addLayout(h_layout_svm)
#         v_layout3.addWidget(QHline())
#
#         h_layout_ml_train = QHBoxLayout()
#         h_layout_ml_train.addWidget(QLabel("Samples:"))
#         h_layout_ml_train.addWidget(self.ml_train_samples_number_le)
#         h_layout_ml_train.addWidget(QLabel(" per class          "))
#         h_layout_ml_train.addWidget(self.ml_train_btn)
#         v_layout3.addLayout(h_layout_ml_train)
#         v_layout3.addWidget(QHline())
#
#         h_layout_ml_evaluate = QHBoxLayout()
#         h_layout_ml_evaluate.addWidget(self.ml_evaluate_btn)
#
#         h_layout_ml_evaluate_reco_rate = QHBoxLayout()
#         h_layout_ml_evaluate_reco_rate.addWidget(QLabel("Recognition Rate: "))
#         h_layout_ml_evaluate_reco_rate.addWidget(self.ml_evaluate_recognition_rate_le)
#
#         v_layout_ml_evaluate_result = QVBoxLayout()
#         v_layout_ml_evaluate_result.addLayout(h_layout_ml_evaluate_reco_rate)
#         v_layout_ml_evaluate_result.addWidget(self.ml_evaluate_confusion_matrix_pte)
#         v_layout_ml_evaluate_result.addWidget(self.ml_evaluate_result_save_btn)
#         h_layout_ml_evaluate.addLayout(v_layout_ml_evaluate_result)
#
#         v_layout3.addLayout(h_layout_ml_evaluate)
#
#         v_layout3.addWidget(QHline())
#         v_layout3.addWidget(self.ml_predict_btn)
#
#         groupbox3.setLayout(v_layout3)
#
#         # visual results
#         groupbox4 = QGroupBox()
#         groupbox4.setTitle("Analyse Results")
#
#         v_layout4 = QVBoxLayout()
#         v_layout4.addWidget(self.btn_generate_class_image)
#
#         groupbox4.setLayout(v_layout4)
#
#         # global layout
#         v_layout = QVBoxLayout()
#         v_layout.addWidget(groupbox0)
#         v_layout.addWidget(groupbox1)
#         v_layout.addWidget(groupbox2)
#         v_layout.addWidget(groupbox3)
#         # v_layout.addWidget(groupbox4)
#
#         self.setLayout(v_layout)
#
#     def set_init_value(self):
#         self.root_dir_line.setReadOnly(True)
#         self.ml_evaluate_recognition_rate_le.setReadOnly(True)
#         self.ml_evaluate_confusion_matrix_pte.setReadOnly(True)
#         self.view_features_feature1_le.setReadOnly(True)
#         self.view_features_feature2_le.setReadOnly(True)
#
#
# class TrainEvaluateFeatureChoosePanel(QWidget):
#     def __init__(self):
#         super(TrainEvaluateFeatureChoosePanel, self).__init__()
#         self.label_train = QLabel("For Train...")
#         self.label_evaluate = QLabel("For Evaluate...")
#
#         self.checkboxes_train = list()
#         self.checkboxes_evaluate = list()
#
#         self.btn_submit = QPushButton("OK")
#
#     def init_ui(self, feature_file_names):
#         for i in range(len(feature_file_names)):
#             self.checkboxes_train.append(QCheckBox(feature_file_names[i]))
#             self.checkboxes_evaluate.append(QCheckBox(feature_file_names[i]))
#
#         v_layout_train = QVBoxLayout()
#         v_layout_train.addWidget(self.label_train)
#
#         v_layout_evaluate = QVBoxLayout()
#         v_layout_evaluate.addWidget(self.label_evaluate)
#
#         for i in range(len(self.checkboxes_train)):
#             v_layout_train.addWidget(self.checkboxes_train[i])
#             v_layout_evaluate.addWidget(self.checkboxes_evaluate[i])
#
#         h_layout_choices = QHBoxLayout()
#         h_layout_choices.addLayout(v_layout_train)
#         h_layout_choices.addLayout(v_layout_evaluate)
#
#         v_layout_global = QVBoxLayout()
#         v_layout_global.addLayout(h_layout_choices)
#         v_layout_global.addWidget(self.btn_submit)
#         self.setLayout(v_layout_global)
#
#
# class FeatureChoosePanel(QWidget):
#     def __init__(self, title="Choose features"):
#         super(FeatureChoosePanel, self).__init__()
#         self.setWindowTitle(title)
#         self.features_chx_group = list()
#         self.submit_btn = QPushButton("OK")
#
#     def init_ui(self, feature_names):
#         feature_number = len(feature_names)
#
#         for i in range(feature_number):
#             self.features_chx_group.append(QCheckBox(feature_names[i]))
#
#         # v_layout_feature_names = QVBoxLayout()
#         #
#         # for i in range(len(self.features_chx_group)):
#         #     v_layout_feature_names.addWidget(self.features_chx_group[i])
#         # self.setLayout(v_layout_feature_names)
#
#         grid_layout = QGridLayout()
#
#         for i in range(0, feature_number // 5 + 1):
#             for j in range(0, 5):
#                 index = i * 5 + j
#                 if index >= feature_number:
#                     break
#                 else:
#                     grid_layout.addWidget(self.features_chx_group[index], i, j)
#         grid_layout.addWidget(self.submit_btn, feature_number // 5, 4)
#         self.setLayout(grid_layout)
#
#
# class QHline(QFrame):
#     def __init__(self):
#         super(QHline, self).__init__()
#         self.setFrameShape(QFrame.HLine)
#         self.setFrameShadow(QFrame.Sunken)
#
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#
#     # mainwindow = MainWindow()
#     # mainwindow.show()
#
#     mywindow = CentralWindow()
#     mywindow.show()
#
#     # feature_file_names = ["z01.csv", "z02.csv", "z03.csv"]
#     # choosepanel = TrainEvaluateFeatureChoosePanel()
#     # choosepanel.init_ui(feature_file_names)
#     # choosepanel.show()
#
#     # feature_choose_panel = FeatureChoosePanel()
#     # feature_choose_panel.init_ui(FEATURE_NAMES)
#     # feature_choose_panel.show()
#
#     sys.exit(app.exec_())
