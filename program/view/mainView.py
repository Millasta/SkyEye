# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainView.ui'
#
# Created by: PyQt5 UI code generator 5.8
#
# WARNING! All changes made in this file will be lost!
from sys import maxsize

import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import QFileDialog, QAbstractItemView

from program.controller.controller import Controller
from program.controller.worker import ExtractWorker, PredictWorker, TrainWorker, AutoConfigWorker, AutoSelectWorker
from program.model.featureextraction.feature import DefinitionFeature
from program.model.featureextraction.image_features_extractor import all_definition_features
from program.model.workspace import Workspace


class UiMainWindow(object):
    def __init__(self, main_window, controller: Controller = None, workspace: Workspace = None) -> None:
        super().__init__()
        self.parameters_edit = list()
        self.central_widget = QtWidgets.QWidget(main_window)
        self.gridLayout = QtWidgets.QGridLayout(self.central_widget)
        self.tabWidget = QtWidgets.QTabWidget(self.central_widget)
        self.tab = QtWidgets.QWidget()
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab)
        self.directory_path_edit = QtWidgets.QLineEdit(self.tab)
        self.modify_path_button = QtWidgets.QPushButton(self.tab)
        self.update_workspace_button = QtWidgets.QPushButton(self.tab)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.label = QtWidgets.QLabel(self.tab)
        self.training_images_list_view = QtWidgets.QListView(self.tab)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.label_2 = QtWidgets.QLabel(self.tab)
        self.class_names_list_view = QtWidgets.QListView(self.tab)
        self.tab_2 = QtWidgets.QWidget()
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab_2)
        self.remove_feature_selected_button = QtWidgets.QPushButton(self.tab_2)
        self.auto_selection_button = QtWidgets.QPushButton(self.tab_2)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.label_4 = QtWidgets.QLabel(self.tab_2)
        self.selected_features_list_view = QtWidgets.QListView(self.tab_2)
        self.extraction_progress_bar = QtWidgets.QProgressBar(self.tab_2)
        self.auto_select_text_log = QtWidgets.QTextEdit(self.tab_2)
        self.view_features_button = QtWidgets.QPushButton(self.tab_2)
        self.add_selected_feature_button = QtWidgets.QPushButton(self.tab_2)
        self.extract_features_button = QtWidgets.QPushButton(self.tab_2)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.label_3 = QtWidgets.QLabel(self.tab_2)
        self.available_features_list_view = QtWidgets.QListView(self.tab_2)
        self.tab_3 = QtWidgets.QWidget()
        self.gridLayout_4 = QtWidgets.QGridLayout(self.tab_3)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.train_button = QtWidgets.QPushButton(self.tab_3)
        self.evaluate_button = QtWidgets.QPushButton(self.tab_3)
        self.predict_button = QtWidgets.QPushButton(self.tab_3)
        self.model_combo_box = QtWidgets.QComboBox(self.tab_3)
        self.parameters_group_box = QtWidgets.QGroupBox(self.tab_3)
        self.verticalLayout_parameters = QtWidgets.QVBoxLayout(self.parameters_group_box)
        self.train_progress_bar = QtWidgets.QProgressBar(self.tab_3)
        self.menu_bar = QtWidgets.QMenuBar(main_window)
        self.menuFile = QtWidgets.QMenu(self.menu_bar)
        self.status_bar = QtWidgets.QStatusBar(main_window)
        self.actionClean_Base = QtWidgets.QAction(main_window)
        self.actionSave_model = QtWidgets.QAction(main_window)
        self.actionOpen_model = QtWidgets.QAction(main_window)
        self.actionExit = QtWidgets.QAction(main_window)
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.nb_samples_train = QtWidgets.QSpinBox(self.tab_3)
        self.nb_samples_evaluation = QtWidgets.QSpinBox(self.tab_3)
        self.auto_config_button = QtWidgets.QPushButton(self.tab_3)
        self.label_5 = QtWidgets.QLabel(self.tab_3)
        self.label_6 = QtWidgets.QLabel(self.tab_3)

        self.model_class_names_list_view = QStandardItemModel(self.class_names_list_view)
        self.model_training_images_list_view = QStandardItemModel(self.training_images_list_view)
        self.model_available_features_list_view = QStandardItemModel(self.available_features_list_view)
        self.model_selected_features_list_view = QStandardItemModel(self.selected_features_list_view)

        self.selected_features_list_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.available_features_list_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.training_images_list_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.class_names_list_view.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.extract_worker = None
        self.predict_worker = None
        self.train_worker = None
        self.auto_config_worker = None
        self.auto_worker = None

        self.main_window = main_window
        self.controller = controller
        self.workspace = workspace

        self.model_is_loaded = False

    def setup_ui(self, main_window, extract_worker: ExtractWorker, predict_worker: PredictWorker,
                 train_worker: TrainWorker, auto_config_worker: AutoConfigWorker, auto_worker: AutoSelectWorker):
        main_window.setObjectName("MainWindow")
        main_window.resize(557, 432)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("Eye.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        main_window.setWindowIcon(icon)
        self.central_widget.setObjectName("centralwidget")
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget.setObjectName("tabWidget")
        self.tab.setObjectName("tab")
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.directory_path_edit.setObjectName("directory_path_edit")
        self.gridLayout_2.addWidget(self.directory_path_edit, 0, 0, 1, 1)
        self.modify_path_button.setObjectName("modify_path_button")
        self.gridLayout_2.addWidget(self.modify_path_button, 0, 1, 1, 1)
        self.update_workspace_button.setObjectName("update_workspace_button")
        self.gridLayout_2.addWidget(self.update_workspace_button, 0, 2, 1, 1)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.training_images_list_view.setObjectName("training_images_list_view")
        self.verticalLayout_2.addWidget(self.training_images_list_view)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_2.setObjectName("label_2")
        self.verticalLayout_3.addWidget(self.label_2)
        self.class_names_list_view.setObjectName("class_names_list_view")
        self.verticalLayout_3.addWidget(self.class_names_list_view)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        self.gridLayout_2.addLayout(self.horizontalLayout, 1, 0, 1, 3)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2.setObjectName("tab_2")
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.remove_feature_selected_button.setObjectName("remove_feature_selected_button")
        self.gridLayout_3.addWidget(self.remove_feature_selected_button, 1, 1, 1, 1)
        spacer_item = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacer_item, 0, 1, 1, 1)
        spacer_item1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacer_item1, 4, 1, 1, 1)
        self.auto_selection_button.setObjectName("auto_selection_button")
        self.gridLayout_3.addWidget(self.auto_selection_button, 3, 1, 1, 1)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.verticalLayout_parameters.setObjectName("verticalLayout_parameters")
        self.verticalLayout_parameters.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.setup_parameters()
        self.label_4.setObjectName("label_4")
        self.verticalLayout_5.addWidget(self.label_4)
        self.selected_features_list_view.setObjectName("selected_features_list_view")
        self.verticalLayout_5.addWidget(self.selected_features_list_view)
        self.gridLayout_3.addLayout(self.verticalLayout_5, 0, 2, 5, 1)
        self.extraction_progress_bar.setProperty("value", 0)
        self.extraction_progress_bar.setObjectName("extraction_progress_bar")
        self.gridLayout_3.addWidget(self.extraction_progress_bar, 6, 0, 1, 3)
        self.auto_select_text_log.setObjectName("auto_select_text_log")
        self.gridLayout_3.addWidget(self.auto_select_text_log, 8, 0, 1, 3)
        self.view_features_button.setObjectName("view_features_button")
        self.gridLayout_3.addWidget(self.view_features_button, 7, 1, 1, 1)
        self.add_selected_feature_button.setObjectName("add_selected_feature_button")
        self.gridLayout_3.addWidget(self.add_selected_feature_button, 2, 1, 1, 1)
        self.extract_features_button.setObjectName("extract_features_button")
        self.gridLayout_3.addWidget(self.extract_features_button, 5, 1, 1, 1)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_3.setObjectName("label_3")
        self.verticalLayout_4.addWidget(self.label_3)
        self.available_features_list_view.setObjectName("available_features_list_view")
        self.verticalLayout_4.addWidget(self.available_features_list_view)
        self.gridLayout_3.addLayout(self.verticalLayout_4, 0, 0, 5, 1)
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3.setObjectName("tab_3")
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.nb_samples_train.setObjectName("spinBox")
        self.gridLayout_5.addWidget(self.nb_samples_train, 0, 1, 1, 1)
        self.nb_samples_evaluation.setObjectName("spinBox_2")
        self.gridLayout_5.addWidget(self.nb_samples_evaluation, 1, 1, 1, 1)
        self.auto_config_button.setObjectName("pushButton_8")
        self.gridLayout_5.addWidget(self.auto_config_button, 0, 2, 2, 1)
        self.label_5.setObjectName("label_5")
        self.gridLayout_5.addWidget(self.label_5, 0, 0, 1, 1)
        self.label_6.setObjectName("label_6")
        self.gridLayout_5.addWidget(self.label_6, 1, 0, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_5, 2, 0, 1, 2)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.train_button.setObjectName("train_button")
        self.horizontalLayout_2.addWidget(self.train_button)
        self.evaluate_button.setObjectName("evaluate_button")
        self.horizontalLayout_2.addWidget(self.evaluate_button)
        self.predict_button.setObjectName("predict_button")
        self.horizontalLayout_2.addWidget(self.predict_button)
        self.gridLayout_4.addLayout(self.horizontalLayout_2, 4, 0, 1, 2)
        self.model_combo_box.setObjectName("model_combo_box")
        self.gridLayout_4.addWidget(self.model_combo_box, 0, 0, 1, 2)
        self.parameters_group_box.setObjectName("parameters_group_box")
        self.gridLayout_4.addWidget(self.parameters_group_box, 1, 0, 1, 2)
        self.train_progress_bar.setProperty("value", 0)
        self.train_progress_bar.setObjectName("train_progress_bar")
        self.gridLayout_4.addWidget(self.train_progress_bar, 5, 0, 1, 2)
        self.tabWidget.addTab(self.tab_3, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)
        main_window.setCentralWidget(self.central_widget)
        self.menu_bar.setGeometry(QtCore.QRect(0, 0, 557, 26))
        self.menu_bar.setObjectName("menu_bar")
        self.menuFile.setObjectName("menuFile")
        main_window.setMenuBar(self.menu_bar)
        self.status_bar.setObjectName("status_bar")
        main_window.setStatusBar(self.status_bar)
        self.actionClean_Base.setObjectName("actionClean_Base")
        self.actionSave_model.setObjectName("actionSave_model")
        self.actionOpen_model.setObjectName("actionOpen_model")
        self.actionExit.setObjectName("actionExit")
        self.menuFile.addAction(self.actionClean_Base)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionSave_model)
        self.menuFile.addAction(self.actionOpen_model)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menu_bar.addAction(self.menuFile.menuAction())

        self.evaluate_button.setEnabled(False)
        self.predict_button.setEnabled(False)
        self.train_button.setEnabled(False)

        self.actionSave_model.setEnabled(False)

        self.available_features_list_view.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.selected_features_list_view.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        self.directory_path_edit.setReadOnly(True)

        self.extraction_progress_bar.setVisible(False)
        self.train_progress_bar.setVisible(False)
        self.view_features_button.setEnabled(False)

        font = QtGui.QFont()
        font.setPointSize(12)
        self.auto_select_text_log.setFont(font)
        self.auto_select_text_log.setVisible(False)

        self.model_combo_box.addItem('SVM')

        self.extract_worker = extract_worker
        self.predict_worker = predict_worker
        self.train_worker = train_worker
        self.auto_config_worker = auto_config_worker
        self.auto_worker = auto_worker

        self.nb_samples_train.setMaximum(maxsize)
        self.nb_samples_train.setValue(500)
        self.nb_samples_evaluation.setMaximum(maxsize)
        self.nb_samples_evaluation.setValue(100)

        self.update(self.workspace)

        self.translate_ui()
        self.bind_actions()
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(main_window)

    def translate_ui(self):
        _translate = QtCore.QCoreApplication.translate
        self.main_window.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.modify_path_button.setText(_translate("MainWindow", "Modify"))
        self.update_workspace_button.setText(_translate("MainWindow", "Update"))
        self.label.setText(_translate("MainWindow", "Training images"))
        self.label_2.setText(_translate("MainWindow", "Classes"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Workspace"))
        self.remove_feature_selected_button.setText(_translate("MainWindow", "<<"))
        self.auto_selection_button.setText(_translate("MainWindow", "auto"))
        self.label_4.setText(_translate("MainWindow", "Selected features"))
        self.view_features_button.setText(_translate("MainWindow", "View features"))
        self.add_selected_feature_button.setText(_translate("MainWindow", ">>"))
        self.extract_features_button.setText(_translate("MainWindow", "Extracted features"))
        self.label_3.setText(_translate("MainWindow", "Available features"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Features"))
        self.train_button.setText(_translate("MainWindow", "Train"))
        self.evaluate_button.setText(_translate("MainWindow", "Evaluate"))
        self.predict_button.setText(_translate("MainWindow", "Predict"))
        self.parameters_group_box.setTitle(_translate("MainWindow", "Parameters"))
        self.auto_config_button.setText(_translate("MainWindow", "Auto config"))
        self.label_5.setText(_translate("MainWindow", "Number of samples per class for the training : "))
        self.label_6.setText(_translate("MainWindow", "Number of samples per class for the evaluation : "))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "RM"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionClean_Base.setText(_translate("MainWindow", "Clean Base"))
        self.actionSave_model.setText(_translate("MainWindow", "Save model"))
        self.actionOpen_model.setText(_translate("MainWindow", "Open model"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))

    def bind_actions(self):
        self.remove_feature_selected_button.clicked.connect(lambda: self.remove_feature())
        self.add_selected_feature_button.clicked.connect(lambda: self.add_feature())
        self.modify_path_button.clicked.connect(lambda: self.modify_path())
        self.update_workspace_button.clicked.connect(lambda: self.update_workspace())
        self.extract_features_button.clicked.connect(lambda: self.extract_features())
        self.train_button.clicked.connect(lambda: self.train())
        self.predict_button.clicked.connect(lambda: self.predict())
        self.evaluate_button.clicked.connect(lambda: self.evaluate())
        self.auto_config_button.clicked.connect(lambda: self.auto_config())
        self.auto_selection_button.clicked.connect(lambda: self.auto_select())
        self.view_features_button.clicked.connect(lambda: self.view_features())
        self.actionExit.triggered.connect(lambda: self.close_it())
        self.actionOpen_model.triggered.connect(lambda: self.open())
        self.actionSave_model.triggered.connect(lambda: self.save())

    def remove_feature(self):
        if len(self.selected_features_list_view.selectedIndexes()) >= 0:
            for index in [i.row() for i in self.selected_features_list_view.selectedIndexes()]:
                self.model_available_features_list_view.appendRow(
                    QStandardItem(self.model_selected_features_list_view.item(index)))
            for index in [i.row() for i in reversed(self.selected_features_list_view.selectedIndexes())]:
                self.model_selected_features_list_view.removeRow(index)

            self.model_available_features_list_view.sort(0)
            self.model_selected_features_list_view.sort(0)

    def add_feature(self):
        if len(self.available_features_list_view.selectedIndexes()) >= 0:
            for index in [i.row() for i in self.available_features_list_view.selectedIndexes()]:
                self.model_selected_features_list_view.appendRow(
                    QStandardItem(self.model_available_features_list_view.item(index)))
            for index in [i.row() for i in reversed(self.available_features_list_view.selectedIndexes())]:
                self.model_available_features_list_view.removeRow(index)

            self.model_available_features_list_view.sort(0)
            self.model_selected_features_list_view.sort(0)

    def modify_path(self):
        selected_directory = QFileDialog.getExistingDirectory(caption='Choose Root Directory',
                                                              directory='C:\\Users')
        self.controller.change_directory_path(selected_directory)

    def update(self, workspace: Workspace):
        self.workspace = workspace

        if self.workspace is not None:
            self.directory_path_edit.setText(self.workspace.directory_path)

            self.model_class_names_list_view.clear()
            self.model_training_images_list_view.clear()
            self.model_selected_features_list_view.clear()
            self.model_available_features_list_view.clear()

            for class_name in self.workspace.class_names:
                # Create an item with a caption
                item = QStandardItem(class_name)
                self.model_class_names_list_view.appendRow(item)

            self.model_class_names_list_view.sort(0)
            self.class_names_list_view.setModel(self.model_class_names_list_view)

            for image_name in self.workspace.training_image_names:
                # Create an item with a caption
                item = QStandardItem(image_name)
                self.model_training_images_list_view.appendRow(item)

            self.model_training_images_list_view.sort(0)
            self.training_images_list_view.setModel(self.model_training_images_list_view)

            for definition_feature in all_definition_features:
                # Create an item with a caption
                item = QStandardItem(definition_feature.full_name)
                if definition_feature in self.workspace.extracted_features:
                    self.model_selected_features_list_view.appendRow(item)
                else:
                    self.model_available_features_list_view.appendRow(item)

            self.model_selected_features_list_view.sort(0)
            self.model_available_features_list_view.sort(0)
            if self.model_selected_features_list_view.rowCount() > 0:
                self.train_button.setEnabled(True)
                self.view_features_button.setEnabled(True)
            self.available_features_list_view.setModel(self.model_available_features_list_view)
            self.selected_features_list_view.setModel(self.model_selected_features_list_view)

    def update_workspace(self):
        self.controller.update_workspace()

    def extract_features(self):
        list_features = set()
        for i in range(self.model_selected_features_list_view.rowCount()):
            data = self.model_selected_features_list_view.item(i)
            list_features.add(DefinitionFeature(feature_name=data.data(0)))
        self.extract_worker.updateProgress.connect(self.set_extraction_progress)
        self.status_bar.showMessage('The model is extracting the training images features, it may take a long time...')
        self.is_working()
        self.controller.extract_features(list_features)

    def extract_features_for_train(self):
        list_features = set()
        for i in range(self.model_selected_features_list_view.rowCount()):
            data = self.model_selected_features_list_view.item(i)
            list_features.add(DefinitionFeature(feature_name=data.data(0)))
        self.extract_worker.updateProgress.connect(self.set_extraction_progress_train)
        self.status_bar.showMessage(
            'The model is extracting the training images features, it may take a long time...')
        self.is_working()
        self.controller.extract_features(list_features)

    def train(self):
        self.extract_features_for_train()

    def predict(self):
        selected_files = QFileDialog.getOpenFileNames(caption='Choose images to predict',
                                                      directory=self.workspace.predict_image_path,
                                                      filter='Images TIFF(*.tif)')
        self.predict_worker.updateProgress.connect(self.set_prediction_progress)
        self.status_bar.showMessage('The model is predicting your images, it may take a long time...')
        if len(selected_files[0]) != 0:
            self.is_working()
        self.train_progress_bar.setValue(5)
        self.controller.predict(selected_files[0])

    def evaluate(self):
        self.controller.evaluate()

    def close_it(self):
        self.main_window.close()

    def set_extraction_progress(self, progress):
        self.extraction_progress_bar.setVisible(True)
        self.extraction_progress_bar.setValue(progress)
        if progress >= 100:
            self.extraction_progress_bar.setVisible(False)
            self.extraction_progress_bar.setValue(0)
            self.extract_features_button.setEnabled(True)
            self.train_button.setEnabled(True)
            self.modify_path_button.setEnabled(True)
            self.update_workspace_button.setEnabled(True)
            self.view_features_button.setEnabled(True)
            self.add_selected_feature_button.setEnabled(True)
            self.remove_feature_selected_button.setEnabled(True)
            self.auto_config_button.setEnabled(True)
            self.auto_selection_button.setEnabled(True)
            self.status_bar.showMessage('Extraction done !')

    def set_extraction_progress_train(self, progress):
        self.extraction_progress_bar.setVisible(True)
        self.extraction_progress_bar.setValue(progress)
        if progress >= 100:
            self.extraction_progress_bar.setVisible(False)
            self.extraction_progress_bar.setValue(0)
            self.extract_features_button.setEnabled(True)
            self.train_button.setEnabled(True)
            self.modify_path_button.setEnabled(True)
            self.update_workspace_button.setEnabled(True)
            self.view_features_button.setEnabled(True)
            self.add_selected_feature_button.setEnabled(True)
            self.remove_feature_selected_button.setEnabled(True)
            self.auto_config_button.setEnabled(True)
            self.auto_selection_button.setEnabled(True)
            self.status_bar.showMessage('The model is training itself, please wait...')
            self.is_working()
            self.train_worker.job_done.connect(self.train_done)
            self.controller.train(self.nb_samples_train.value(), self.nb_samples_evaluation.value())

    def set_prediction_progress(self, progress):
        self.train_progress_bar.setVisible(True)
        self.train_progress_bar.setValue(progress)
        if progress >= 100:
            self.train_progress_bar.setVisible(False)
            self.train_progress_bar.setValue(0)
            self.enable_all_buttons()
            self.status_bar.showMessage('Predictions done !')
            os.startfile(self.workspace.ml_predict_results_path, 'open')

    def train_done(self):
        self.status_bar.showMessage('Training done !')
        self.model_is_loaded = False
        self.enable_all_buttons()

    def is_working(self):
        self.actionOpen_model.setEnabled(False)
        self.actionSave_model.setEnabled(False)
        self.extract_features_button.setEnabled(False)
        self.evaluate_button.setEnabled(False)
        self.predict_button.setEnabled(False)
        self.train_button.setEnabled(False)
        self.modify_path_button.setEnabled(False)
        self.update_workspace_button.setEnabled(False)
        self.view_features_button.setEnabled(False)
        self.add_selected_feature_button.setEnabled(False)
        self.remove_feature_selected_button.setEnabled(False)
        self.auto_config_button.setEnabled(False)
        self.auto_selection_button.setEnabled(False)

    def enable_all_buttons(self):
        self.actionOpen_model.setEnabled(True)
        self.actionSave_model.setEnabled(True)
        self.extract_features_button.setEnabled(True)
        if not self.model_is_loaded:
            self.evaluate_button.setEnabled(True)
        self.predict_button.setEnabled(True)
        self.train_button.setEnabled(True)
        self.modify_path_button.setEnabled(True)
        self.update_workspace_button.setEnabled(True)
        self.view_features_button.setEnabled(True)
        self.add_selected_feature_button.setEnabled(True)
        self.remove_feature_selected_button.setEnabled(True)
        self.auto_config_button.setEnabled(True)
        self.auto_selection_button.setEnabled(True)

    def open(self):
        selected_file = QFileDialog.getOpenFileName(caption='Choose the model to use',
                                                    directory=self.workspace.ml_svm_path,
                                                    filter='Recognition Model(*.rml)')
        self.controller.open(selected_file[0])
        self.model_is_loaded = True
        self.enable_all_buttons()

    def save(self):
        name = QFileDialog.getSaveFileName(caption='Save File', directory=self.workspace.ml_svm_path,
                                           filter='Recognition Model(*.rml)')
        self.controller.save(name[0])

    def setup_parameters(self):
        self.parameters_edit = list()
        for i in range(len(self.workspace.recognition_model.parameters)):
            layout = QtWidgets.QHBoxLayout()
            self.verticalLayout_parameters.addLayout(layout)
            label = QtWidgets.QLabel()
            label.setText(self.workspace.recognition_model.parameters[i][0] + ' : ')
            text_edit = QtWidgets.QLineEdit()
            text_edit.setText(str(self.workspace.recognition_model.parameters[i][1]))
            self.parameters_edit.append(text_edit)
            layout.addWidget(label)
            layout.addWidget(text_edit)

    def get_parameters(self) -> {str, float}:
        result = {}
        for i in range(len(self.workspace.recognition_model.parameters)):
            result[self.workspace.recognition_model.parameters[i][0]] = float(self.parameters_edit[i].text())
        return result

    def auto_config(self):
        self.status_bar.showMessage('The model is searching the best parameters, please wait...')
        self.is_working()
        self.auto_config_worker.job_done.connect(self.auto_config_done)
        self.controller.auto_config()

    def auto_config_done(self):
        self.status_bar.showMessage('Search done !')
        for i in range(len(self.workspace.recognition_model.parameters)):
            text_edit = self.parameters_edit[i]
            text_edit.setText(str(self.workspace.recognition_model.parameters[i][1]))
        self.enable_all_buttons()

    def auto_select(self):
        self.is_working()
        self.status_bar.showMessage(
            'Before search the best features, we need to extract all available features from all the training images...')
        self.extract_worker.updateProgress.connect(self.set_extraction_progress_for_auto_select)
        self.controller.extract_features(set(all_definition_features))

    def auto_select_done(self, result: [DefinitionFeature]):
        self.status_bar.showMessage('Auto selection done !')
        self.model_selected_features_list_view.clear()
        self.model_available_features_list_view.clear()
        for definition_feature in all_definition_features:
            # Create an item with a caption
            item = QStandardItem(definition_feature.full_name)
            if definition_feature in result:
                self.model_selected_features_list_view.appendRow(item)
            else:
                self.model_available_features_list_view.appendRow(item)
        self.auto_select_text_log.setVisible(False)
        self.controller.change_workspace_selected_features(result)
        self.enable_all_buttons()

    def set_extraction_progress_for_auto_select(self, progress):
        self.set_extraction_progress(progress)
        if progress >= 100:
            self.is_working()
            self.status_bar.showMessage('The model is searching the best features to use, please wait...')
            self.auto_select_text_log.setVisible(True)
            self.auto_worker.job_done.connect(self.auto_select_done)
            self.auto_worker.log.connect(self.log)
            self.controller.auto_select(self.nb_samples_train.value(), self.nb_samples_evaluation.value())

    def log(self, log: str):
        text = self.auto_select_text_log.toPlainText()
        self.auto_select_text_log.setText(log + '\n' + text)

    def view_features(self):
        self.controller.display_view_feature()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    my_main_window = QtWidgets.QMainWindow()
    ui = UiMainWindow(my_main_window)
    ui.setup_ui(my_main_window)
    my_main_window.show()
    sys.exit(app.exec_())
