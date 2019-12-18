# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'featureView.ui'
#
# Created by: PyQt5 UI code generator 5.8
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QStandardItemModel, QStandardItem

from program.controller.controller import Controller
from program.model.featureextraction.feature import DefinitionFeature


class UiFeatureView(object):
    def __init__(self, main_window, controller: Controller = None) -> None:
        self.main_window = main_window
        self.central_widget = QtWidgets.QWidget(main_window)
        self.gridLayout = QtWidgets.QGridLayout(self.central_widget)
        self.view_button = QtWidgets.QPushButton(self.central_widget)
        self.groupBox = QtWidgets.QGroupBox(self.central_widget)
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox)
        self.feature_list_view = QtWidgets.QListView(self.groupBox)
        self.exit_button = QtWidgets.QPushButton(self.central_widget)
        self.label = QtWidgets.QLabel(self.central_widget)

        self.controller = controller
        self.model_available_features_list_view = QStandardItemModel(self.feature_list_view)

    def setup_ui(self, main_window, feature_list: [DefinitionFeature]):
        main_window.setObjectName("MainWindow")
        main_window.resize(311, 600)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("Eye.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        main_window.setWindowIcon(icon)
        self.central_widget.setObjectName("centralwidget")
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.gridLayout.setObjectName("gridLayout")
        spacer_item = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacer_item, 2, 0, 1, 1)
        self.view_button.setObjectName("view_button")
        self.gridLayout.addWidget(self.view_button, 2, 1, 1, 1)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.feature_list_view.setObjectName("feature_list_view")
        self.gridLayout_2.addWidget(self.feature_list_view, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.groupBox, 1, 0, 1, 3)
        self.exit_button.setObjectName("exit_button")
        self.gridLayout.addWidget(self.exit_button, 2, 2, 1, 1)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 3)
        main_window.setCentralWidget(self.central_widget)

        self.feature_list_view.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.feature_list_view.setModel(self.model_available_features_list_view)

        for definition_feature in feature_list:
            # Create an item with a caption
            item = QStandardItem(definition_feature.full_name)
            self.model_available_features_list_view.appendRow(item)

        self.model_available_features_list_view.sort(0)

        self.translate_ui()
        self.bind_actions()
        QtCore.QMetaObject.connectSlotsByName(main_window)

    def translate_ui(self):
        _translate = QtCore.QCoreApplication.translate
        self.main_window.setWindowTitle(_translate("MainWindow", "Feature View"))
        self.view_button.setText(_translate("MainWindow", "View"))
        self.groupBox.setTitle(_translate("MainWindow", "Features available"))
        self.exit_button.setText(_translate("MainWindow", "Exit"))
        self.label.setText(_translate("MainWindow", "Please select two features and click on \"View\""))

    def bind_actions(self):
        self.exit_button.clicked.connect(lambda: self.exit())
        self.view_button.clicked.connect(lambda: self.view_feature())

    def exit(self):
        self.main_window.close()

    def view_feature(self):
        list_features = list()
        for index in [i.row() for i in self.feature_list_view.selectedIndexes()]:
            data = self.model_available_features_list_view.item(index)
            list_features.append(DefinitionFeature(feature_name=data.data(0)))
        self.controller.view(list_features)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    my_main_window = QtWidgets.QMainWindow()
    ui = UiFeatureView(my_main_window)
    ui.setup_ui(my_main_window, set())
    my_main_window.show()
    sys.exit(app.exec_())
