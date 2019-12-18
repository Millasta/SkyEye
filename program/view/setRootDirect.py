# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'setRootDirect.ui'
#
# Created by: PyQt5 UI code generator 5.8
#
# WARNING! All changes made in this file will be lost!
from os.path import expanduser

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMainWindow

from program.controller.controller import Controller


class UiRootWindow(object):
    def __init__(self, main_window, controller: Controller = None):
        super().__init__()
        self.main_window = main_window
        self.central_widget = QtWidgets.QWidget(self.main_window)
        self.gridLayout = QtWidgets.QGridLayout(self.central_widget)
        self.exit_button = QtWidgets.QPushButton(self.central_widget)
        self.choose_button = QtWidgets.QPushButton(self.central_widget)
        self.validate_button = QtWidgets.QPushButton(self.central_widget)
        self.directory_edit = QtWidgets.QLineEdit(self.central_widget)
        self.label = QtWidgets.QLabel(self.central_widget)

        self.selected_directory = ''
        self.controller = controller

    def setup_ui(self, main_window):
        main_window.setObjectName("MainWindow")
        main_window.resize(353, 108)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        size_policy.setHeightForWidth(main_window.sizePolicy().hasHeightForWidth())
        main_window.setSizePolicy(size_policy)
        main_window.setMinimumSize(QtCore.QSize(0, 0))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("Eye.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        main_window.setWindowIcon(icon)
        self.central_widget.setEnabled(True)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        size_policy.setHeightForWidth(self.central_widget.sizePolicy().hasHeightForWidth())
        self.central_widget.setSizePolicy(size_policy)
        self.central_widget.setAutoFillBackground(False)
        self.central_widget.setObjectName("centralWidget")
        self.gridLayout.setObjectName("gridLayout")
        spacer_item = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacer_item, 4, 0, 1, 1)
        self.exit_button.setObjectName("exitButton")
        self.gridLayout.addWidget(self.exit_button, 4, 3, 1, 1)
        self.choose_button.setObjectName("chooseButton")
        self.gridLayout.addWidget(self.choose_button, 1, 3, 1, 1)
        self.validate_button.setObjectName("validateButton")
        self.gridLayout.addWidget(self.validate_button, 4, 1, 1, 1)
        self.directory_edit.setObjectName("directoryEdit")
        self.gridLayout.addWidget(self.directory_edit, 1, 0, 1, 2)
        self.directory_edit.setReadOnly(True)
        self.label.setMaximumSize(QtCore.QSize(16777215, 16777196))
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 4)
        main_window.setCentralWidget(self.central_widget)

        self.translate_ui(main_window)
        self.bind_actions()
        QtCore.QMetaObject.connectSlotsByName(main_window)

    def translate_ui(self, main_window):
        _translate = QtCore.QCoreApplication.translate
        main_window.setWindowTitle(_translate("MainWindow", "Set Root Directory"))
        self.exit_button.setText(_translate("MainWindow", "Exit"))
        self.choose_button.setText(_translate("MainWindow", "Choose"))
        self.validate_button.setText(_translate("MainWindow", "Validate"))
        self.label.setText(_translate("MainWindow", "Please choose a directory where we will save your results"))

    def bind_actions(self):
        self.exit_button.clicked.connect(lambda: self.close_it())
        self.choose_button.clicked.connect(lambda: self.choose_folder())
        self.validate_button.clicked.connect(lambda: self.validate_it())

    def close_it(self):
        self.main_window.close()

    def choose_folder(self):
        self.selected_directory = QFileDialog.getExistingDirectory(caption='Choose Root Directory',
                                                                   directory=expanduser("~"))
        self.directory_edit.setText(self.selected_directory)

    def validate_it(self):
        self.controller.root_directory_validate(self.selected_directory)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    my_main_window = QMainWindow()
    ui = UiRootWindow(my_main_window)
    ui.setup_ui(my_main_window)
    my_main_window.show()
    sys.exit(app.exec_())
