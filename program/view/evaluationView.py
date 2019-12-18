# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'evaluationView.ui'
#
# Created by: PyQt5 UI code generator 5.8
#
# WARNING! All changes made in this file will be lost!
from os.path import expanduser

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog

from program.controller.controller import Controller
from program.model.datastructure.evaluation import Evaluation


class EvaluationView(object):
    def __init__(self, main_window, controller: Controller = None) -> None:
        self.main_window = main_window
        self.central_widget = QtWidgets.QWidget(main_window)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.central_widget)
        self.groupBox_2 = QtWidgets.QGroupBox(self.central_widget)
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_2)
        self.recognition_rate_label = QtWidgets.QLabel(self.groupBox_2)
        self.groupBox = QtWidgets.QGroupBox(self.central_widget)
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox)
        self.matrix_label = QtWidgets.QLabel(self.groupBox)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.save_eval_button = QtWidgets.QPushButton(self.central_widget)
        self.save_model_button = QtWidgets.QPushButton(self.central_widget)
        self.exit_button = QtWidgets.QPushButton(self.central_widget)

        self.controller = controller
        self.evaluation = None

    def setup_ui(self, main_window, evaluation: Evaluation):
        self.evaluation = evaluation
        main_window.setObjectName("EvaluationView")
        main_window.resize(429, 217)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("Eye.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        main_window.setWindowIcon(icon)
        self.central_widget.setObjectName("centralwidget")
        self.verticalLayout.setObjectName("verticalLayout")
        font = QtGui.QFont()
        font.setPointSize(12)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_3.setObjectName("gridLayout_3")
        font = QtGui.QFont()
        font.setPointSize(28)
        self.recognition_rate_label.setFont(font)
        self.recognition_rate_label.setText("")
        self.recognition_rate_label.setAlignment(QtCore.Qt.AlignCenter)
        self.recognition_rate_label.setObjectName("recognition_rate_label")
        self.gridLayout_3.addWidget(self.recognition_rate_label, 0, 0, 1, 1)
        self.verticalLayout.addWidget(self.groupBox_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_2.setObjectName("gridLayout_2")
        spacer_item = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacer_item, 0, 0, 1, 1)
        self.matrix_label.setText("")
        self.matrix_label.setObjectName("matrix_label")
        self.gridLayout_2.addWidget(self.matrix_label, 0, 1, 1, 1)
        spacer_item1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacer_item1, 0, 2, 1, 1)
        self.verticalLayout.addWidget(self.groupBox)
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacer_item2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacer_item2)
        self.save_eval_button.setObjectName("save_eval_button")
        self.horizontalLayout.addWidget(self.save_eval_button)
        self.save_model_button.setObjectName("save_model_button")
        self.horizontalLayout.addWidget(self.save_model_button)
        self.exit_button.setObjectName("exit_button")
        self.horizontalLayout.addWidget(self.exit_button)
        self.verticalLayout.addLayout(self.horizontalLayout)
        main_window.setCentralWidget(self.central_widget)

        self.translate_ui(evaluation)
        self.bind_actions()
        QtCore.QMetaObject.connectSlotsByName(main_window)

    def translate_ui(self, evaluation: Evaluation):
        _translate = QtCore.QCoreApplication.translate
        self.main_window.setWindowTitle(_translate("EvaluationView", "Evaluation"))
        self.groupBox_2.setTitle(_translate("EvaluationView", "Recognition rate"))
        self.groupBox.setTitle(_translate("EvaluationView", "Confusion matrix"))
        self.save_eval_button.setText(_translate("EvaluationView", "Save evaluation"))
        self.save_model_button.setText(_translate("EvaluationView", "Save model"))
        self.exit_button.setText(_translate("EvaluationView", "Exit"))

        self.recognition_rate_label.setText(str(round(evaluation.recognition_rate*100, 2)) + '%')
        self.matrix_label.setText(evaluation.recognition_matrix_to_string())

    def bind_actions(self):
        self.exit_button.clicked.connect(lambda: self.exit())
        self.save_eval_button.clicked.connect(lambda: self.save_eval())
        self.save_model_button.clicked.connect(lambda: self.save_model())

    def exit(self):
        self.main_window.close()

    def save_model(self):
        name = QFileDialog.getSaveFileName(caption='Save Model', directory=expanduser("~"),
                                           filter='Recognition Model(*.rml)')
        self.controller.save(name[0])

    def save_eval(self):
        name = QFileDialog.getSaveFileName(caption='Save File', directory=expanduser("~"),
                                           filter='Text File(*.txt)')
        self.controller.save_evaluation(name[0], self.evaluation)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    my_main_window = QtWidgets.QMainWindow()
    ui = EvaluationView(my_main_window)
    ui.setup_ui(my_main_window)
    my_main_window.show()
    sys.exit(app.exec_())
