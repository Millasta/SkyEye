# """ This module represents the entrance of program. """
# import logging
# import os
# import shutil
# from time import gmtime, strftime
#
# import cv2
# from PyQt5.QtWidgets import QFileDialog
#
# from program.controller.skyeye import workspace
#
#
# # from program.view.view import TrainEvaluateFeatureChoosePanel
#
#
# def check_env():
#     """Check if python env and libraries needed are well installed."""
#     logging.info('Environment is ready...')
#
#
# def current_time():
#     return strftime("%Y-%m-%d %H:%M:%S", gmtime())
#
#
# def check_root():
#     """Check if the root folder is well placed"""
#     logging.info('Root folder is found...')
#
#
# def setting_root_dir(root_dir_line, statusbar):
#     ''' Set the root directory of the project and create all the directories required
#
#             :param root_dir_line: The text field of the path
#             :type root_dir_line: QLineEdit
#             :param statusbar: The status bar
#             :type statusbar: QStatusBar
#     '''
#
#     fn = QFileDialog.getExistingDirectory(None, "Choose Root Directory")
#
#     if len(fn) != 0:
#         workspace.set_directory_path(fn)
#         root_dir_line.setText(fn)
#         statusbar.showMessage("Root Dir is set. ")
#
#
# # processing images
# def process_images(statusbar):
#     statusbar.showMessage("Start cleaning files...")
#     # remove hidden system files or from macOS files
#     for root, subdirs, files in os.walk(workspace.directory_path):
#         for filename in files:
#             file_path = os.path.join(root, filename)
#             if filename.startswith('.'):
#                 os.unlink(file_path)
#
#     for root, subdirs, files in os.walk(workspace.train_image_path):
#         for filename in files:
#             image_file_path = os.path.join(root, filename)
#             if not filename.endswith(".tif"):
#                 os.unlink(image_file_path)
#             else:
#                 img = cv2.imread(image_file_path)
#                 if (img.shape is not None) and len(img.shape) == 3:
#                     cv2.imwrite(image_file_path, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
#
#     for f in os.listdir(workspace.predict_features_path):
#         if not f.endswith(".tif"):
#             os.unlink(os.path.join(workspace.predict_features_path, f))
#         else:
#             image_file_path = os.path.join(workspace.predict_features_path, f)
#             img = cv2.imread(image_file_path)
#             if (img.shape is not None) and len(img.shape) == 3:
#                 cv2.imwrite(image_file_path, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
#
#     statusbar.showMessage("Files are well cleaned.")
#
#
# # def extract_all_features(statusbar):
# #     """Extract all features for train-images-base images and predict-images images"""
# #     print('Start extracting all features for train-images-base images...')
# #
# #     learningImagesFeaturesExtractor = LearningImagesFeaturesExtractor()
# #     learningImagesFeaturesExtractor.compute_all_features()
# #     print('All features table for train-images-base images is ready.')
# #
# #     # print('Start extracting all features for predict-images images...')
# #     # testImageFeatureExtractor = TestImagesFeaturesExtractor()
# #     # testImageFeatureExtractor.compute_all_features()
# #     # print('All features table for predict-images images is ready.')
#
# def extract_train_base_features(statusbar):
#     """Extract all features for train-images-base images and predict-images images"""
#     logging.info('Start extracting all features for train-base images...')
#     statusbar.showMessage("Start extracting all features for train-base images... ")
#
#     workspace.extract_training_images_features()
#     logging.info('All features table for train-base images is ready.')
#     statusbar.showMessage("Extract Train Base Image Features - Done. ")
#
#
# def extract_predict_base_features(statusbar):
#     logging.info('Start extracting all features for predict images...')
#     statusbar.showMessage("Start extracting all features for predict images...")
#     testImageFeatureExtractor = TestImagesFeaturesExtractor()
#     testImageFeatureExtractor.compute_all_features()
#     logging.info('All features table for predict images is ready.')
#     statusbar.showMessage("Extract Predict Base Image Features - Done. ")
#
#
# def init_train_evaluate_features_choosepanel(panel, statusbar):
#     train_base_feature_files = list()
#
#     for f in os.listdir(workspace.train_features_path):
#         if os.path.isfile(os.path.join(workspace.train_features_path, f)) and (not f.startswith(".")) \
#                 and f.endswith(".csv"):
#             train_base_feature_files.append(f)
#
#     if len(train_base_feature_files) != 0:
#         # print(train_base_feature_files)
#         panel.init_ui(train_base_feature_files)
#         panel.show()
#
#     statusbar.showMessage("Initial feature files choose panel ... ")
#
#
# def split_train_evaluate_features(panel, statusbar):
#     for old_train_feature_file in os.listdir(workspace.ml_train_samples_path):
#         old_file_path = os.path.join(workspace.ml_train_samples_path, old_train_feature_file)
#         try:
#             if os.path.isfile(old_file_path):
#                 os.unlink(old_file_path)
#         except Exception as e:
#             print(e)
#
#     for old_evaluate_feature_file in os.listdir(workspace.ml_evaluate_samples_path):
#         old_file_path = os.path.join(workspace.ml_evaluate_samples_path, old_evaluate_feature_file)
#         try:
#             if os.path.isfile(old_file_path):
#                 os.unlink(old_file_path)
#         except Exception as e:
#             print(e)
#
#     for chx_train_feature_file in panel.checkboxes_train:
#         if chx_train_feature_file.isChecked():
#             try:
#                 old_file_path = os.path.join(workspace.train_features_path, chx_train_feature_file.text())
#                 new_file_path = os.path.join(workspace.ml_train_samples_path, chx_train_feature_file.text())
#                 if os.path.isfile(old_file_path):
#                     shutil.copyfile(old_file_path, new_file_path)
#             except Exception as e:
#                 print(e)
#
#     for chx_evaluate_feature_file in panel.checkboxes_evaluate:
#         if chx_evaluate_feature_file.isChecked():
#             try:
#                 old_file_path = os.path.join(workspace.train_features_path, chx_evaluate_feature_file.text())
#                 new_file_path = os.path.join(workspace.ml_evaluate_samples_path, chx_evaluate_feature_file.text())
#                 if os.path.isfile(old_file_path):
#                     shutil.copyfile(old_file_path, new_file_path)
#             except Exception as e:
#                 print(e)
#
#     panel.close()
#
#     statusbar.showMessage("Feature files have been split into ml/train and ml/evaluate.")
#
#
# def init_feature_names_choosepanel(panel, statusbar):
#     panel.init_ui(FEATURE_NAMES)
#     panel.show()
#
#     statusbar.showMessage("Initial feature names choose panel ... ")
#
#
# def choose_features_to_view(panel, feature1_le, feature2_le, statusbar):
#     chosen_feature_names = list()
#     for chx_feature_name in panel.features_chx_group:
#         if chx_feature_name.isChecked():
#             chosen_feature_names.append(chx_feature_name.text())
#
#     if len(chosen_feature_names) >= 2:
#         feature1_le.setText(chosen_feature_names[0])
#         feature2_le.setText(chosen_feature_names[1])
#     panel.close()
#     statusbar.showMessage("2 features have been chosen. ")
#
#
# def view_features(num_le, feature1_le, feature2_le, statusbar):
#     statusbar.showMessage("Start generating features view samples... ")
#     sample_number = int(num_le.text())
#     feature_names = list()
#     feature_names.append(feature1_le.text())
#     feature_names.append(feature2_le.text())
#
#     # load samples
#     learningImagesFeaturesExtractor = LearningImagesFeaturesExtractor()
#     all_samples, all_labels = learningImagesFeaturesExtractor.load_all_learning_source(
#         workspace.ml_train_samples_path())
#     samples, labels = learningImagesFeaturesExtractor.generate_learning_samples(all_samples, all_labels,
#                                                                                 num_limits_each_label=sample_number)
#
#     # choose features
#     samples_new = FeatureSelection.select_features(samples, feature_names)
#
#     # view features
#     FeatureSelection.feature_view(samples_new, labels, feature1_le.text(), feature2_le.text())
#
#     statusbar.showMessage("View Features- Done")
#
#
# ###
# ### SVM
# ###
#
# def svm_new(model, c_str, gamma_str):
#     model.model.C = float(c_str)
#     model.model.gamma = float(gamma_str)
#
#
# def svm_auto_config(model, svm_c_le, svm_gamma_le, statusbar, number_each_class=100):
#     learningImagesFeaturesExtractor = LearningImagesFeaturesExtractor()
#
#     statusbar.showMessage("Start searching best C and gamma...")
#
#     logging.info('Start generating train samples...')
#     all_samples, all_labels = learningImagesFeaturesExtractor.load_all_learning_source(
#         workspace.ml_train_samples_path)
#     samples_train, labels_train = learningImagesFeaturesExtractor.generate_learning_samples(all_samples, all_labels,
#                                                                                             num_limits_each_label=number_each_class)
#
#     result = model.grid_search_c_gamma(samples_train, labels_train)
#     best_C = result.get("C")
#     best_gamma = result.get("gamma")
#
#     svm_c_le.setText(str(best_C))
#     svm_gamma_le.setText(str(best_gamma))
#
#     statusbar.showMessage("Auto Config SVM - Done!")
#
#
# def svm_load(model, c_editor, gamma_editor):
#     fn = QFileDialog.getOpenFileName(None, "Load SVM from file", workspace.ml_svm_path)
#     model.load(fn[0])
#     c_editor.setText(str(model.model.C))
#     gamma_editor.setText(str(model.model.gamma))
#
#
# def svm_save(model):
#     fn = QFileDialog.getSaveFileName(None, "Save SVM to file", workspace.ml_svm_path)
#     model.save(fn[0])
#
#
# # train
# def ml_train(model, statusbar, number_each_class=1000):
#     learningImagesFeaturesExtractor = LearningImagesFeaturesExtractor()
#
#     logging.info('Start generating train samples...')
#     statusbar.showMessage("Start generating train samples... ")
#     all_samples, all_labels = learningImagesFeaturesExtractor.load_all_learning_source(
#         workspace.ml_train_samples_path)
#     samples_train, labels_train = learningImagesFeaturesExtractor.generate_learning_samples(all_samples, all_labels,
#                                                                                             num_limits_each_label=number_each_class)
#     logging.info('train samples are ready.')
#     statusbar.showMessage("train samples are ready. ")
#
#     logging.info('training SVM...')
#     statusbar.showMessage("Start training SVM...")
#     model.train(samples_train, labels_train)
#     # model.train(all_samples, all_labels)
#     logging.info('training is done')
#     statusbar.showMessage("Train - Done!")
#     logging.info(model.model)
#
#
# def ml_evaluate(model, statusbar, rate_le, confusion_matrix_pte):
#     statusbar.showMessage("Start evaluating SVM ...")
#     if not os.path.exists(workspace.ml_evaluate_samples_path):
#         os.makedirs(workspace.ml_evaluate_samples_path)
#
#     if len(os.listdir(workspace.ml_evaluate_samples_path)) == 0:
#         logging.info("The evaluate dir is empty. Please add some evaluate files. And then try again. ")
#     else:
#         learningImagesFeaturesExtractor = LearningImagesFeaturesExtractor()
#         logging.info('Start generating evaluate samples...')
#         evaluate_samples, evaluate_labels = learningImagesFeaturesExtractor.load_all_learning_source(
#             workspace.ml_evaluate_samples_path)
#         logging.info('evaluate samples are ready.')
#
#         reco, confusion = model.evaluate_model(evaluate_samples, evaluate_labels)
#
#         logging.info('Recognition Rate: %.2f %%' % (reco * 100))
#
#         logging.info('confusion matrix:')
#         logging.info(confusion)
#
#         rate_le.setText(' %.2f %%' % (reco * 100))
#         confusion_matrix_pte.clear()
#         confusion_matrix_pte.insertPlainText('Confusion Matrix : \n' + str(confusion))
#
#         statusbar.showMessage("Evaluate - Done!")
#
#
# def ml_evaluate_save_result(rate_le, confusion_matrix_pte):
#     evaluate_result_str = "Recognition Rate: " + rate_le.text() + '\n\n' + confusion_matrix_pte.toPlainText()
#     fn = QFileDialog.getSaveFileName(None, "Save Evaluate Result", workspace.ml_evaluate_samples_path)
#     if fn[0] is not None:
#         with open(fn[0], 'w') as save_file:
#             save_file.write(evaluate_result_str)
#
#
# def ml_predict(model, statusbar):
#     fn = QFileDialog.getOpenFileName(None, "Choose a feature file to predict", workspace.predict_features_path)
#
#     if fn[0] is not None:
#         statusbar.showMessage("Start predicting...")
#         dirpath, filename = os.path.split(fn[0])
#         result = model.predict_image(dirpath, filename)
#
#         if not os.path.exists(workspace.ml_predict_results_path):
#             os.makedirs(workspace.ml_predict_results_path)
#
#         result_dir_name = filename.split('.')[0]
#
#         result_dir = os.path.join(workspace.ml_predict_results_path, result_dir_name)
#
#         if not os.path.exists(result_dir):
#             os.makedirs(result_dir)
#
#         result.save_to_file(result_dir, filename)
#         result.generate_class_image(result_dir, filename, workspace.predict_image_path)
#         result.save_all_single_class_images(result_dir)
#         result.save_all_class_image(result_dir)
#
#         statusbar.showMessage("Predict-Done!")
#
#
# # analyse results
# def generate_class_image():
#     fn = QFileDialog.getOpenFileName()
#     dirpath, filename = os.path.split(fn[0])
#
#     image_result = ImageResult()
#     image_result.generate_class_image(dirpath, filename)
#
#
# if __name__ == '__main__':
#     """The main entrance"""
#     logging.info("--------------Job's starting! {} --------------".format(current_time()))
#     # check_env()
#     # check_root()
#     # extract_all_features()
#     # generate_learning_table()
#     # machinelearning()
#     logging.info("--------------Job's done! {} --------------".format(current_time()))
