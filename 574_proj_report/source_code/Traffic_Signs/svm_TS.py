# Support Vector Machine (SVM)
import numpy as np
import os
import pandas as pd
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt

def load_data(data_dir):
    # """Loads a data set and returns two lists:
    #
    # images: a list of Numpy arrays, each representing an image.
    # labels: a list of numbers that represent the images labels.
    # """

    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            img = skimage.data.imread(f)
            # resize to same size, if it was cut by bounding box, no need to resize
            img = skimage.transform.resize(img, (32,32))
            img = np.reshape(img,(3072))
            # print(img.shape)
            images.append(img)
            labels.append(int(d))
    return images, labels


# Load training and testing datasets.
train_data_dir = os.path.join("","datasets/BelgiumTS/Training")
test_data_dir = os.path.join("","datasets/BelgiumTS/Testing")

# train_data_dir = "bb_Training"
# test_data_dir = "bb_Test"

X_train, Y_train = load_data(train_data_dir)
X_test, Y_test = load_data(test_data_dir)

train_images = X_train
train_labels = Y_train
test_images = X_test
test_labels = Y_test

# print(X_train)
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
print(X_train.shape)
# print(Y_train)
print(Y_train.shape)


# Importing the dataset
# dataset = pd.read_csv('filtered_cars_train_annos.csv')
# train_label = np.asarray(dataset.iloc[:, -2])
# img_list = np.asarray(dataset.iloc[:, -1])
#
# num = 0
# train_car = []
# while (num < len(img_list)):
#     image_name = img_list[num]
# #     print(image_name)
#     img = cv2.imread('resize_cars_train/'+image_name, cv2.IMREAD_GRAYSCALE)
#     img = np.reshape(img,(32768))
#     train_car.append(img)
#     num += 1
#
#
# x_train = np.array(train_car)
# print(x_train.shape)
# y_train = train_label

# classID_left = np.genfromtxt('classid_left.txt', delimiter=',')
# classID_left = classID_left.astype('int')

# dict = {value: key for (key, value) in enumerate(classID_left)}
# converted_y_train = []
# for cl in y_train:
#     converted_y_train.append(dict.get(cl))
#
# converted_y_train = np.array(converted_y_train)


# X = dataset.iloc[:, [2, 3]].values
# y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
# from sklearn.cross_validation import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)

# Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(x_train)
# # X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV
#
# clf = SVC() # here you can choose different classifier in sklearn
#
# param_grid = {
#     'C': [ 0, 1, 10, 100, 1000],
#     'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
#     'gamma': [0.001, 0.01, 0.1, 1.0, 0.005, 0.0005, 'auto'],
#     'degree': [2, 3, 4, 5, 6],
#     'coef0': [0.0, 0.5, 1.0, 5.0, 0.01, 0.001]
# }
#
# # for grid search, please check the online document for details
# grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, verbose=1, )
#
# grid_search.fit(X_train, Y_train)
#
# # print grid_search.best_params_
#
#
# # up to now, get the best parameters
# best_gamma = grid_search.best_params_['gamma']
# best_C = grid_search.best_params_['C']
# best_kernel = grid_search.best_params_['kernel']
# best_degree = grid_search.best_params_['degree']
# best_coef0 = grid_search.best_params_['coef0']
#
#
# # re build the model using best parameters
# clf = SVC(C = best_C, kernel = best_kernel, gamma = best_gamma, degree = best_degree, coef0 = best_coef0)
#
# print(clf.fit(X_train, Y_train))
# print(clf.score(X_test,Y_test))


# 88.5%
classifier = SVC(kernel = 'linear', random_state = 0)
print(classifier.fit(X_train, Y_train))
print(classifier.score(X_test,Y_test))

# 58.7%
# classifier = SVC(kernel = 'rbf', random_state = 0)
# print(classifier.fit(X_train, Y_train))
# print(classifier.score(X_test,Y_test))
# 65.07%
# classifier = SVC(kernel = 'rbf', random_state = 0, gamma=0.0005)
# print(classifier.fit(X_train, Y_train))
# print(classifier.score(X_test,Y_test))
# 38.57%
# classifier = SVC(kernel = 'rbf', random_state = 0, gamma=0.00005)
# print(classifier.fit(X_train, Y_train))
# print(classifier.score(X_test,Y_test))
# 80.95%
# classifier = SVC(kernel = 'rbf', random_state = 0, gamma=0.005)
# print(classifier.fit(X_train, Y_train))
# print(classifier.score(X_test,Y_test))
# 20.67%
# classifier = SVC(kernel = 'rbf', random_state = 0, gamma=0.05)
# print(classifier.fit(X_train, Y_train))
# print(classifier.score(X_test,Y_test))
# 34.08%
# classifier = SVC(kernel = 'poly', random_state = 0, degree = 2)
# print(classifier.fit(X_train, Y_train))
# print(classifier.score(X_test,Y_test))
# 33.33%
# classifier = SVC(kernel = 'poly', random_state = 0, degree = 3, coef0=0.0, gamma=0.0005)
# print(classifier.fit(X_train, Y_train))
# print(classifier.score(X_test,Y_test))
# 72.89%
# classifier = SVC(kernel = 'poly', random_state = 0, degree = 3, coef0=0.0, gamma=0.005)
# print(classifier.fit(X_train, Y_train))
# print(classifier.score(X_test,Y_test))
# %
# classifier = SVC(kernel = 'poly', random_state = 0, degree = 3, coef0=0.0, gamma=0.005)
# print(classifier.fit(X_train, Y_train))
# print(classifier.score(X_test,Y_test))
# %
# classifier = SVC(kernel = 'poly', random_state = 0, degree = 3, coef0=0.0, gamma=0.005)
# print(classifier.fit(X_train, Y_train))
# print(classifier.score(X_test,Y_test))
# 50.9%
# classifier = SVC(kernel = 'sigmoid', random_state = 0, gamma='auto', coef0=0.0)
# print(classifier.fit(X_train, Y_train))
# print(classifier.score(X_test,Y_test))
# 2.42%
# classifier = SVC(kernel = 'sigmoid', random_state = 0, gamma='auto', coef0=10)
# print(classifier.fit(X_train, Y_train))
# print(classifier.score(X_test,Y_test))
# %
# classifier = SVC(kernel = 'sigmoid', random_state = 0, gamma='auto', coef0=2)
# print(classifier.fit(X_train, Y_train))
# print(classifier.score(X_test,Y_test))


