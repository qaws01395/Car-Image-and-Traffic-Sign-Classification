# Support Vector Machine (SVM)

# Importing the libraries
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import cv2

# Importing the dataset
dataset = pd.read_csv('CarCls.csv')
train_label = np.asarray(dataset.iloc[:, -2])
img_list = np.asarray(dataset.iloc[:, -1])

num = 0
train_car = []
while (num < len(img_list)):
    image_name = img_list[num]
#     print(image_name)
    img = cv2.imread('mini_bounding_cars_trainv2/'+image_name, cv2.IMREAD_GRAYSCALE)
    img = np.reshape(img,(32768))
    train_car.append(img)
    num += 1


x_train = np.array(train_car)
print(x_train.shape)
y_train = train_label

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
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV



X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)

# Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(x_train)
# # X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
#classifier = SVC(kernel = 'linear', random_state = 0,degree = 1)

# here I just give an example to these parameters. It has infinite combinations. You can list more.


clf = SVC() # here you can choose different classifier in sklearn

param_grid = {
    'C': [ 1, 10, 100, 1000],
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'gamma': [0.001, 0.01, 0.1, 1.0],
}

# for grid search, please check the online document for details
grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, verbose=1, )

grid_search.fit(X_train, Y_train)

# print grid_search.best_params_


# up to now, get the best parameters
best_gamma = grid_search.best_params_['gamma']
best_C = grid_search.best_params_['C']
best_kernel = grid_search.best_params_['kernel']


# re build the model using best parameters
clf = SVC(C = best_C, kernel = best_kernel, gamma = best_gamma)






print(clf.fit(X_train, Y_train))
print(clf.score(X_test,Y_test))

# Predicting the Test set results
# y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(x_train, y_train)

# # Visualising the Training set results
# from matplotlib.colors import ListedColormap
# X_set, y_set = X_train, y_train
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('SVM (Training set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()
#
# # Visualising the Test set results
# from matplotlib.colors import ListedColormap
# X_set, y_set = X_test, y_test
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('SVM (Test set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()
