import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV

dataset = pd.read_csv('CarCls.csv')
train_label = np.asarray(dataset.iloc[:, -2])
img_list = np.asarray(dataset.iloc[:, -1])

num = 0
train_car = []
while (num < len(img_list)):
    image_name = img_list[num]
    img = cv2.imread('mini_bounding_cars_trainv2/'+image_name, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img,None)
    for i in range(len(kp)):
        train_car.append(des[i])
    num += 1
train_car = np.matrix(train_car)

kmeans = KMeans(n_clusters=15, random_state=0).fit(train_car)
num = 0
img_clusters_histogram =[]
clusters = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
while (num < len(img_list)):
    image_name = img_list[num]
    print(image_name)
    img = cv2.imread('mini_bounding_cars_trainv2/'+image_name, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img,None)
    for i in range(len(kp)):
        clusters[kmeans.predict(des[i].reshape(1, -1))[0]] += 1
    print("image",num,": ",clusters)
    img_clusters_histogram.append(clusters)
    clusters = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    num += 1

img_clusters_histogram = np.array(img_clusters_histogram)
print("img_clusters_histogram shape:",img_clusters_histogram.shape)

clf = SVC() # here you can choose different classifier in sklearn

param_grid = {
    'C': [ 1, 10, 100, 1000],
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'gamma': [0.001, 0.01, 0.1, 1.0],
}
grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, verbose=1, )

grid_search.fit(img_clusters_histogram, train_label)
best_gamma = grid_search.best_params_['gamma']
best_C = grid_search.best_params_['C']
best_kernel = grid_search.best_params_['kernel']
clf = SVC(C = best_C, kernel = best_kernel, gamma = best_gamma)
print(clf.fit(img_clusters_histogram, train_label))
