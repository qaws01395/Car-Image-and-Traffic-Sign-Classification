import numpy as np
import pandas as pd
import cv2

# read bounding box file
bb_path = "filtered_cars_train_annos.csv"
pre_train_anno = pd.read_csv(bb_path, header = None)
pre_train_car = np.asarray(pre_train_anno.iloc[:, :-2])
train_label = np.asarray(pre_train_anno.iloc[:, -2])
img_list = np.asarray(pre_train_anno.iloc[:, -1])

print(img_list)
# print(len(img_list))

num = 0
train_car = []

while (num < len(img_list)):
#while (num < 100):
    image_name = img_list[num]
#     print(image_name)
    img = cv2.imread('cars_train/'+image_name, cv2.IMREAD_COLOR)

    y1 = pre_train_car[num][1]
    y2 = pre_train_car[num][3]
    x1 = pre_train_car[num][0]
    x2 = pre_train_car[num][2]

#     print(y1)
#     print(y2)
#     print(x1)
#     print(x2)
    train_car.append(img[y1:y2, x1:x2])
    # cv2.imshow('imageTest',img[y1:y2, x1:x2])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # save imgs with only the bounding box covers
    cv2.imwrite('bounding_cars_train/'+image_name, img[y1:y2, x1:x2])
    # resize imgs and save them
    resized_image = cv2.resize(img[y1:y2, x1:x2], (256, 128))
    cv2.imwrite('resize_cars_train/'+image_name, resized_image)

    num = num + 1

for i in range(len(train_car)):
    print(train_car[i].shape)
