import numpy as np
import pandas as pd
import cv2

# read bounding box file
bb_path = "CarCls.csv"
pre_train_anno = pd.read_csv(bb_path, header = None)
pre_train_car = np.asarray(pre_train_anno.iloc[:, :-2])
train_label = np.asarray(pre_train_anno.iloc[:, -2])
img_list = np.asarray(pre_train_anno.iloc[:, -1])

print(img_list)
# print(len(img_list))

train_car = []

arr = [1, 47, 83];



for i in range(len(arr)):
    num = 0
    while (num < len(img_list)):
    #while (num < 100):
        if (train_label[num] == arr[i]):
            image_name = img_list[num]
            print(image_name)
            img = cv2.imread('resize_cars_train/'+image_name, cv2.IMREAD_COLOR)

#            y1 = pre_train_car[num][1]
#            y2 = pre_train_car[num][3]
#            x1 = pre_train_car[num][0]
#            x2 = pre_train_car[num][2]
#
#
            train_car.append(img)

            
            # cv2.imshow('imageTest',img[y1:y2, x1:x2])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # save imgs with only the bounding box covers
            cv2.imwrite('mini_bounding_cars_trainv2/'+image_name, img)
            
            # resize imgs and save them
#            resized_image = cv2.resize(img[y1:y2, x1:x2], (256, 128))
#            cv2.imwrite('mini_resize_cars_trainv2/'+image_name, resized_image)
        num = num + 1

for i in range(len(train_car)):
    print(train_car[i].shape)
