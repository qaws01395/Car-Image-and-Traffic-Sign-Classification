#!/usr/bin/python

import csv
# with open('cars_meta.csv','r') as File:
#     classCar = csv.reader(File, delimiter = ',')
#     unique_brand = []
#     count = 1
#     last_brand = ''
#     for row in classCar:
#         for brand in row:
#             car_name = brand.split(' ')
#             cmpy_name = car_name[0]
#             if cmpy_name != last_brand:
#                 unique_brand.append(count)
#                 last_brand = cmpy_name
#             count += 1
#     print(unique_brand)
#     print(len(unique_brand))

# print out what kind of cars left from 196 classes
# with open('cars_meta.csv','r') as File:
#     classCar = csv.reader(File, delimiter = ',')
#     for row in classCar:
#         for i in unique_brand:
#             print(row[i-1])


unique_brand = [1,47,106,126,150]


#delete cars image we dont want
import os, sys
with open('cars_train_annos.csv','r') as File:
    trainFile = csv.reader(File, delimiter = ',')
    for row in trainFile:
        if int(row[4]) not in unique_brand:
            os.remove('cars_train/'+row[5])

#get corresponding label
train_label = open('cars_train_annos.csv', 'r')
filtered_train_label = open('mini_filtered_cars_train_annos.csv', 'w')
fwriter = csv.writer(filtered_train_label)
for row in csv.reader(train_label):
    if int(row[4]) in unique_brand:
        fwriter.writerow(row)
train_label.close()
filtered_train_label.close()
