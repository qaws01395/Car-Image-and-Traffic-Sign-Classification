import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.utils import np_utils


# Allow image embeding in notebook
# %matplotlib inline

seed = 5
np.random.seed(seed)

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
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels


# Load training and testing datasets.
# ROOT_PATH = "/traffic"
train_data_dir = os.path.join("","datasets/BelgiumTS/Training")
test_data_dir = os.path.join("","datasets/BelgiumTS/Testing")
# train_data_dir = "Training"
# test_data_dir = "Testing"

# train_data_dir = os.path.join("","datasets/BelgiumTS/bb_Training")
# test_data_dir = os.path.join("","datasets/BelgiumTS/bb_Testing")

images, labels = load_data(train_data_dir)
test_images, test_labels = load_data(test_data_dir)

print("Unique Labels: {0}\nTotal Images: {1}".format(len(set(labels)), len(images)))




# Resize images
images32 = [skimage.transform.resize(image, (32, 32))
                for image in images]
# display_images_and_labels(images32, labels)

labels_a = np.array(labels)
Y_train = np_utils.to_categorical(labels_a, 62)




images_a = np.array(images32)



from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten

# declare a Sequential model
model = Sequential()

# Add the first Convolutional layer
from keras.layers import Conv2D, MaxPooling2D
model.add(Conv2D(
    input_shape=(32,32,3),
    filters=64, # number of filters
    kernel_size=3, # 2x2 filter
    strides=1,
    padding='same', # the size of image remains the same
))
model.add(Activation('relu'))

# Add the first max pooling
# info losses occur during convolution, pooling solves this problen and picks up useful info for the next layer.
# reduce amount of parameters => reduce rates of overfitting
model.add(MaxPooling2D(
    pool_size=2, # 2x2
    strides=2,
    padding='same'
    # data_format='channels_first'
))
# (32, 4, 4)

# Add the second Convolutional layer
model.add(Conv2D( 64, 5, strides=1, padding='same'))
model.add(Activation('relu'))

# Add the second max pooling
model.add(MaxPooling2D( 2, 2, padding='same'))
# (64, 7, 7)

# fully connected network
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))

# output: 49 neurons
model.add(Dense(62))
model.add(Activation('softmax'))


# --- Choose optimizer ---
from keras.optimizers import SGD, Adam, RMSprop, Adagrad

# update = -lr*gradient + m*last_update
# 1/t decay -> lr = lr / (1+ decay*t), t: number of done updates

# sgd = SGD(lr=0.01, momentum=0.2, decay=1e-6, nesterov=False) # epochs=15 69.68% 70.91%
# sgd = SGD(lr=0.01, decay=1e-2, momentum=0.9, nesterov=True) # 74.64% 75.55%
rmsprop = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0) # 6.90% 16.74%
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # training 75.08% test 82.38%
# adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # 72.45% 73.88%
# adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-09, decay=0.0) # 75.03% 81.46%

# compile the model
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# specify early stopping
from keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='acc', min_delta=0.003, patience=2, verbose=0, mode='max')
early_stopping = EarlyStopping(monitor='loss', min_delta=0.003, patience=3)

# train the model
history = model.fit(images_a, Y_train, epochs=15, batch_size=32, validation_split=0.25, callbacks=[early_stopping])

#
#
# Result
loss, accuracy = model.evaluate(images_a, Y_train)
print("\ntrain loss: {}".format(loss))
print("\ntrain accuracy: {}".format(accuracy))




from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


print((len(test_images)))

test_images32 = [skimage.transform.resize(image, (32, 32))
                 for image in test_images]

images_b = np.array(test_images32)

# print(images32)
# print(images_b)
test_predictions = model.predict(images_b)

# print(test_predictions)


# match_count = sum([int(y == y_) for y, y_ in zip(test_labels, test_predictions)])
# accuracy = match_count / len(test_labels)
# print(len(test_labels))
# print("Accuracy: {:.3f}".format(accuracy))

test_labels = np.array(test_labels)

test_pred = np.argmax(test_predictions, axis=1)
# test_label = np.argmax(test_labels, axis=1)
print("*** Results from Testing Sample ***")
print("overall testing data accuracy is %f " %accuracy_score(test_labels, test_pred))
print(classification_report(test_labels, test_pred))
print(confusion_matrix(test_labels, test_pred))



def display_images_and_labels(images, labels):
    """Display the first image of each label."""
#     unique_labels = set(labels)
    unique_labels = labels
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        # Pick the first image for each label.
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label {0}".format(label))
        i += 1
        _ = plt.imshow(image)
    plt.show()

# display_images_and_labels(images32, labels)



# Pick 10 random images
sample_indexes = random.sample(range(len(test_images32)), 10)
sample_images = [test_images32[i] for i in sample_indexes]
sample_labels = [test_labels[i] for i in sample_indexes]



display_images_and_labels(sample_images, sample_labels)



# Run the "predicted_labels" op.


sample_images = np.array(sample_images)
sample_labels = np.array(sample_labels)

sample_predictions = model.predict(sample_images)


sample_pred = np.argmax(sample_predictions, axis=1)
print(sample_labels)
print(sample_pred)


# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = sample_pred[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
             fontsize=12, color=color)
    plt.imshow(sample_images[i])