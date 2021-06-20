import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2
import pandas as pd
import tensorflow as tf
# from tensorflow_core.contrib.layers import flatten

print(tf.__version__)


# TODO: Fill this in based on where you saved the training and testing data
training_file = "train.p"
validation_file = "valid.p"
testing_file = "test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(y_train)

# TODO: Number of validation examples
n_validation = len(y_valid)

# TODO: Number of testing examples.
n_test = len(y_test)

# TODO: What's the shape of an traffic sign image?
image_shape = np.shape(X_train[0])

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

# print("Number of training examples =", n_train)
# print("Number of testing examples =", n_test)
# print("Image data shape =", image_shape)
# print("Number of classes =", n_classes)


### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.


sign_names = pd.read_csv("signnames.csv")
signs = sign_names.SignName.values.tolist()
# print(signs[42])
# c = 0
# fig, axs = plt.subplots(4, 11, figsize=(24, 9))
# while c < 43:
#     i = c // 11
#     j = c % 11
#     ind = list(y_train).index(c)
#     axs[i, j].imshow(X_train[ind])
#     text = signs[c]
#     axs[i, j].set_title(text,fontsize=8)
#     c = c + 1
# axs[3, 10].hist(y_train)
# axs[3, 10].set_xlabel('Classes')
# axs[3, 10].set_ylabel('Occurrence of Classes')
#
# fig.tight_layout()
# plt.show()



# Data normalization
X_train, y_train = shuffle(X_train, y_train)


def preprocess_img(images):

    shape_n = image_shape[0:2] + (1,)
    new_images = np.empty(shape=(len(images),) + shape_n, dtype=int)

    for i in range(len(images)):
        normalized_img = cv2.normalize(images[i], np.zeros(image_shape[0:2]), 0, 255, cv2.NORM_MINMAX)
        grayscaled_img = cv2.cvtColor(normalized_img, cv2.COLOR_RGB2GRAY)

        new_images[i] = np.reshape(grayscaled_img, shape_n)

    return new_images


def resample_data(images, img_labels):
    num = max([len(np.where(images == c_id)[0]) for c_id in sign_names.keys()])

    resampled_images = np.empty(shape=(num * n_classes,) + images.shape[1:], dtype=int)
    resampled_labels = np.empty(shape=(num * n_classes,), dtype=int)
    j = 0

    for c_id in sign_names.keys():
        c_inds = np.where(y_train == c_id)[0]
        c_inds_len = len(c_inds)

        for i in range(0, num):
            resampled_images[j] = images[c_inds[i % c_inds_len]]
            resampled_labels[j] = img_labels[c_inds[i % c_inds_len]]
            j += 1

    # at this stage data is definitely not randomly shuffled, so shuffle it
    return shuffle(resampled_images, resampled_labels)


X_train_norm = preprocess_img(X_train)
X_test_norm = preprocess_img(X_test)
X_valid_norm = preprocess_img(X_valid)

X_train_norm, y_train_norm = resample_data(X_test_norm, y_train)
y_test_norm = y_test
y_valid_norm = y_valid

image_shape_norm = X_train_norm[0].shape








