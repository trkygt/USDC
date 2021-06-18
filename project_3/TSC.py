import pickle
import numpy as np
import matplotlib.pyplot as plt


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

# import pandas as pd
# sign_names = pd.read_csv("signnames.csv")
# signs = sign_names.SignName.values.tolist()
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
training_data = np.zeros_like(X_train)
i = 0
for x in X_train:
    x = x - np.mean(x)
    training_data[i] = x
    i = i + 1

testing_data = np.zeros_like(X_test)
i = 0
for x in X_test:
    x = x - np.mean(x)
    testing_data[i] = x
    i = i + 1



