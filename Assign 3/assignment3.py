import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils
# keras imports for the dataset and building our neural network
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils

import numpy as np

# to calculate accuracy
from sklearn.metrics import accuracy_score

# loading the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# building the input vector from the 28x28 pixels
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data to help with the training
X_train /= 255
X_test /= 255

# one-hot encoding using keras' numpy-related utilities
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

# building a linear stack of layers with the sequential model
model = Sequential()
# convolutional layer
model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(1,1)))
# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(100, activation='relu'))
# output layer
model.add(Dense(10, activation='softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# training the model for 10 epochs
model.fit(X_train, Y_train, batch_size=128, epochs=1, validation_data=(X_test, Y_test))


# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_train, Y_train)

print(test_loss)
print(test_accuracy)

"""
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Design and implement the CNN model with one convolutional layer
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
print(f'Test Loss: {test_loss:.4f}')
# Get the weights (filters) of the first convolutional layer
first_conv_layer = model.layers[0]
filters = first_conv_layer.get_weights()[0]


# Plot the filters
fig, axs = plt.subplots(4, 8, figsize=(12, 6))
for i in range(4):
    for j in range(8):
        axs[i, j].imshow(filters[:, :, 0, i * 8 + j], cmap='viridis')
        axs[i, j].axis('off')

plt.show()

from tensorflow.keras import models

# Choose an image from the test set
image_to_visualize = test_images[0]

# Create a model that outputs the activations of all layers
layer_outputs = [layer.output for layer in model.layers]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# Get the activations for the chosen image
activations = activation_model.predict(image_to_visualize.reshape(1, 28, 28, 1))

# Visualize the activations for the first convolutional layer
first_layer_activation = activations[0]
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
plt.show()
"""

# 2. Visualization

# Visualize Filters/Kernels
def visualize_filters(layer_index, num_filters=25):
    layer = model.layers[layer_index]
    filters = layer.get_weights()[0]
    
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    
    for i in range(num_filters):
        ax = axs[i // 5, i % 5]
        ax.imshow(filters[:, :, 0, i], cmap='viridis')
        ax.axis('off')
    
    plt.show()

# Visualize the filters of the initial convolutional layer
visualize_filters(layer_index=0)

# Use Feature Map Visualization
def visualize_feature_maps(layer_index, input_image):
    activation_model = models.Model(inputs=model.input, outputs=model.layers[layer_index].output)
    activations = activation_model.predict(np.expand_dims(input_image, axis=0))

    num_filters = activations.shape[3]
    cols = 8
    rows = num_filters // cols + 1

    fig, axs = plt.subplots(rows, cols, figsize=(15, 15))

    for i in range(num_filters):
        ax = axs[i // cols, i % cols]
        ax.matshow(activations[0, :, :, i], cmap='viridis')
        ax.axis('off')

    plt.show()

# Choose an image from the test set for feature map visualization
image_to_visualize = X_test[1]

# Visualize feature maps for the first convolutional layer
visualize_feature_maps(layer_index=0, input_image=image_to_visualize)