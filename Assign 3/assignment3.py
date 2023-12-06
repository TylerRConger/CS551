# https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
# https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/

from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
import numpy as np
from sklearn.metrics import accuracy_score

# Loading the dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Building the input vector from the 28x28 pixels
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalizing the data to help with the training
X_train /= 255
X_test /= 255

# One-hot encoding using keras numpy utilities
n_classes = 10
print("Shape before one-hot encoding: ", Y_train.shape)
Y_train = to_categorical(Y_train, n_classes)
Y_test = to_categorical(Y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

# Use the sequential model
model = Sequential()
# Creation of the convolutional layer
model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(1,1)))
model.add(Flatten())
# Hidden layer
model.add(Dense(100, activation='relu'))
# End Layer
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Training the model for 10 epochs
model.fit(X_train, Y_train, batch_size=128, epochs=1, validation_data=(X_test, Y_test))

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, Y_test)

# Print out loss and accuracy stats
print(test_loss)
print(test_accuracy)

# Visualize Filters/Kernels
def visualize_filters(layer_index, num_filters=25):

    # Get layers and weights
    layer = model.layers[layer_index]
    filters = layer.get_weights()[0]

    # Plot 25 filters
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    
    # Add each item to the 25 filter plot
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
    rows = (num_filters - 1) // cols + 1

    fig, axs = plt.subplots(rows, cols, figsize=(15, 15))

    for i in range(num_filters):
        ax = axs[i // cols, i % cols]
        ax.matshow(activations[0, :, :, i], cmap='viridis')
        ax.axis('off')

    # Hide any remaining empty subplots
    for i in range(num_filters, rows * cols):
        axs.flatten()[i].axis('off')

    plt.show()

# Choose an image from the test set for feature map visualization
image_to_visualize = X_test[1]

# Visualize feature maps for the first convolutional layer
visualize_feature_maps(layer_index=0, input_image=image_to_visualize)