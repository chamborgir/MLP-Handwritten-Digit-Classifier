# import libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


#NOTE: load and preprocess dataset (MNIST dataset)
#load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# print(x_train.shape)  # (60000, 28, 28) → 60K images, each 28x28 pixels
# print(y_train.shape)  # (60000,) → 60K labels, one for each image
# print(x_test.shape)   # (10000, 28, 28) → 10K test images
# print(y_test.shape)   # (10000,) → 10K test labels

# normalize pixel values to range [0,1] (important for neural network)
x_train, x_test = x_train/255.0, x_test/255.0

# flatter the 28x28 images into 784-dimensional vectors
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
# (-1, 784) is equivalent to x_train.reshape(60000, 784) but we let Python handle it dynamically.



#define the MLP model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)), #hidden layer 1
    keras.layers.Dense(64, activation='relu'),  # layer 2
    keras.layers.Dense(10, activation='softmax')  # output (10 classes)
])

#compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


#train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test,y_test))


#evaluate model performance
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

#save model
model.save("mlp_mnist.h5")


#NOTE: show visually
#get predictions
predictions = model.predict(x_test)

#function to plot an image and prediction
def plot_prediction(index):
    plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {np.argmax(predictions[index])}, Actual: {y_test[index]}")
    plt.axis('off')
    plt.show()

#test images
for i in range(5):
    plot_prediction(i)