#Import Tensorflow and NumPy
import tensorflow as tf
import numpy as np

#Callback to cancel training after the model reaches 99.99% accuracy
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc') >= 1.00):
      print("\nReached 100% accuracy. Cancelling training!")
      self.model.stop_training = True

#Load the MNIST dataset
mnist = tf.keras.datasets.mnist

#Identify the training and test data in the dataset
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#Normalize the data
x_train = x_train.reshape(60000, 28, 28, 1)
x_train = x_train / 255.0
x_test = x_test.reshape(10000, 28, 28, 1)
x_test = x_test / 255.0

#Instantiate the callback
callbacks = myCallback()

#Define the model (takes images that are 28x28)
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

#Specifies how to optimize the model and calculate loss/accuracy.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Runs the model 10 times
model.fit(x_train, y_train, epochs=35, callbacks=[callbacks])

#Defines the possible classifications for the test images
classifications = model.predict(x_test)

#Prints the first test image's label
print("Number in the image tested:")
print(y_test[0])

#Prints what number the model thought the first test image was
print("Number guessed by the model:")
np.argmax(classifications[0])
