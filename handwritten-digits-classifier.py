#Import Tensorflow and NumPy
import tensorflow as tf
import numpy as np

#Callback to cancel training after the model reaches 99.99% accuracy
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

#Load the MNIST dataset
mnist = tf.keras.datasets.mnist

#Identify the training and test data in the dataset
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

#Instantiate the callback
callbacks = myCallback()

#Define the model (takes images that are 28x28)
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

#Specifies how to optimize the model and calculate loss/accuracy.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Runs the model 10 times
model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])

#Defines the possible classifications for the test images
classifications = model.predict(x_test)

#Prints the first test image's label
print("Number in the image tested:")
print(y_test[0])

#Prints what number the model thought the first test image was
print("Number guessed by the model:")
np.argmax(classifications[0])
