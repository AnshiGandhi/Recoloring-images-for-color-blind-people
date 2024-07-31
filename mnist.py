import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

model = tf.keras.models.Sequential([
    # relu function to work with non-linear data
    # 3*3 matrix
    # feature extraction
    tf.keras.layers.Conv2D(28, (3,3), activation='relu', input_shape=(28, 28, 1)), 
    tf.keras.layers.MaxPooling2D(2,2),
    # overcome overfitting
    # dropping a few neurons
    tf.keras.layers.Dropout(0.2), 
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
# training for 10 epochs
model.fit(x=x_train,y=y_train, epochs=10)

# testing our accuracy
model.evaluate(x_test, y_test, verbose=1)

# saving the model
model.save("mnist.h5")

# we get a training accuracy of about 99% and test set accuracy of 98%