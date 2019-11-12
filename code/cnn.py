from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
from utils import *
from confusion_matrix import plotConfusionMatrix, confusionMatrix, class_accuracy
import matplotlib.pyplot as plt


def prepareTrainingset():
    '''
    loads dataset and converts all the image array to grayscale images
    :param file:
    :return: array of images and array of classes
    '''
    images, cls = load_training_data()
    images = (images *2 / 255) -1
    return images, cls.reshape(-1, 1)

def prepareTestset():
    '''
    loads dataset and converts all the image array to grayscale images
    :param file:
    :return: array of images and array of classes
    '''
    images, cls = load_test_data()
    images = (images *2 / 255) -1
    return images, cls.reshape(-1, 1)


batch_size = 32
num_classes = 10
epochs = 2

# The data, split between train and test sets:
x_train, y_train = prepareTrainingset()
x_test, y_test = prepareTestset()


# Convert class vectors to binary class matrices.
y_train_one_hot = keras.utils.to_categorical(y_train, num_classes)
y_test_one_hot = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (5, 5), padding='same', strides=(1,1),
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))


model.add(Conv2D(32, (5, 5),padding='same', strides=(1,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))


model.add(Conv2D(64, (5, 5), padding='same', strides=(1,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))


model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))


model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

# Training the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


history = model.fit(x_train, y_train_one_hot,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test_one_hot),
                    shuffle=True)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Score of the trained model
scores = model.evaluate(x_test, y_test_one_hot, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# compute the confusion matrix
prediction_prob = model.predict(x_test)
predictions = np.argmax(prediction_prob, axis=1)
cm, classes = confusionMatrix(y_test, predictions)

# compute class accuracies
cls_accuracy, classes = class_accuracy(cm, classes)
cls_error_rate = 1-cls_accuracy
print("========================================")
print("Class labels: ", classes)
print("Class Accuracies: ", np.round(cls_accuracy, 3))
print("Class error rates: ", np.round(cls_error_rate, 3))

# finally plot confusion matrix
plotConfusionMatrix(cm, classes)