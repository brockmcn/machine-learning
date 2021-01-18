#    name: pa2Template.py
# purpose: template for building a Keras model
#          for hand written number classification
#    NOTE: Submit a different python file for each model
# -------------------------------------------------

import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical

import tensorflow as tf
import datetime

import argparse

from m1_cnn_pre import processTestData


def parseArguments():
    parser = argparse.ArgumentParser(
        description='Build a Keras model for Image classification')

    parser.add_argument('--training_x', action='store',
                        dest='XFile', default="", required=True,
                        help='matrix of training images in npy')
    parser.add_argument('--training_y', action='store',
                        dest='yFile', default="", required=True,
                        help='labels for training set')

    parser.add_argument('--outModelFile', action='store',
                        dest='outModelFile', default="", required=True,
                        help='model name for your Keras model')
    return parser.parse_args()

def main():
    np.random.seed(1671)

    # Parse arguments
    #parms = parseArguments()

    # Load data
    X_train = np.load('MNIST_X_train.npy')
    y_train = np.load('MNIST_y_train.npy')
    X_test = np.load('MNIST_X_test.npy')
    y_test = np.load('MNIST_y_test.npy')

    # Pre-process data
    X_train, y_train = processTestData(X_train, y_train)
    X_test, y_test = processTestData(X_test, y_test)
    X_train_shape = X_train.shape[1:]
    print("X_train ndims: {}".format(X_train.ndim))
    print(X_train.shape)
    print("y_train ndims: {}".format(y_train.ndim))
    print(y_train.shape)

    # Build model
    print('KERA modeling build starting...')
    model = Sequential()

    # Layers 1
    model.add(Conv2D(32, kernel_size=5, padding='same', activation='relu', input_shape=X_train_shape))
    model.add(MaxPooling2D(pool_size=3))

    # Layers 2
    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu', input_shape=X_train_shape))
    model.add(MaxPooling2D(pool_size=2))

    # Finish
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    # Summary
    model.summary()

    # Tensorboard
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_c = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Fit
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=8, callbacks=[tb_c])
    print('KERA model building finished')

    # Save Model
    model.save('m2_cnn.h5')

if __name__ == '__main__':
    main()