from keras.datasets import mnist
import keras

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # flatten images
    x_train = x_train.reshape(-1, 28*28).astype('float32')
    x_test = x_test.reshape(-1, 28*28).astype('float32')

    # normalize color values to 0 - 1
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)