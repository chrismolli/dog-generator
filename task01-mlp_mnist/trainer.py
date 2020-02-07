from model import create_model
from keras.datasets import mnist
import keras

# setup model
layers = [512 ,512,512,256,128,64,32]
model = create_model(layers, 0.1, True, 0.3, 'relu')
print(model.summary())

""" load mnist data """
from keras.datasets import mnist

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

""" setup optimizer """
from keras.optimizers import SGD

optimizer = SGD(lr=0.05, decay=1e-6, momentum=0.9)

""" training """
# Compile Model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

from keras.utils import plot_model
plot_model(model, to_file='model.png')

"""

# Training
history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=30,
                    verbose=1,
                    validation_data=(x_test, y_test))

# Evaluate over test
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
"""