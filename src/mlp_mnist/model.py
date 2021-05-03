from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, GaussianNoise, Activation, Reshape
from keras import regularizers

def create_model(layers, dropout_rate = 0, batch_norm = True, gaussian_noise_stddev = 0, activation ='relu', l2_reg = 0.01):
    model = Sequential()
    # flat input layer
    model.add(Reshape(target_shape=(784,), input_shape=(28,28,1)))
    if gaussian_noise_stddev:
            model.add(GaussianNoise(gaussian_noise_stddev))
    # stack dense layers
    for layer in layers:
        # layers
        model.add(Dense(layer ,kernel_regularizer=regularizers.l2(l2_reg)))
        # optional features
        if gaussian_noise_stddev:
            model.add(GaussianNoise(gaussian_noise_stddev))
        # activation
        model.add(Activation(activation))
        if batch_norm:
            model.add(BatchNormalization())
        if dropout_rate:
            # decrease drop out with layer progression
            model.add(Dropout(dropout_rate))
    # add classification layer
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model




