from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, \
    Flatten, BatchNormalization, GaussianNoise, Dropout, Activation
from keras import regularizers


def create_model(dropout_rate=0.5, gn_stddev=0.3, l2_reg = 0.0):
    model = Sequential()

    # add convolution layers
    model = add_conv_layer(model, 64, (3,3), dropout_rate, gn_stddev, l2_reg, input_shape=(32,32,3))
    model = add_pool_layer(model, (2,2))

    model = add_conv_layer(model, 128, (3, 3), dropout_rate, gn_stddev, l2_reg)
    model = add_pool_layer(model, (2, 2))

    model = add_conv_layer(model, 256, (3, 3), dropout_rate, gn_stddev, l2_reg)
    model = add_conv_layer(model, 256, (3, 3), dropout_rate, gn_stddev, l2_reg)
    model = add_pool_layer(model, (2, 2))

    model = add_conv_layer(model, 512, (3, 3), dropout_rate, gn_stddev, l2_reg)
    model = add_conv_layer(model, 512, (3, 3), dropout_rate, gn_stddev, l2_reg)
    model = add_pool_layer(model, (2, 2))

    model = add_flatten_layer(model)

    # add dense layers
    #model = add_dense_layer(model, 1024, dropout_rate, gn_stddev, l2_reg)
    model = add_dense_layer(model, 256, dropout_rate, gn_stddev, l2_reg)
    model = add_dense_layer(model, 256, dropout_rate, gn_stddev, l2_reg)
    model = add_dense_layer(model, 128, dropout_rate, gn_stddev, l2_reg)

    # add final classification layer
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model



def add_conv_layer(model, filters, kernel=(3,3), dropout_rate=0.0, gn_stddev=0.3, l2_reg=0.0, activation='relu', input_shape=None):
    if input_shape is not None:
        model.add(Conv2D(filters, kernel, padding='same', kernel_regularizer=regularizers.l2(l2_reg), input_shape=input_shape))
    else:
        model.add(Conv2D(filters, kernel, padding='same', kernel_regularizer=regularizers.l2(l2_reg)))
    if gn_stddev:
        model.add(GaussianNoise(gn_stddev))
    model.add(Activation(activation))
    model.add(BatchNormalization()) # BN after activation!
    if dropout_rate:
        model.add(Dropout(dropout_rate))
    return model



def add_flatten_layer(model):
    model.add(Flatten())
    return model



def add_pool_layer(model, pool_size):
    model.add(MaxPool2D(pool_size=pool_size))
    return model



def add_dense_layer(model, size, dropout_rate=0.0, gn_stddev=0.3, l2_reg=0.0, activation='relu', input_shape=None):
    if input_shape is not None:
        model.add(Dense(units=size,activation=activation,kernel_regularizer=regularizers.l2(l2_reg),input_shape=input_shape))
    else:
        model.add(Dense(units=size, activation=activation,kernel_regularizer=regularizers.l2(l2_reg)))
    if gn_stddev:
        model.add(GaussianNoise(gn_stddev))
    model.add(Activation(activation))
    model.add(BatchNormalization()) # BN after activation!
    if dropout_rate:
        model.add(Dropout(dropout_rate))
    return model

