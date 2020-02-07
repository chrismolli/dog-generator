from keras.layers import Dense, Conv2D, Conv2DTranspose, BatchNormalization, Activation, Reshape, LeakyReLU,\
    Dropout, Flatten
from keras.models import Sequential

KERNEL_SIZE_GENERATOR = 3
KERNEL_SIZE_DISCRIMINATOR = 3

def create_generator(latent_size, img_size, starting_filters=64):
    """
    Creates a generator model for use in a DCGAN.
    :param latent_size:
    :param img_size:
    :param starting_filters: number of maps to start with from latent
    :return:
    """
    model = Sequential()

    # block 0
    model.add(Dense(starting_filters*(img_size[0] // (2 ** 5))  *  (img_size[1] // (2 ** 5)),
                    activation="relu",
                    input_shape=(latent_size,)))
    # first layer dimensions img_size / (2^number_of_upsampling_layers)
    model.add(Reshape(((img_size[0] // (2 ** 5)),
                       (img_size[1] // (2 ** 5)),
                       starting_filters)))
    model.add(BatchNormalization(momentum=0.8))

    # block 1
    # model.add(UpSampling2D())
    model.add(Conv2DTranspose(1024, kernel_size=KERNEL_SIZE_GENERATOR, padding="same", strides=2))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dropout(0.2))

    # block 2
    # model.add(UpSampling2D())
    model.add(Conv2DTranspose(512, kernel_size=KERNEL_SIZE_GENERATOR, padding="same", strides=2))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dropout(0.2))

    # block 3
    # model.add(UpSampling2D())
    model.add(Conv2DTranspose(256, kernel_size=KERNEL_SIZE_GENERATOR, padding="same", strides=2))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))

    # block 4
    # model.add(UpSampling2D())
    model.add(Conv2DTranspose(128, kernel_size=KERNEL_SIZE_GENERATOR, padding="same", strides=2))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))

    # block 5
    # model.add(UpSampling2D())
    model.add(Conv2DTranspose(64, kernel_size=KERNEL_SIZE_GENERATOR, padding="same", strides=2))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))

    # block 6 without upsampling
    model.add(Conv2D(32, kernel_size=KERNEL_SIZE_GENERATOR, padding="same", strides=1))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))

    # final image
    model.add(Conv2D(3, kernel_size=KERNEL_SIZE_GENERATOR, padding="same", strides=1))
    model.add(Activation("tanh"))

    model.summary()

    return model

def create_discriminator(img_size):
    """
    Creates a discriminator model for use in a DCGAN.
    :param img_size:
    :return:
    """
    model = Sequential()

    model.add(Conv2D(64, kernel_size=KERNEL_SIZE_DISCRIMINATOR, strides=2, padding="same", input_shape=(img_size[0],img_size[1],3)))
    model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2D(128, kernel_size=KERNEL_SIZE_DISCRIMINATOR, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2D(256, kernel_size=KERNEL_SIZE_DISCRIMINATOR, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2D(512, kernel_size=KERNEL_SIZE_DISCRIMINATOR, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    return model


