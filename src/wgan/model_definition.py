from keras import backend
from keras.constraints import Constraint
from keras.initializers import RandomNormal
from keras.layers import Dense, Conv2D, Conv2DTranspose, BatchNormalization, Activation, Reshape, LeakyReLU, \
    Dropout, Flatten
from keras.models import Sequential, load_model

from wasserstein import wasserstein_loss

KERNEL_SIZE_GENERATOR = 3
KERNEL_SIZE_CRITIC = 3

""" weight clipper """
class WeightClipper(Constraint):
    # clip model weights to a given hypercube
    def __init__(self, clip_value):
        self.clip_value = clip_value

    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)

    def get_config(self):
        return {'clip_value': self.clip_value}

""" model defintions """
def create_generator(latent_size, img_size, starting_filters=64):
    """
    Creates a generator model for use in a WGAN.
    :param latent_size:
    :param img_size:
    :param starting_filters: number of maps to start with from latent
    :return:
    """

    init = RandomNormal(stddev=0.02)

    model = Sequential()

    # block 0
    model.add(Dense(starting_filters*(img_size[0] // (2 ** 5))  *  (img_size[1] // (2 ** 5)),
                    input_shape=(latent_size,)))
    # model.add(LeakyReLU())
    model.add(Activation("relu"))
    model.add(Reshape(((img_size[0] // (2 ** 5)), (img_size[1] // (2 ** 5)), starting_filters)))
    model.add(BatchNormalization())

    # block 1
    model.add(Conv2DTranspose(1024, kernel_size=KERNEL_SIZE_GENERATOR, padding="same", strides=2, kernel_initializer=init))
    # model.add(LeakyReLU())
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # block 2
    model.add(Conv2DTranspose(512, kernel_size=KERNEL_SIZE_GENERATOR, padding="same", strides=2, kernel_initializer=init))
    # model.add(LeakyReLU())
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # block 3
    model.add(Conv2DTranspose(256, kernel_size=KERNEL_SIZE_GENERATOR, padding="same", strides=2, kernel_initializer=init))
    # model.add(LeakyReLU())
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))

    # block 4
    model.add(Conv2DTranspose(128, kernel_size=KERNEL_SIZE_GENERATOR, padding="same", strides=2, kernel_initializer=init))
    # model.add(LeakyReLU())
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    # block 5
    model.add(Conv2DTranspose(64, kernel_size=KERNEL_SIZE_GENERATOR, padding="same", strides=2, kernel_initializer=init))
    # model.add(LeakyReLU())
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    # block 6 without upsampling
    model.add(Conv2D(32, kernel_size=KERNEL_SIZE_GENERATOR, padding="same", strides=1, kernel_initializer=init))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))

    # final image
    model.add(Conv2D(3, kernel_size=KERNEL_SIZE_GENERATOR, padding="same", strides=1, kernel_initializer=init))
    model.add(Activation("tanh"))

    return model

def create_critic(img_size, critic_clipping=1e-2):
    """
    Creates a critic model for use in a WGAN.
    :param img_size:
    :return:
    """

    init = RandomNormal(stddev=0.02)
    clipper = WeightClipper(critic_clipping)

    model = Sequential()

    model.add(Conv2D(64, kernel_size=KERNEL_SIZE_CRITIC, strides=2, padding="same",
                     kernel_constraint=clipper,
                     kernel_initializer=init,
                     input_shape=(img_size[0],img_size[1],3,)))
    model.add(LeakyReLU())
    model.add(BatchNormalization())

    model.add(Conv2D(128, kernel_size=KERNEL_SIZE_CRITIC, strides=2, padding="same",
                     kernel_initializer=init,
                     kernel_constraint=clipper))
    model.add(LeakyReLU())
    model.add(BatchNormalization())

    model.add(Conv2D(256, kernel_size=KERNEL_SIZE_CRITIC, strides=2, padding="same",
                     kernel_initializer=init,
                     kernel_constraint=clipper))
    model.add(LeakyReLU())
    model.add(BatchNormalization())

    model.add(Conv2D(512, kernel_size=KERNEL_SIZE_CRITIC, strides=2, padding="same",
                     kernel_initializer=init,
                     kernel_constraint=clipper))
    model.add(LeakyReLU())
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(1, kernel_initializer=init))

    return model

def create_wgan(optimizer, latent_size, img_size, critic_clipping):
    # build and compile generator
    generator = create_generator(latent_size=latent_size, img_size=img_size)

    # build and compile discriminator
    critic = create_critic(img_size=img_size, critic_clipping=critic_clipping)
    critic.compile(loss=wasserstein_loss, optimizer=optimizer)

    # build and compile combined image
    critic.trainable = False
    combined = Sequential()
    combined.add(generator)
    combined.add(critic)
    combined.compile(loss=wasserstein_loss, optimizer=optimizer)

    return combined, critic, generator


def load_wgan(model_directory, latent_size, optimizer):
    # load discriminator and generator, try backup file if failed
    try:
        critic = load_model(model_directory+"/critic.h5")
    except:
        critic = load_model(model_directory + "/critic_backup.h5")

    try:
        generator = load_model(model_directory + "/generator.h5")
    except:
        generator = load_model(model_directory + "/generator_backup.h5")

    # build and compile combined image
    critic.trainable = False
    combined = Sequential()
    combined.add(generator)
    combined.add(critic)
    combined.compile(loss=wasserstein_loss, optimizer=optimizer)

    return combined, critic, generator


