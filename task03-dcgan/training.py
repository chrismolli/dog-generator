from model_definition import create_discriminator, create_generator
from keras.models import Model, load_model
from keras.layers import Input
from keras.optimizers import Adam
import pickle
import numpy as np
import os
from generation import generate_save_image_gallery
from csv_logging import log_to_csv, read_latest_log_entry
import progressbar

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import io
from plotting import plot_csv_log

def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

def train(model_directory, latent_size, epochs=100, batch_size=32, dataset_path="", checkpoint_every = 1):
    """
    Creates new DCGAN to train on dataset from dataset_path.
    :param model_directory:
    :param latent_size:
    :param epochs:
    :param batch_size:
    :return:
    """

    """ load data """
    with open(dataset_path,"rb") as file:
        x_train = pickle.load(file)
    # infer image size from dataset
    img_size = x_train.shape[1:3]
    # rescale images from -1 to 1
    x_train = (x_train.astype("float32") - 127.5) / 127.5
    print(f"Loaded dataset of {x_train.shape}")

    """ setup optimzer """
    optimizer = Adam(0.0002,0.5)

    """ compose gan from generator and discriminator or load if existent """
    def create_gan(optimizer, latent_size, img_size):
        # build and compile generator
        generator = create_generator(latent_size=latent_size,img_size=img_size)
        generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # build and compile discriminator
        discriminator = create_discriminator(img_size=img_size)
        discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # setup training variables
        z = Input(shape=(latent_size,))
        fake_img = generator(z)

        discriminator.trainable = False
        valid = discriminator(fake_img)

        # build and compile combined image
        combined = Model(z, valid)
        combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        return (combined, discriminator, generator)

    def load_gan(optimizer):
        # load discriminator and generator, try backup file if failed
        try:
            discriminator = load_model(model_directory+"/discriminator.h5")
        except:
            discriminator = load_model(model_directory + "/discriminator_backup.h5")

        try:
            generator = load_model(model_directory + "/generator.h5")
        except:
            generator = load_model(model_directory + "/generator_backup.h5")

        # setup training variables
        z = Input(shape=(latent_size,))
        fake_img = generator(z)

        discriminator.trainable = False
        valid = discriminator(fake_img)

        # build and compile combined image
        combined = Model(z, valid)
        combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        return (combined, discriminator, generator)

    try:
        # continue training
        (combined, discriminator, generator) = load_gan(optimizer)
        current_epoch = read_latest_log_entry(model_directory)
        print(f"\nContinuing from Epoch {current_epoch}")
    except:
        # start new training
        (combined, discriminator, generator) = create_gan(optimizer,latent_size,img_size)
        current_epoch = -1
        print("\nCreating new model!")

    # save model summaries
    os.makedirs(model_directory, exist_ok=True)
    with open(model_directory+"/dis_sum.txt","w+") as file:
        file.write(get_model_summary(discriminator))
    with open(model_directory+"/gen_sum.txt","w+") as file:
        file.write(get_model_summary(generator))

    # training
    print("\nStarting training...")
    half_batch = batch_size // 2
    batches_per_epoch = x_train.shape[0] // batch_size

    for epoch in range(current_epoch+1,epochs):
        print(f"\nTraining on Epoch {epoch}...")
        with progressbar.ProgressBar(max_value=batches_per_epoch) as bar:

            # init metrics
            g_loss_epoch = 0
            d_loss_epoch = np.zeros([2,1]).squeeze()

            # train
            for batch in range(batches_per_epoch):
                # train generator
                discriminator.trainable = False
                noise = np.random.normal(0, 1, (batch_size, latent_size))
                g_labels = np.random.rand(batch_size, 1) * 0.3 + 0.7  # flipped soft label
                g_loss = combined.train_on_batch(noise, g_labels)
                g_loss_epoch += g_loss

                # take half batch of real images
                idx = np.random.randint(0, x_train.shape[0], half_batch)
                imgs = x_train[idx]

                # sample noise and generate a half batch of new images
                noise = np.random.normal(0, 1, (half_batch, latent_size))
                gen_imgs = generator.predict(noise)

                # create soft labels
                real_labels = 0.3 * np.random.rand(half_batch, 1) + 0.7
                fake_labels = 0.3 * np.random.rand(half_batch, 1)

                # train discriminator
                discriminator.trainable = True
                d_loss_real = discriminator.train_on_batch(imgs, real_labels)
                d_loss_fake = discriminator.train_on_batch(gen_imgs, fake_labels)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                d_loss_epoch = np.add(d_loss_epoch,d_loss)

                # update epoch progress
                bar.update(batch)

        # print loss
        d_loss_epoch /= batches_per_epoch
        g_loss_epoch /= batches_per_epoch
        print(f"\nFinished Epoch {epoch} d_loss {d_loss_epoch}, g_loss {g_loss_epoch}")

        # save training history
        log_to_csv(model_directory, epoch, d_loss_epoch, g_loss_epoch)

        # checkpoint epoch wise
        if epoch % checkpoint_every == 0 or epoch == epochs - 1:
            generate_save_image_gallery(model_directory, generator, epoch, latent_size=latent_size)
            discriminator.save(model_directory + "/discriminator.h5")
            discriminator.save(model_directory + "/discriminator_backup.h5")
            generator.save(model_directory + "/generator.h5")
            generator.save(model_directory + "/generator_backup.h5")
            plot_csv_log(model_directory)

if __name__ == "__main__":
    latent_size = 100
    epochs = 100
    batch_size = 128
    dataset_path = "../datasets/dogset_d2.bin"
    if os._exists(dataset_path):
        train("local_test", latent_size, epochs, batch_size, dataset_path)
    else:
        print("##########################")
        print(f"Could not find dataset {dataset_path}. Please download the datasets first using the provided Dropbox Link!")
        print("##########################")