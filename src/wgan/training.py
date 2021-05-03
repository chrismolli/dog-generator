import os
import pickle

import numpy as np
import progressbar
from keras.optimizers import RMSprop, Adam

from csv_logging import log_to_csv, read_latest_log_entry, get_model_summary
from generation import generate_save_image_gallery
from model_definition import create_wgan, load_wgan
from plotting import plot_csv_log

# ugly monkey patching of keras to allow model loading ...
import keras
from model_definition import WeightClipper
from wasserstein import wasserstein_loss
keras.losses.wasserstein_loss = wasserstein_loss
keras.constraints.WeightClipper = WeightClipper

def train(model_directory, latent_size, epochs=100, batch_size=64, learning_rate=5e-5, critic_clipping=1e-2, n_critic=5, dataset_path="", checkpoint_every=1):
    """
    :param model_directory:
    :param latent_size:
    :param epochs:
    :param batch_size:
    :param learning_rate:
    :param critic_clipping:
    :param n_critic:
    :param dataset_path:
    :param checkpoint_every:
    :return:
    """

    """ load dataset """
    with open(dataset_path, "rb") as file:
        x_train = pickle.load(file)
    # infer image size from dataset
    img_size = x_train.shape[1:3]
    # rescale images from -1 to 1
    x_train = (x_train.astype("float32") - 127.5) / 127.5
    print(f"Loaded dataset of {x_train.shape}")

    """ setup optimizer """
    optimizer = RMSprop(lr=learning_rate)

    """ setup model """
    try:
        combined, critic, generator = load_wgan(model_directory, latent_size, optimizer)
        current_epoch = read_latest_log_entry(model_directory)
        print(f"\nContinuing from Epoch {current_epoch}")
    except:
        combined, critic, generator = create_wgan(optimizer, latent_size, img_size, critic_clipping)
        current_epoch = -1
        print("\nCreating new model!")

    # save model summaries
    os.makedirs(model_directory, exist_ok=True)
    with open(model_directory + "/critic_sum.txt", "w+") as file:
        file.write(get_model_summary(critic))
    with open(model_directory + "/generator_sum.txt", "w+") as file:
        file.write(get_model_summary(generator))

    """ calc batches per epoch """
    half_batch = batch_size // 2
    batches_per_epoch = x_train.shape[0] // batch_size

    print("cr-1 cf+1 gf-1")

    """ training """
    for epoch in range(current_epoch+1, epochs):
        print(f"\nTraining on Epoch {epoch}...")
        c_loss_real, c_loss_fake, g_loss = 0, 0, 0
        with progressbar.ProgressBar(max_value=batches_per_epoch) as bar:
            for batch in range(batches_per_epoch):
                # train critic (discriminator) model
                critic.trainable = True
                for critic_update in range(n_critic):
                    # sample halfbatch of real data
                    x_real = x_train[np.random.randint(0, x_train.shape[0], half_batch)]
                    y_real = (-1)*np.ones([half_batch,1])
                    # sample halfbatch of fake data
                    noise = np.random.normal(0, 1, (half_batch, latent_size))
                    x_fake = generator.predict(noise)
                    y_fake = (1)*np.ones([half_batch,1])
                    # train discriminator on batch
                    # (weight clipping is done inside the model)
                    c_loss_real += critic.train_on_batch(x_real,y_real)
                    c_loss_fake += critic.train_on_batch(x_fake,y_fake)
                c_loss_real /= n_critic
                c_loss_fake /= n_critic
                # train generator model
                # sample full batch of fake data
                x_fake = np.random.normal(0, 1, (batch_size, latent_size))
                y_fake = (-1)*np.ones([batch_size, 1])
                # freeze critic
                critic.trainable = False
                # train generator-critic on batch
                g_loss += combined.train_on_batch(x_fake, y_fake)
                bar.update(batch)

        # logging
        c_loss_real /= batches_per_epoch
        c_loss_fake /= batches_per_epoch
        g_loss /= batches_per_epoch
        w_distance = - (c_loss_fake + c_loss_real)
        log_to_csv(model_directory,epoch, w_distance, g_loss)
        print(f"Finished Epoch {epoch}, w_distance {w_distance}, c_fake {c_loss_fake}, c_real {c_loss_real}, g_loss {g_loss}")

        # checkpoints
        if epoch % checkpoint_every == 0 or epoch == epochs - 1:
            generate_save_image_gallery(model_directory, generator, epoch, latent_size=latent_size)
            critic.save(model_directory + "/critic.h5")
            critic.save(model_directory + "/critic_backup.h5")
            generator.save(model_directory + "/generator.h5")
            generator.save(model_directory + "/generator_backup.h5")
            plot_csv_log(model_directory)

if __name__ == "__main__":
    latent_size = 100
    epochs = 100
    batch_size = 128
    dataset_path = "../datasets/dogset_d2.bin"
    if os._exists(dataset_path):
        train("local_test", latent_size=latent_size, batch_size=batch_size, epochs=epochs, dataset_path=dataset_path)
    else:
        print("##########################")
        print(
            f"Could not find dataset {dataset_path}. Please download the datasets first using the provided Dropbox Link!")
        print("##########################")

