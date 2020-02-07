import numpy as np
import os
try:
    from imageio import imwrite
except:
    from scipy.misc import imsave as imwrite
import matplotlib.pyplot as plt

def generate_images(n_images, generator, latent_size):
    noise = np.random.normal(0, 1, (n_images, latent_size))
    imgs = generator.predict(noise)
    imgs *= 127.5 # rescale back to 0-255 image
    imgs += 127.5
    return imgs.astype("uint8")

def generate_save_images(pname, n_images, generator, epoch, latent_size=100):
    # generate images
    gen_imgs = generate_images(n_images, generator, latent_size)
    # gen_imgs = 0.5 * gen_imgs + 0.5

    # create folder
    pname = pname + "/samples"
    os.makedirs(pname,exist_ok=True)

    # save images
    for i, img in enumerate(gen_imgs):
        imwrite(pname + f"/{epoch}_{i}.png", img)

def generate_save_image_gallery(pname, generator, epoch, n_columns=10, n_rows=5, latent_size=100):
    # generate images
    gen_imgs = generate_images(n_columns*n_rows, generator, latent_size)
    gen_imgs = np.stack(gen_imgs, axis =0)

    # create folder
    pname = pname + "/samples"
    os.makedirs(pname, exist_ok=True)

    # create gallery
    img_size = gen_imgs[0].shape
    gallery_rows = []
    for col in range(n_columns):
        gallery_row = [gen_imgs[col + n_columns * row, :, :, :].squeeze() for row in range(n_rows)]
        gallery_rows.append(np.vstack(gallery_row))
    gallery = np.hstack(gallery_rows)

    # save gallery
    plt.imsave(pname+f"/gallery_e{epoch}.png", gallery)