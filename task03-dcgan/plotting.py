import matplotlib.pyplot as plt
import pandas as pd

def plot_csv_log(model_directory, batch_size=128, d_size=1e6, max_it=6500, iter_mode=False, text=""):
    # read log
    log = pd.read_csv(model_directory+"/log.csv", sep=",")

    # create plot
    fig, ax1 = plt.subplots(figsize=(5,2))

    if iter_mode:
        line1 = ax1.plot(log["epoch"]*d_size/batch_size, log["d_loss"], label="d_loss", color="C0")
        ax1.set_xlabel('Training Iteration')
        ax1.set_ylabel('Discriminator Loss', color="C0")

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        line2 = ax2.plot(log["epoch"]*d_size/batch_size, log["g_loss"], label="g_loss", color="C1")
        ax2.set_ylabel('Generator Loss', color="C1")  # we already handled the x-label with ax1

        ax1.set_xlim([0,max_it])
    else:
        line1 = ax1.plot(log["epoch"]+1, log["d_loss"], label="d_loss", color="C0")
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Discriminator Loss', color="C0")

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        line2 = ax2.plot(log["epoch"]+1, log["g_loss"], label="g_loss", color="C1")
        ax2.set_ylabel('Generator Loss', color="C1")  # we already handled the x-label with ax1

    plt.title(text)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    ax1.set_xlim([1, 200])



    plt.savefig(model_directory + "/loss.pdf")
    plt.close()
