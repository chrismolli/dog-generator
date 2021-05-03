import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

def plot_csv_log(model_directory):
    # read log
    log = pd.read_csv(model_directory+"/log.csv", sep=",")

    fig, ax1 = plt.subplots(figsize=(5,2))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Wasserstein Distance', color="C0")
    ax1.plot(log["epoch"], log["w_distance"], label="c_loss_real", color="C0")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('Generator Loss', color="C2")  # we already handled the x-label with ax1
    ax2.plot(log["epoch"],log["g_loss"],label="g_loss", color="C2")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.savefig(model_directory+"/loss.pdf")
    plt.close()

def plot_csv_log_v2(model_directory, max_epoch=None):
    # read log
    log = pd.read_csv(model_directory+"/log.csv", sep=",")

    if max_epoch:
        log = log[:max_epoch]

    fig, ax1 = plt.subplots(figsize=(5,3))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Generator Loss', color="C2")  # we already handled the x-label with ax1
    ax1.plot(log["epoch"],log["g_loss"],label="g_loss", color="C2")
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    if max_epoch:
        ax1.set_xlim([0,max_epoch])

    plt.savefig(model_directory+"/generator_loss.pdf")
    plt.close()

    fig, ax1 = plt.subplots(figsize=(5, 2))

    ax1.set_xlabel('Epoch')

    ax1.set_ylabel('Wasserstein Distance', color="C0")
    ax1.plot(log["epoch"], log["w_distance"], label="c_loss_real", color="C0")
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    if max_epoch:
        ax1.set_xlim([0,max_epoch])

    plt.savefig(model_directory + "/wassertstein_distance.pdf")
    plt.close()