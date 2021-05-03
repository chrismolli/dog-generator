import matplotlib.pyplot as plt
import pandas as pd

def plot_training_log(fname, metric="acc"):
    # load log data
    log = pd.read_csv(fname,sep=';')
    # setup plot
    fig = plt.figure(figsize=(7,4))
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    # plot accuracy
    plt.plot(log[metric],label='training {}'.format(metric))
    plt.plot(log["val_{}".format(metric)],label="validation {}".format(metric))
    # settings
    plt.legend()
    return fig

def save_log_plots(log_path, save_dir):
    fig1 = plot_training_log(log_path)
    plt.savefig(save_dir + "/accuracy_log.pdf")
    fig2 = plot_training_log(log_path, "loss")
    plt.savefig(save_dir + "/loss_log.pdf")

def schedule_plotter(fname, schedule, epochs):
    x = list(range(epochs + 1))
    y = [schedule(x_) for x_ in x]
    plt.figure(figsize=(5, 2.5))
    plt.plot(x, y, '-')
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.xlim([0, epochs])
    plt.title("Learning Rate Schedule")
    plt.subplots_adjust(left=0.16, bottom=0.22, top=0.81)
    plt.savefig(fname)


if __name__ == '__main__':
    fig1 = plot_training_log('training_log.csv')
    fig2 = plot_training_log('training_log.csv',"loss")
    plt.show()
