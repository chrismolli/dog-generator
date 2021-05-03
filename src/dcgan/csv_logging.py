import os
import pandas as pd
import numpy as np

def log_to_csv(model_directory, epoch, d_loss, g_loss):
    log_path = model_directory + "/log.csv"
    # check if file exist
    if not os.path.exists(log_path):
        with open(log_path, "w+") as file:
            file.write("epoch,d_loss,g_loss\n")
    # append log entry
    with open(log_path, "a") as file:
        file.write(f"{epoch},{d_loss[0]},{g_loss}\n")

def read_latest_log_entry(model_directory):
    log_path = model_directory + "/log.csv"
    if os.path.exists(log_path):
        log = pd.read_csv(log_path, sep=",")
        return log["epoch"].iloc[-1]
    else:
        return np.array([0, 0, 0, 0])
