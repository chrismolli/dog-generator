import os
import pandas as pd
import io

def log_to_csv(model_directory, epoch, w_distance, g_loss):
    log_path = model_directory + "/log.csv"
    # check if file exist
    if not os.path.exists(log_path):
        with open(log_path, "w+") as file:
            file.write("epoch,w_distance,g_loss\n")
    # append log entry
    with open(log_path, "a") as file:
        file.write(f"{epoch},{w_distance},{g_loss}\n")

def read_latest_log_entry(model_directory):
    log_path = model_directory + "/log.csv"
    if os.path.exists(log_path):
        log = pd.read_csv(log_path, sep=",")
        return log["epoch"].iloc[-1]
    else:
        return np.array([0, 0, 0, 0])

def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string