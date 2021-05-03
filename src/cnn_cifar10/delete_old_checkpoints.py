from glob import glob
import os

def delete_old_checkpoints(model_directory):
    """  get latest checkpoint and load model """
    checkpoints = glob(model_directory+"*.hdf5")
    current_epoch = 0
    latest_checkpoint = None
    for checkpoint in checkpoints:
        epoch = int(checkpoint.split("_")[-2][5:])
        if epoch > current_epoch:
            current_epoch = epoch
            latest_checkpoint = checkpoint
    """ delete checkpoints """
    for checkpoint in checkpoints:
        if checkpoint != latest_checkpoint:
            os.remove(checkpoint)