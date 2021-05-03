from keras import backend

def wasserstein_loss(y_true, y_pred):
    """ implementation of the wasserstein loss """
    return backend.mean(y_true * y_pred)

