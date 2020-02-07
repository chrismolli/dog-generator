import os
from datetime import datetime
from glob import glob
import numpy as np

import keras
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, CSVLogger
from keras.datasets import cifar10
from keras.models import load_model
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator

from mixup import mixup_extend_data
from plotter_functions import schedule_plotter, save_log_plots

def train_existent_model(model_directory, batch_size=128, epochs=150):
    """ hyperparameter """
    # learning rates
    learning_rate = 0.2

    # epochs
    early_stopping_patience = 0

    # mixup extension
    mixup_extension_percentage = 1

    """  get latest checkpoint and load model """
    checkpoints = glob(model_directory+"/*.hdf5")
    current_epoch = 0
    latest_checkpoint = None
    for checkpoint in checkpoints:
        epoch = int(checkpoint.split("_")[-2][5:])
        if epoch > current_epoch:
            current_epoch = epoch
            latest_checkpoint = checkpoint
    model = load_model(latest_checkpoint)

    """ setup optimizer and compile model """
    optimizer = SGD(lr=learning_rate, decay=1e-6, momentum=0.9)
    # optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    """ load training data """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # set data type images
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # normalize color values to -1 to 1
    x_train = (x_train - 127.5) / 127.5
    x_test = (x_test - 127.5) / 127.5

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    """ data augmentation"""
    # perform mixup
    n_mixup = int(mixup_extension_percentage * x_train.shape[0])
    (x_train, y_train) = mixup_extend_data(x_train, y_train, n_mixup)

    # prepare data generators for training and testing
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=20,
        zoom_range=[1.0, 1.2],
        horizontal_flip=True)

    testdatagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
    )

    # fit data generators for feature normalization
    datagen.fit(x_train)
    testdatagen.fit(x_train)

    """ configure callbacks """
    # early stopping
    if early_stopping_patience > 0:
        early_stop = EarlyStopping(monitor='val_loss', patience=early_stopping_patience)

    # learning rate annealing
    def sigmoid_schedule(epoch):
        if epoch < 40:
            return 0.005 + (0.25 - 0.005) / (1 + np.exp(0.3 * (epoch - 20)))
        else:
            return 0.005 + (0.25 - 0.005) / (1 + np.exp(0.1 * (epoch - 60)))

    def cyclic_schedule(epoch, intervall, min_lr, max_lr):
        epoch %= intervall
        return min_lr + 1 / 2 * (max_lr - min_lr)*(1 + np.cos(epoch / intervall * np.pi))

    def schedule(epoch):
        return cyclic_schedule(epoch, 40, 0.001, 0.3)

    lr_scheduler = LearningRateScheduler(schedule, verbose=0)

    # csv logger
    csv_logger = CSVLogger(model_directory+'/training_log.csv', append=True, separator=';')

    """ train """
    # setup checkpoint callback
    os.makedirs(model_directory, exist_ok=True)
    checkpoint = ModelCheckpoint(
        model_directory + "/date{}_".format(datetime.now().strftime("%Y%m%d%H%M%S")) + "epoch{epoch:02d}_valacc{val_acc:.4f}.hdf5",
        monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # plot learning schedule
    schedule_plotter(model_directory + "/learning_schedule.pdf", schedule, epochs)
    # collect all callbacks
    if early_stopping_patience > 0:
        callbacks = [lr_scheduler, early_stop, checkpoint, csv_logger]
    else:
        callbacks = [lr_scheduler, checkpoint, csv_logger]
    # training
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                  steps_per_epoch=len(x_train) / batch_size,
                                  epochs=epochs,
                                  validation_data=testdatagen.flow(x_test, y_test),
                                  callbacks=callbacks,
                                  verbose=1,
                                  initial_epoch=current_epoch)
    # save log plots
    save_log_plots(model_directory + "/training_log.csv", model_directory)
