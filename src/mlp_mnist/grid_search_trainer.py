from plotter import plot_history
import os
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from keras.optimizers import Adam, SGD
from model import create_model
from dataset import load_mnist
from datetime import datetime
import pickle
import time
from lib.mailbot import Mailbot
import traceback
from keras.preprocessing.image import ImageDataGenerator

""" save folder """
sname = "models_da_block_sgd_lrreboost"

""" setup mailbot for training notifications """
mailbot = Mailbot("lib/mailbot.json")

""" Hyperparameters """
# layer stacks
small_stack = [256,128,64,32]
medium_stack = [512,512,256,128,64,32]
big_stack = [512,512,512,256,256,256,128,64,32]
block_stack = [1024,1024,1024]
layer_stacks = {"block":block_stack}

# learning rates
learning_rates = [0.1]

# gaussian noise stddevs
gn_stddevs = [0.1,0.2,0.3,0.4]

# hidden layer activations
activations = ["relu"]

# epochs
epochs = 250
early_stopping_patience = -1

# dropout
dropout_rate = [0.3,0.5]

# batchsize
batch_size = 128

# regularization
l2_reg = 0.00


""" prepare data """
(x_train, y_train), (x_test, y_test) = load_mnist()



""" configure data augmentation """
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range = 5,
    horizontal_flip=False,
    vertical_flip=False)


""" configure callbacks """
# reduce lr on plateau
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                               patience=3, min_lr=0.001)

# early stopping
if early_stopping_patience > 0:
    early_stop = EarlyStopping(monitor='val_loss', patience=early_stopping_patience)


# learning rate annealing
def schedule(epoch):
    if epoch < 25:
        return 0.1
    elif epoch < 50:
        return 0.01
    elif epoch < 100:
        return 0.001
    elif epoch < 120:
        return 0.1
    elif epoch < 150:
        return 0.01
    else:
        return 0.001

lr_scheduler = LearningRateScheduler(schedule, verbose=0)



""" gridsearch """

t = time.time()
best_val_acc = 0

try:
    for layer_stack in layer_stacks:
        layers = layer_stacks[layer_stack]
        for lr in learning_rates:
            for gn in gn_stddevs:
                for activation in activations:
                    for dr in dropout_rate:
                        # configure optimizer
                        # optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
                        optimizer = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
                        # compile model
                        model = create_model(layers, dr, True, gn, activation, l2_reg)
                        model.compile(loss='categorical_crossentropy',
                                      optimizer=optimizer,
                                      metrics=['accuracy'])
                        # compile folder name
                        pname = "{}/{}_lr{}_gn{}_{}_dr{}_l2{}/".format(sname,layer_stack,lr,gn,activation,dr,l2_reg)
                        # setup checkpoint callback
                        os.makedirs(pname, exist_ok=True)
                        checkpoint = ModelCheckpoint(pname + "date{}_".format(datetime.now().strftime("%Y%m%d%H%M%S")) + "epoch{epoch:02d}_valacc{val_accuracy:.2f}.hdf5",
                                                     monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
                        # collect all callbacks
                        if early_stopping_patience > 0:
                            callbacks = [lr_scheduler, early_stop, checkpoint]
                        else:
                            callbacks = [lr_scheduler, checkpoint]
                        # training
                        history=model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                            steps_per_epoch=len(x_train) / batch_size, 
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            callbacks=callbacks,
                            verbose=1)
                        # plot training
                        plot_history(history,pname)
                        # save history
                        with open(pname+"history.bin", 'wb') as file:
                            pickle.dump(history, file)
                        # rename pname
                        os.rename(pname, pname[:-1]+"_valacc{:.4f}/".format(max(history.history['val_accuracy'])))
                        # save best val acc for email
                        if max(history.history['val_accuracy']) > best_val_acc:
                            best_val_acc = max(history.history['val_accuracy'])
except:
    """ notify if training failed """
    elapsed = time.time() - t
    error_message = traceback.format_exc()
    subject = "[BLUELAGOON] Training failed!"
    message = "Training Time {}\n{}".format(elapsed, error_message)
    mailbot.send(message, subject)

else:
    """ notify if training finished """
    elapsed = time.time() - t
    subject = "[BLUELAGOON] Training finished!"
    message = "Training Time {}\nBest Validation Acc".format(elapsed,best_val_acc)
    mailbot.send(message,subject)