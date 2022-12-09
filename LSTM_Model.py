import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras import Sequential

from sklearn.model_selection import KFold
import time



####################################################

early_stopping = True 
monitor = 'loss' 
patience = 4
min_delta = 0.01

## CustomVerbose() ... a custom callback 
custom_verbose= False 
train_verbose = True #if custom_verbose = False then would you like default verbose (True) or all verbose off (False) when training 

## learning_rate_scheduler
learning_rate_scheduler_bool = True

metrics = ["accuracy"]

## model_checkpoint callback creator. Choices: 'both', 'model', 'weights'
model_and_weights_saved = 'None' 
model_results_path = '/home/openvessel/Documents/Data Science/LSTM_Stock_Project/LSTM_Model_Results_4'

## tensorboard 
tensorboard = True 

## ReduceLROnPlateau ? 
reduce_lr_on_plateau = True 




########################################################


from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, Callback, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorboard.plugins.hparams import api as hp
import os
from datetime import datetime, date
import math


class Custom_Verbose(Callback): #####################
    def __init__(self, metrics):
        self.metrics = metrics  
    
    def on_train_begin(self, logs = None): #####################
        print("The training session has begun...")
    
    def on_epoch_begin(self, epoch, logs = None):
        print("Epoch #{} has begun...".format(epoch)) #####################
    
    def on_epoch_end(self, epoch, logs={}):
        print("Epoch #{} has ended.".format(epoch))
        print("\tResults for Epoch#{}:".format(epoch)) #####################
        #for i in self.metrics: 
        print("\t\t loss = {:7.2f}".format(logs["loss"]))
        print("\t\t accuracy = {:7.2f}".format(logs['accuracy']))

    def on_train_end(self, logs = {}): #####################
        print("Training has ended.")
        print("\tTraining Results:")
        #for i in self.metrics: 
        print("\t\t loss = {:7.2f}".format(logs["loss"]))
        print("\t\t accuracy = {:7.2f}".format(logs['accuracy']))


def learning_rate_scheduler(epoch):
    initial_rate = 0.3
    drop_rate = 0.035
    epoch_interval_to_drop = 2 # every "x" epochs you drop the initial rate by the drop rate

    lrate = initial_rate - ((epoch-1) * drop_rate)


    # Original Scheduler but I modified the numbers 

    # initial_lrate = 0.3  #originally 0.1
    # drop = 0.01  # look up default 
    # epochs_drop = 2



    # lrate = initial_lrate * math.pow(drop,  
    #         math.floor((1+epoch)/epochs_drop))
    return lrate


def callback_selector(early_stopping, monitor, patience, min_delta, learning_rate_scheduler_bool, custom_verbose, metrics, model_and_weights_saved, tensorboard, reduce_lr_on_plateau, model_results_path):
    
    callbacks = [] 

    now = datetime.now()

    date_and_time = now.strftime("%d/%m/%Y %H:%M:%S")

    #### ModelCheckpoint() / models_and_weights_saved

    if model_and_weights_saved == 'both': 
        checkpoint = ModelCheckpoint('Liver_Detection-version:{}'.format(date_and_time)) #####################
        callbacks.append(checkpoint)

    elif model_and_weights_saved == 'model':
        checkpoint = ModelCheckpoint('Liver_Detection-version:{}'.format(date_and_time), save_model_only = True) #####################
        callbacks.append(checkpoint) 

    elif model_and_weights_saved == 'weights':
        checkpoint = ModelCheckpoint('Liver_Detection-version:{}'.format(date_and_time), save_weights_only = True)
        callbacks.append(checkpoint) 

    #### EarlyStopping / early_stopping 
    if early_stopping == True:
        early_stopping_object = EarlyStopping(monitor = monitor, patience = patience, min_delta = min_delta) ##################### finish 
        callbacks.append(early_stopping_object)

    #### LearningRateScheduler / learning_rate_scheduler

    if learning_rate_scheduler_bool == True: 
        learning_rate_scheduler_object = LearningRateScheduler(learning_rate_scheduler) ##################### finish 
        callbacks.append(learning_rate_scheduler_object)

    #### Custom_Verbose / custom_verbose
    if custom_verbose == True:
        callbacks.append(Custom_Verbose(metrics = metrics)) #####################

    model_path = ''
    if tensorboard == True: 
        
        model_path = os.path.join(model_results_path, 'Model_{}'.format(datetime.now().strftime("%m%d--%H%M")))
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        
        tensorboard_model_path = os.path.join(model_path, "Tensorboard_Results")
        if not os.path.exists(tensorboard_model_path):
            os.mkdir(tensorboard_model_path)

        callbacks.append(TensorBoard(log_dir = tensorboard_model_path, update_freq = 'epoch'))
    
    if reduce_lr_on_plateau == True:
        callbacks.append(ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 2, min_lr= 0.001))


    #WHEN WE WANT TO DO HYPERPARAMETER TUNING: 
        # HP_NUM_UNITS=hp.HParam('num_units', hp.Discrete([64, 128]))
        # HP_DROPOUT=hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
        # HP_LEARNING_RATE= hp.HParam('learning_rate', hp.Discrete([0.001, 0.0005, 0.0001]))
        # HP_OPTIMIZER=hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'rmsprop']))
        # METRIC_ACCURACY='accuracy'
        
        # log_dir = '/home/openvessel/Documents/OpenVessel/Liver_Detection_2.0_Results/Tensorboard_Results/logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M')
        # with tf.summary.create_file_writer(log_dir).as_default():
        #     hp.hparams_config(
        #         hparams= [HP_NUM_UNITS, HP_DROPOUT,  HP_OPTIMIZER, HP_LEARNING_RATE],
        #         metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
        #     )



##################### get rid of this? 
    # callback_object_list = [early_stopping, learning_rate_scheduler]

    # for i in callback_object_list: 
    #     if i == None:
    #         callback_object_list.pop(i) ##################### correct term "pop"? 
    #     else: 
    #         callbacks.append(i)

    # if early_stopping != False: 
    #     callbacks.append(early_stopping)
    
    # if learning_rate_scheduler != False: 
    #     callbacks.append(learning_rate_scheduler) #####################
    
    return callbacks, model_path


#################################################################

def create_LSTM():

    model = Sequential()

    model.add(LSTM(50, input_shape = (30,28),return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(50,return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(50,return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(50,return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(25, activation = "relu"))
    model.add(Dropout(0.2))

    model.add(Dense(5))

    model.compile(loss='mse', optimizer='rmsprop', metrics = ["accuracy"])

    return model



def cross_validate(X, y, model_results_path, K = 3, **kwargs):
    scores = []
    histories = []
    callbacks, model_path = callback_selector(early_stopping = early_stopping, monitor = monitor, patience = patience, min_delta = min_delta, learning_rate_scheduler_bool = learning_rate_scheduler_bool, custom_verbose= custom_verbose, metrics = metrics, model_and_weights_saved = model_and_weights_saved, tensorboard = tensorboard, reduce_lr_on_plateau = reduce_lr_on_plateau, model_results_path=model_results_path)
    for train, test in KFold(n_splits=K, shuffle=True).split(X,y):
        print(train)
        print(test)
        model = create_LSTM() # compile model
        start = time.time()
        histories.append(model.fit(X[train], y[train], epochs = 10, batch_size = 32, callbacks=callbacks,
                                   validation_data = (X[test],y[test]), # feed in the test data for plotting
                                   **kwargs).history)
        model.save(model_results_path)
        print(time.time() - start)
        scores.append(model.evaluate(X[test], y[test], verbose = 0)) # evaluate the test dataset
    print("average test loss: ", np.asarray(scores)[:,0].mean())
    print("average test accuracy: ", np.asarray(scores)[:,1].mean())
    print(model.summary())
    return scores, histories


def plot_histories(histories, metrics = ['loss', 'accuracy', 'val_accuracy','val_loss']):
    """
    function to plot the histories of data
    """
    fig, axes = plt.subplots(nrows = (len(metrics) - 1) // 2 + 1, ncols = 2, figsize = (16,16))
    axes = axes.reshape((len(metrics) - 1) // 2 + 1, 2)
    for i,metric in enumerate(metrics):
        for history in histories:
            axes[(i+2)//2 - 1, 1 - (i+1)%2].plot(history[metric])
            axes[(i+2)//2 - 1, 1 - (i+1)%2].legend([i for i in range(len(histories))])
            axes[(i+2)//2 - 1, 1 - (i+1)%2].set_xticks(np.arange(max(history[metric])))



