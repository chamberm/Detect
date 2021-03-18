from __future__ import division, print_function, absolute_import
import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense, BatchNormalization
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers, backend 
from numpy.random import seed
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import datetime
import seaborn as sns

def save(model):
    pass

def plot_loss(model_history):
    train_loss=[value for key, value in model_history.items() if 'loss' in key.lower()][0]
    valid_loss=[value for key, value in model_history.items() if 'loss' in key.lower()][1]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'xkcd:purply'
    ax1.set_xlabel('Epoch',size=42)
    ax1.set_ylabel('Loss', color="black",size=42)
    ax1.plot(train_loss, '--', color="black", label='Train Loss',linewidth=4)
    ax1.plot(valid_loss, color=color, label='Test Loss',linewidth=4)
    ax1.tick_params(axis='y', labelcolor="black")
    plt.legend(loc='upper right',fontsize=28)
    plt.title('Model Loss',size=48)
    ax1.tick_params(labelsize=32)
    fig.tight_layout()
    #fig.savefig('figures/AE_loss.png', dpi=200)
    st.write(fig)
    plt.close(fig)

def fit(autoencoder, X_train, epochs, size):
    ##################
    # HyperParameters
    ##################
    nb_epoch = epochs
    batch_size = size
    t_ini = datetime.datetime.now()
    history = autoencoder.fit(X_train, X_train,
                            epochs=nb_epoch,
                            batch_size=batch_size,
                            shuffle=True,
                            validation_split=0.1,
                            verbose=0
                            )

    t_fin = datetime.datetime.now()
    #print('Time to run the model: {} Sec.'.format((t_fin - t_ini).total_seconds()))

    df_history = pd.DataFrame(history.history)
    #plot_loss(df_history)
    
def create_model(model, lr, acts):
    backend.clear_session()
    X_train = model.get_train()
    X_test = model.get_test()
    
    input_dim = X_train.shape[1]
    encoding_dim =input_dim/2

    input_layer = Input(shape=(input_dim, ))
    encoder = Dense(int(encoding_dim), activation='relu')(input_layer)
    encoder = Dense(int(encoding_dim/2), activation='relu', activity_regularizer=regularizers.l2(10e-5))(encoder) 
    decoder = Dense(int(encoding_dim), activation='relu')(encoder)
    decoder = Dense(input_dim, activation=acts)(decoder)
    
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    #autoencoder.summary()
    customAdam = keras.optimizers.Adam(lr=lr)
    autoencoder.compile(optimizer=customAdam, loss='mse', metrics=["mean_squared_error", "mean_absolute_error"])
    
    return autoencoder

def run_once(model):

    X_train = model.get_train()
    X_test = model.get_test()
    seed(10)
    tf.random.set_seed(10)
  
    # define scope of configs
    lr = 0.001 
    n_epochs = 12 
    n_batch = 24 
    acts = 'tanh' 

    autoencoder = create_model(model, lr, acts)
    fit(autoencoder, X_train, n_epochs, n_batch)
        
    X_pred_train = autoencoder.predict(np.array(X_train))
    x_hat = autoencoder.predict(np.array(X_test))
    
    X_pred_train = pd.DataFrame(X_pred_train, 
                          columns=X_train.columns)
    X_pred_train.index = X_train.index
    
    x_hat = pd.DataFrame(x_hat, 
                          columns=X_test.columns)
    x_hat.index = X_test.index
    
    #mae = np.mean(np.abs(x_hat-X_test), axis = 1)
    
    return x_hat #,mae

def run(model):
    #seed(10)
    tf.random.set_seed(10)
    X_train = model.get_train()
    X_test = model.get_test()
    
    # define scope of configs
    lr = [0.001] 
    n_epochs = [25]
    n_batch = [24] 
    acts = ['tanh'] 
    # create configs
    configs = list()
    for i in lr:
        for j in acts:
            for k in n_epochs:
                for l in n_batch:
                    st.write("Learning rate: ", i, "Activation: ", j, "Epochs: ", k, "batch: ", l)
                    autoencoder = create_model(model, i, j)
                    fit(autoencoder, X_train, k, l)
        
    X_pred_train = autoencoder.predict(np.array(X_train))
    X_pred_test = autoencoder.predict(np.array(X_test))
    
    X_pred_train = pd.DataFrame(X_pred_train, 
                          columns=X_train.columns)
    X_pred_train.index = X_train.index

    scoredTrain = pd.DataFrame(index=X_train.index)
    scoredTrain['error'] = np.mean(np.abs(X_pred_train-X_train), axis = 1)
    
    X_pred_test = pd.DataFrame(X_pred_test, 
                          columns=X_test.columns)
    X_pred_test.index = X_test.index

    scoredTest = pd.DataFrame(index=X_test.index)
    scoredTest['error'] = np.mean(np.abs(X_pred_test-X_test), axis = 1)
    MAE_test = scoredTest['error']
    MAE_train = scoredTrain['error']
    
    return MAE_train, MAE_test
    