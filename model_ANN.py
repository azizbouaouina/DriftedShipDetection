# importing the required modules
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import sklearn
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import models
from keras.layers import Dense
import seaborn as sns
from sklearn.metrics import confusion_matrix
from util import plot_history

  
#load data for the different ships 
def load_data(file_list):
    
    excl_list = []
    for file in file_list:
        navire=pd.read_excel(file)
        excl_list.append(navire)
     
    # create a new dataframe to store the merged excel file.
    df = pd.DataFrame()
     
    for excl_file in excl_list:
        df = df.append(excl_file, ignore_index=True)

    # calculation of the difference between true heading and course over ground    
    df.loc[df["Cap () :"] > 180 ,"Cap () :" ] = 360 - df["Cap () :"]
    df.loc[df["Cap vrai () :"] > 180 ,"Cap vrai () :" ] = 360 - df["Cap vrai () :"]
    df['diff']=abs(df['Cap () :']-df['Cap vrai () :'])

    df=df[['Vitesse (noeuds) :','Statut de navigation :','diff','avis']]

    # convert lists to numpy arrays
    X = df[['Statut de navigation :','Vitesse (noeuds) :','diff']].to_numpy()
    y = df[['avis']].to_numpy()

    return  X, y

#load data for the whole day
def load_data_day(file_list):
    
    liste_x=[]
    for df in file_list:
        # calculation of the difference between true heading and course over ground    
        df.loc[df["Cap () :"] > 180 ,"Cap () :" ] = 360 - df["Cap () :"]
        df.loc[df["Cap vrai () :"] > 180 ,"Cap vrai () :" ] = 360 - df["Cap vrai () :"]
        df['diff']=abs(df['Cap () :']-df['Cap vrai () :'])
    
        df=df[['Vitesse (noeuds) :','Statut de navigation :','diff']]
    
        # convert lists to numpy arrays
        X = df[['Statut de navigation :','Vitesse (noeuds) :','diff']].to_numpy()
        liste_x.append(X)

    return  liste_x

def train_ANN(file_liste,hyper_param):
    
    lr = hyper_param[0]
    epochs = hyper_param[1]
    batch = hyper_param[2]
    layers = hyper_param[3]
    nb_neurons = hyper_param[4]

    # load data
    X, y = load_data(file_liste)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    
    # build network topology
    if layers == 1:
        model = models.Sequential()
        model.add(Dense(1, activation='sigmoid', input_shape=(3,)))
    
    else : 
        model = models.Sequential()
        model.add(Dense(nb_neurons[0], activation='tanh', input_shape=(3,)))
        
        for i in range(len(nb_neurons)-1):
            model.add(Dense(nb_neurons[i+1], activation='tanh'))
        
        model.add(Dense(1, activation='sigmoid'))
  
    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimiser,loss='BinaryCrossentropy',metrics=[tf.keras.metrics.BinaryAccuracy(
        name="accuracy", dtype=None, threshold=0.5
    )])    
    
    model.summary()

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),batch_size=batch,epochs=epochs)
    
    plot_history(history)
    
    return model,X_test,y_test

# test the ANN model on a signle or multiple ships
def test_ANN(file_list,model,seuil):
    
    X_test,y_test = load_data(file_list)
    prediction = model.predict(X_test)
    prediction[prediction<=seuil]=0
    prediction[prediction>seuil]=1
    fig1 = plt.figure()
    cm = confusion_matrix(y_test,prediction,labels=[0,1])
    sns.heatmap(cm,cmap='Greens',annot=True,
                xticklabels=[0,1],yticklabels=[0,1],fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('réel')
    plt.show()

# test the ANN model on a day of ais data
def test_ANN_journee(file_list,model,seuil):
    
    X_test_liste = load_data_day(file_list)
    liste_count_1=[]
    liste_prediction=[]
    for X_test in X_test_liste :
            
        prediction = model.predict(X_test)
        prediction[prediction<=seuil]=0
        prediction[prediction>seuil]=1
        liste =[]
        for i in prediction :
            liste.append(i[0])
            liste_prediction.append(i[0])
        liste_count_1.append(liste.count(1))
        
    return liste_prediction,liste_count_1
    

    
    
