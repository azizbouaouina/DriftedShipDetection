# importing the required modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization,SimpleRNN
import seaborn as sns
from sklearn.metrics import confusion_matrix
from util import plot_history


 
def load_data(file_list,SEQ_LEN):
         

    data_frame_list = []
    
    sequential_data = []  # this is a list that will CONTAIN the sequences
     
    for file in file_list:
        data_frame_list.append(pd.read_excel(file))

     
    for df in data_frame_list:

        # calculation of the difference between true heading and course over ground    
        df.loc[df["Cap () :"] > 180 ,"Cap () :" ] = 360 - df["Cap () :"]
        df.loc[df["Cap vrai () :"] > 180 ,"Cap vrai () :" ] = 360 - df["Cap vrai () :"]
        df['diff']=abs(df['Cap () :']-df['Cap vrai () :'])

        df=df[['Vitesse (noeuds) :','Statut de navigation :','diff','avis']]

        prev_msgs = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. 
        
        for i in df.values:  # iterate over the values
            prev_msgs.append([n for n in i[:-1]])  # store all but the target
            if len(prev_msgs) == SEQ_LEN:  # make sure we have seq_len sequences!
                sequential_data.append([np.array(prev_msgs), i[-1]])
     
    X = []
    y = []
        
    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets
        
    return np.array(X), np.array(y)


def load_data_day(file_list,SEQ_LEN):
         

    data_frame_list = []
    
    sequential_data = []  # this is a list that will CONTAIN the sequences
    
    liste_x=[]
     
    for file in file_list:
        data_frame_list.append((file))


    for df in data_frame_list:

        # calculation of the difference between true heading and course over ground    
        df.loc[df["Cap () :"] > 180 ,"Cap () :" ] = 360 - df["Cap () :"]
        df.loc[df["Cap vrai () :"] > 180 ,"Cap vrai () :" ] = 360 - df["Cap vrai () :"]
        df['diff']=abs(df['Cap () :']-df['Cap vrai () :'])

        df=df[['Vitesse (noeuds) :','Statut de navigation :','diff']]

        prev_msgs = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. 
        sequential_data = []
        for i in df.values:  # iterate over the values
            prev_msgs.append([n for n in i[:]])  # store all but the target
            if len(prev_msgs) == SEQ_LEN:  # make sure we have seq_len sequences!
                sequential_data.append(np.array(prev_msgs))
        liste_x.append(np.array(sequential_data))


    return liste_x


def train_RNN(file_liste,hyper_param):
    
    lr = hyper_param[0]
    EPOCHS = hyper_param[1]
    BATCH_SIZE = hyper_param[2]
    SEQ_LEN = hyper_param[3]
    nb_rnn = hyper_param[4]
    nb_dense = hyper_param[5]
    model_type = hyper_param[6]
    
    X,y = load_data(file_liste,SEQ_LEN)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y)
    
    
    # the build of the RNN/LSTM
    model = Sequential()
    model.add(model_type(nb_rnn,activation = 'tanh',use_bias=True,input_shape=(X.shape[1:])))
    
              
    model.add(Dense(nb_dense, activation='tanh'))
    
    model.add(Dense(1, activation='sigmoid'))
    
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    
    # Compile model
    model.compile(loss='BinaryCrossentropy',optimizer=opt,metrics=[tf.keras.metrics.BinaryAccuracy(
        name="accuracy", dtype=None, threshold=0.5
    )]
    )
    
    
    # Train model
    history = model.fit(X_train, y_train,validation_data=(X_test, y_test),batch_size=BATCH_SIZE,epochs=EPOCHS)
    
    plot_history(history)
    
    return model,X_test,y_test

# test the RNN-LSTM model on a signle or multiple ships
def test_RNN_LSTM(file_list,model,seuil):
    
    X_test,y_test = load_data(file_list,model.input.shape[1])
    prediction = model.predict(X_test)
    prediction[prediction<=seuil]=0
    prediction[prediction>seuil]=1
    prediction= prediction.flatten()
    y_test= y_test.flatten()

    fig1 = plt.figure()
    cm = confusion_matrix(y_test,prediction,labels=[0,1])
    sns.heatmap(cm,cmap='Greens',annot=True,
                xticklabels=[0,1],yticklabels=[0,1],fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('réel')
    plt.show()
    

# test the RNN-LSTM model on a day of ais data
def test_RNN_LSTM_journee(file_list,model,seuil):
    
    x_test_liste = load_data_day(file_list,model.input.shape[1])
    liste_count_1=[]
    liste_prediction=[]
    liste_prediction_avec_x=[]# to save the prediction in the day file without losing any data due to sequence length
    for X_test in x_test_liste :
            
        prediction = model.predict(X_test)
        prediction[prediction<=seuil]=0
        prediction[prediction>seuil]=1
        prediction= prediction.flatten()
        prediction = prediction.tolist()
        prediction = [int(i) for i in prediction]
        
        for j in range(model.input.shape[1]-1):
            liste_prediction_avec_x.append('x')
        
        for i in prediction :
            liste_prediction.append(i)
            liste_prediction_avec_x.append(i)
        liste_count_1.append(prediction.count(1))

    return liste_prediction,liste_count_1,liste_prediction_avec_x
    
    
    