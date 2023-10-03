import os
import pandas
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# a function to plot the accuracy and the error during the training phase of the mode
def plot_history(history):

    fig, axs = plt.subplots(2)
    
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(50,100,640, 545)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="apprentissage")
    axs[0].plot(history.history["val_accuracy"], label="test")
    axs[0].set_ylabel("précision")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Évaluation de la précision")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="apprentissage")
    axs[1].plot(history.history["val_loss"], label="test")
    axs[1].set_ylabel("Erreur")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Évaluation de l\'erreur")
    
    
    fig.tight_layout()

    plt.show()

# a function to clean the data from one day and return both the dataframe cleaned 
# and a list containing all the dataframes of each ship
def process_day(file_path):
        
    fichier = file_path
    df = pandas.read_csv(fichier,sep=';')
    df.drop(columns=df.columns[-1],axis=1,inplace=True)
    
    
    liste_date=[]
    for i in df.index:
        liste_date.append(datetime.utcfromtimestamp(df['Horodatage Epoch :'][i]).strftime('%Y-%m-%d %H:%M:%S'))
    
    
    df.insert(5, "Date_UTC", liste_date, True)
    df_number=pandas.DataFrame(df.groupby(["MMSI:"])['Type de navire :'].count()).rename(columns={'Type de navire :': "nombre"})
    
    liste=[]
    for i in df_number.index:
        if df_number['nombre'][i]<10:
            liste.append(i)
    
    df_number=df_number.drop(liste)
    df_number=df_number.reset_index()
    
    
    # creation de la liste liste_df
    data_frame_day=pandas.DataFrame()
    liste_df=[]
    for i in df_number['MMSI:']:
        df_i=(df[df['MMSI:']==i]).reset_index().drop(["index"], axis=1)
        
        if not ( 511 in (df_i['Cap vrai () :'].tolist())) and not( False in (df_i['Vitesse (noeuds) :']<50).tolist()) and df_i['Type de navire :'][0] in np.arange(70,90):
            if len(df_i)>15 : 
                liste_df.append(df_i)
                data_frame_day=pandas.concat([data_frame_day,df_i],axis=0,ignore_index=True)
                # data_frame_day=data_frame_day.append(df_i,ignore_index=True)
    
    return liste_df,data_frame_day
    
# a function to know how many interventions were made in a specific day by the CROSS from the SITREP database
def process_sitrep(date_fichier):

    df_sitrep = pandas.read_excel("sitrep.xlsx")
    df_sitrep=df_sitrep[['MMSI','SENT_AT','REPORTING_DATE_AND_TIME','ID_MESSAGE']]

    df_sitrep = df_sitrep.loc[:, ~df_sitrep.columns.str.contains('^Unnamed')]
    df_sitrep = df_sitrep[df_sitrep.MMSI != 0]
    df_sitrep.drop_duplicates(subset ="MMSI",keep = False, inplace = True)
    
    c=0
    for j in df_sitrep.index:
        if df_sitrep['REPORTING_DATE_AND_TIME'][j] !='(null)':
            if str(np.datetime64(df_sitrep['REPORTING_DATE_AND_TIME'][j],'D')) == date_fichier :
                c+=1
    return c
