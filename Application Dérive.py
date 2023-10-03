# importing the required modules
import tkinter.filedialog as fd
from tkinter import *
import Annotation
import model_ANN
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import model_RNN
import tensorflow as tf
import util
from tkinter import ttk
import time



def annotation_derive():
    win = Toplevel()

    # set the position of the window
    win.geometry("+%d+%d" % (x-380 , y-150))

    # set the title
    win.title("Annotation de dérive")

    # Set the geometry of tkinter frame
    win.geometry("360x100")

    # Set the logo 
    win.iconbitmap("ship.ico")

    # Add a Label widget
    Label(win, text="Choisir un ou plusieurs fichiers : ", font=('Georgia 10')).pack(pady=10)

    def open_files(win):
        filez = fd.askopenfilenames(parent=win, title='Choisir un fichier')
        if filez : 
            liste=list(filez)
            Annotation.run(liste)        

    # Create a Button
    Button(win, text="Importer", command = lambda : open_files(win)).pack()
    
def train_and_save():
    global second_frame
    win = Toplevel()
    
    # set the position of the window
    win.geometry("+%d+%d" % (x-380 , y-200))

    # set the title
    win.title("Entraîner et sauvegarder un modèle")

    # Set the geometry of tkinter frame
    win.geometry("360x700")

    # Set the logo 
    win.iconbitmap("ship.ico")
    
    # ------------- adding a scrollbar ----------------------------------------
    height=700
    width_sb=360
    
    main_frame = Frame(win,width=width_sb,height=height)
    main_frame.place(x=0,y=0)
    # Create A Canvas
    my_canvas = Canvas(main_frame, width=width_sb, height=height)
    my_canvas.place(x=0,y=0)
    # Add A Scrollbar To The Canvas
    my_scrollbar = Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview)
    my_scrollbar.place(x=340,y=0,height=height)
    # Configure The Canvas
    my_canvas.configure(yscrollcommand=my_scrollbar.set)
    my_canvas.bind('<Configure>', lambda e: my_canvas.configure(scrollregion = my_canvas.bbox("all")))
    def _on_mouse_wheel(event):
        my_canvas.yview_scroll(-1 * int((event.delta / 120)), "units")
    my_canvas.bind_all("<MouseWheel>", _on_mouse_wheel)
    # Create ANOTHER Frame INSIDE the Canvas
    second_frame = Frame(my_canvas,width=width_sb,height=height)
    second_frame.place(x=0,y=0)
    # Add that New frame To a Window In The Canvas
    my_canvas.create_window((0,0), window=second_frame, anchor="nw")
    second_frame.configure(height=2000)
    # -------------------------------------------------------------------------
    
    # Add a Label widget
    Label(second_frame, text="Choisir le modèle à utiliser : ", font=('Georgia 10')).place(x=70,y=10)
    
    # Add buttons to choose the model
    b1 = Button(second_frame, text='ANN',command = lambda : train_ANN(win)).place(x=50,y=50)
    b2 = Button(second_frame, text='RNN',command = lambda : train_RNN_LSTM(win,tf.keras.layers.SimpleRNN)).place(x=150, y = 50)
    b3 = Button(second_frame, text='LSTM',command = lambda : train_RNN_LSTM(win,tf.keras.layers.LSTM)).place(x=250, y = 50)

def save_model(model):
    dir = fd.askdirectory()
    model.save(str(dir))

def clear_frame(frame):
   for widgets in frame.winfo_children():
      widgets.destroy()

def train_ANN(win):
    
    # clear frame in case of selecting another time the button model
    clear_frame(second_frame)  
    Label(second_frame, text="Choisir le modèle à utiliser : ", font=('Georgia 10')).place(x=70,y=10)
    b1 = Button(second_frame, text='ANN',command = lambda : train_ANN(win)).place(x=50,y=50)
    b2 = Button(second_frame, text='RNN',command = lambda : train_RNN_LSTM(win,tf.keras.layers.SimpleRNN)).place(x=150, y = 50)
    b3 = Button(second_frame, text='LSTM',command = lambda : train_RNN_LSTM(win,tf.keras.layers.LSTM)).place(x=250, y = 50)

    # input field for the learning rate
    Label(second_frame, text="Choisir le pas d\'apprentissage : ", font=('Georgia 10')).place(x=10,y=100)
    pas = Entry(second_frame,width=10)
    pas.place(x=10,y=125)
    
    # input field for the number of epochs
    Label(second_frame, text="Choisir le nombre d\'epochs : ", font=('Georgia 10')).place(x=10,y=150)
    epochs = Entry(second_frame,width=10)
    epochs.place(x=10,y=175)
    
    # input field for the batch size
    Label(second_frame, text="Choisir le batch size : ", font=('Georgia 10')).place(x=10,y=200)
    batch = Entry(second_frame,width=10)
    batch.place(x=10,y=225)
    
    # input field for the number of layers
    Label(second_frame, text="Choisir le nombre de couches : ", font=('Georgia 10')).place(x=10,y=250)
    e = Entry(second_frame,width=10)
    e.place(x=10,y=275)
    
    # a function to choose the number of neurons for each layer
    def nbNeurons():
        if int(e.get())>1 :
            Label(second_frame, text="Choisir le nombre de neurones par couche : ", font=('Georgia 10')).place(x=10,y=310)
            d={}
            for i in range(int(e.get())-1):
                Label(second_frame, text="couhe n:° : "+str(i), font=('Georgia 10')).place(x=10,y=335+i*25)
                d['e{0}'.format(i)]=Entry(second_frame,width=10)
                d['e{0}'.format(i)].place(x=150,y=335+i*25)
                
            
            # a function to train the model and generate the save button in the case of multiple layers
            def train():
                global hyper_param
                hyper_param=[]
                hyper_param.append(float(pas.get()))
                hyper_param.append(int(epochs.get()))
                hyper_param.append(int(batch.get()))
                hyper_param.append(int(e.get()))
                liste_neurons = []
                for j in range(int(e.get())-1):
                    liste_neurons.append(int(d['e{0}'.format(j)].get()))
                hyper_param.append(liste_neurons)
                model,X_test,y_test = model_ANN.train_ANN(data_set,hyper_param)
                Label(second_frame, text="Matrice de confusion : choisir le seuil : ", font=('Georgia 10')).place(x=10,y=465+(i+1)*25)
                seuil = Entry(second_frame,width=10)
                seuil.place(x=10,y=490+(i+1)*25)
                
                def get_seuil():
                    prediction = model.predict(X_test)
                    prediction[prediction<=float(seuil.get())]=0
                    prediction[prediction>float(seuil.get())]=1

                    fig1 = plt.figure()
                    cm = confusion_matrix(y_test,prediction,labels=[0,1])
                    sns.heatmap(cm,cmap='Greens',annot=True,
                                xticklabels=[0,1],yticklabels=[0,1],fmt='g')
                    plt.xlabel('Prediction')
                    plt.ylabel('réel')
                    plt.show()
                    
                    Label(second_frame, text="Sauvegarder le modèle : ", font=('Georgia 10')).place(x=85,y=525+(i+1)*25)
                    btn_train = Button(second_frame,text='Sauvegarder',command = lambda : save_model(model)).place(x=142,y=550+(i+1)*25)
                
                btn_seuil=Button(second_frame,text='Valider', command=get_seuil).place(x=80,y=490+(i+1)*25)
                
            
            def open_files(win):
                
                global data_set
                files = fd.askopenfilenames(parent=win, title='Choisir les données d\'apprenstissage')
                if files : 
                    data_set=list(files)
                
                Label(second_frame, text="Entraîner le modèle : ", font=('Georgia 10')).place(x=94,y=405+(i+1)*25)    
                btn_train = Button(second_frame,text='Entraîner',command=train).place(x=150,y=430+(i+1)*25) 
            
            Label(second_frame, text="Importer les données d\'apprentissage : ", font=('Georgia 10')).place(x=30,y=340+(i+1)*25)    
            btn_train = Button(second_frame,text='Importer',command=lambda : open_files(second_frame)).place(x=150,y=370+(i+1)*25) 
            
      
        else :
            # a function to train the model and generate the save button in the case of a single layer
            def train():
                global hyper_param
                hyper_param=[]
                hyper_param.append(float(pas.get()))
                hyper_param.append(int(epochs.get()))
                hyper_param.append(int(batch.get()))
                hyper_param.append(int(e.get()))
                liste_neurons = []
                hyper_param.append(liste_neurons)
                model,X_test,y_test = model_ANN.train_ANN(data_set,hyper_param)
                
                Label(second_frame, text="Matrice de confusion : choisir le seuil : ", font=('Georgia 10')).place(x=10,y=430)
                seuil = Entry(second_frame,width=10)
                seuil.place(x=10,y=455)
                
                def get_seuil():
                    prediction = model.predict(X_test)
                    prediction[prediction<=float(seuil.get())]=0
                    prediction[prediction>float(seuil.get())]=1

                    fig1 = plt.figure()
                    cm = confusion_matrix(y_test,prediction,labels=[0,1])
                    sns.heatmap(cm,cmap='Greens',annot=True,
                                xticklabels=[0,1],yticklabels=[0,1],fmt='g')
                    plt.xlabel('Prediction')
                    plt.ylabel('réel')
                    plt.show()
                    
                    Label(second_frame, text="Sauvegarder le modèle : ", font=('Georgia 10')).place(x=85,y=490)
                    btn_train = Button(second_frame,text='Sauvegarder',command = lambda : save_model(model)).place(x=142,y=520)
                
                btn_seuil=Button(second_frame,text='Valider', command=get_seuil).place(x=80,y=455)

            def open_files(win):
                
                global data_set
                files = fd.askopenfilenames(parent=win, title='Choisir les données d\'apprenstissage')
                if files : 
                    data_set=list(files)
                
                Label(second_frame, text="Entraîner le modèle : ", font=('Georgia 10')).place(x=97,y=370)    
                btn_train = Button(second_frame,text='Entraîner',command=train).place(x=150,y=395) 
            
            Label(second_frame, text="Importer les données d\'apprentissage : ", font=('Georgia 10')).place(x=30,y=310)    
            btn_train = Button(second_frame,text='Importer',command=lambda : open_files(second_frame)).place(x=150,y=335) 
            
            
    btn=Button(second_frame,text='Valider', command=nbNeurons).place(x=80,y=275)
    
    
def train_RNN_LSTM(win,model_type):
    
    # clear frame in case of selecting another time the button model
    clear_frame(second_frame)  
    Label(second_frame, text="Choisir le modèle à utiliser : ", font=('Georgia 10')).place(x=70,y=10)
    b1 = Button(second_frame, text='ANN',command = lambda : train_ANN(win)).place(x=50,y=50)
    b2 = Button(second_frame, text='RNN',command = lambda : train_RNN_LSTM(win,tf.keras.layers.SimpleRNN)).place(x=150, y = 50)
    b3 = Button(second_frame, text='LSTM',command = lambda : train_RNN_LSTM(win,tf.keras.layers.LSTM)).place(x=250, y = 50)
    
    # input field for the learning rate
    Label(second_frame, text="Choisir le pas d\'apprentissage : ", font=('Georgia 10')).place(x=10,y=100)
    pas = Entry(second_frame,width=10)
    pas.place(x=10,y=125)
    
    # input field for the number of epochs
    Label(second_frame, text="Choisir le nombre d\'epochs : ", font=('Georgia 10')).place(x=10,y=150)
    epochs = Entry(second_frame,width=10)
    epochs.place(x=10,y=175)
    
    # input field for the batch size
    Label(second_frame, text="Choisir le batch size : ", font=('Georgia 10')).place(x=10,y=200)
    batch = Entry(second_frame,width=10)
    batch.place(x=10,y=225)
    
    # input field for the sequence length
    Label(second_frame, text="Choisir la longueur de la séquence : ", font=('Georgia 10')).place(x=10,y=250)
    e = Entry(second_frame,width=10)
    e.place(x=10,y=275)
    
    # input field for the neuron number of the rnn layer
    if model_type == tf.keras.layers.SimpleRNN :
        Label(second_frame, text="Nombre de neurones de la couche RNN : ", font=('Georgia 10')).place(x=10,y=300)
        nb_rnn = Entry(second_frame,width=10)
        nb_rnn.place(x=10,y=325)
        
    if model_type == tf.keras.layers.LSTM :
        Label(second_frame, text="Nombre de neurones de la couche LSTM : ", font=('Georgia 10')).place(x=10,y=300)
        nb_rnn = Entry(second_frame,width=10)
        nb_rnn.place(x=10,y=325)
    
    # input field for the neuron number of the dense layer
    Label(second_frame, text="Nombre de neurones de la couche Dense : ", font=('Georgia 10')).place(x=10,y=350)
    nb_dense = Entry(second_frame,width=10)
    nb_dense.place(x=10,y=375)
    
    def btn_rnn():
        # a function to train the model and generate the save button in the case of a single layer
        def train():
            global hyper_param
            hyper_param=[]
            hyper_param.append(float(pas.get()))
            hyper_param.append(int(epochs.get()))
            hyper_param.append(int(batch.get()))
            hyper_param.append(int(e.get()))
            hyper_param.append(int(nb_rnn.get()))
            hyper_param.append(int(nb_dense.get()))
            hyper_param.append(model_type)
            
            
            model,X_test,y_test = model_RNN.train_RNN(data_set,hyper_param)
            y_test=y_test.flatten()
            
            Label(second_frame, text="Matrice de confusion : choisir le seuil : ", font=('Georgia 10')).place(x=10,y=530)
            seuil = Entry(second_frame,width=10)
            seuil.place(x=10,y=555)
            
            def get_seuil():
                prediction = model.predict(X_test)
                prediction[prediction<=float(seuil.get())]=0
                prediction[prediction>float(seuil.get())]=1
                prediction= prediction.flatten()
                
    
                fig1 = plt.figure()
                cm = confusion_matrix(y_test,prediction,labels=[0,1])
                sns.heatmap(cm,cmap='Greens',annot=True,
                            xticklabels=[0,1],yticklabels=[0,1],fmt='g')
                plt.xlabel('Prediction')
                plt.ylabel('réel')
                plt.show()
                
                Label(second_frame, text="Sauvegarder le modèle : ", font=('Georgia 10')).place(x=85,y=595)
                btn_train = Button(second_frame,text='Sauvegarder',command = lambda : save_model(model)).place(x=142,y=620)
            
            btn_seuil=Button(second_frame,text='Valider', command=get_seuil).place(x=80,y=555)
    
        def open_files(win):
            
            global data_set
            files = fd.askopenfilenames(parent=win, title='Choisir les données d\'apprenstissage')
            if files : 
                data_set=list(files)
            
            Label(second_frame, text="Entraîner le modèle : ", font=('Georgia 10')).place(x=97,y=470)    
            btn_train = Button(second_frame,text='Entraîner',command=train).place(x=150,y=495) 
        
        Label(second_frame, text="Importer les données d\'apprentissage : ", font=('Georgia 10')).place(x=30,y=410)    
        btn_train = Button(second_frame,text='Importer',command=lambda : open_files(second_frame)).place(x=150,y=435) 

    
    btn=Button(second_frame,text='Valider',command = btn_rnn ).place(x=80,y=375)
    


def load_and_test():
    
    # global t1
    def load_model(t):
        global model,t1
        # clear frame in case of selecting another time the model button
        clear_frame(win)
        Label(win, text="Importer le modèle à utiliser : ", font=('Georgia 10')).place(x=70,y=10)
        b1 = Button(win, text='ANN',command = lambda t =1 : load_model(t)).place(x=50,y=50)
        b2 = Button(win, text='RNN',command = lambda t =2 : load_model(t)).place(x=150, y = 50)
        b3 = Button(win, text='LSTM',command = lambda t =3 : load_model(t)).place(x=250, y = 50)
        Button(win, text='Tester sur des navires',command = lambda : tester_navire(win)).place(x=15, y = 110)
        Button(win, text='Tester sur une journée',command = tester_journee).place(x=180, y = 110)
        t1=t
        dir = fd.askdirectory()
        model = tf.keras.models.load_model(str(dir))


    win = Toplevel()
    
    # set the position of the window
    win.geometry("+%d+%d" % (x-380 , y-150))

    # set the title
    win.title("Importer un modèle et le tester")

    # Set the geometry of tkinter frame
    win.geometry("360x340")

    # Set the logo 
    win.iconbitmap("ship.ico")
    # Add a Label widget
    Label(win, text="Importer le modèle à utiliser : ", font=('Georgia 10')).place(x=70,y=10)
    
    # Add buttons to choose the model

    b1 = Button(win, text='ANN',command = lambda t =1 : load_model(t)).place(x=50,y=50)
    b2 = Button(win, text='RNN',command = lambda t =2 : load_model(t)).place(x=150, y = 50)
    b3 = Button(win, text='LSTM',command = lambda t =3 : load_model(t)).place(x=250, y = 50)
    
    
    
    def tester_navire(win):
        global test_set
        
        # clear frame in case of selecting another time the button of the test
        clear_frame(win)
        Label(win, text="Importer le modèle à utiliser : ", font=('Georgia 10')).place(x=70,y=10)
        b1 = Button(win, text='ANN',command = lambda t =1 : load_model(t)).place(x=50,y=50)
        b2 = Button(win, text='RNN',command = lambda t =2 : load_model(t)).place(x=150, y = 50)
        b3 = Button(win, text='LSTM',command = lambda t =3 : load_model(t)).place(x=250, y = 50)
        Button(win, text='Tester sur des navires',command = lambda : tester_navire(win)).place(x=15, y = 110)
        Button(win, text='Tester sur une journée',command = tester_journee).place(x=180, y = 110)
        
        files = fd.askopenfilenames(parent=win, title='Choisir les données de test')
        if files : 
            test_set=list(files)
        
            Label(win, text="Matrice de confusion : choisir le seuil : ", font=('Georgia 10')).place(x=10,y=155)
            seuil = Entry(win,width=10)
            seuil.place(x=10,y=180)
            
            # depending on the value of 't1' eg which button did we press, we'll be choosing a certain model 
            def valider_test(test_set,model,seuil,t1):
                if t1 == 1 :
                    model_ANN.test_ANN(test_set,model,seuil)
                if t1 == 2 :
                    model_RNN.test_RNN_LSTM(test_set,model,seuil)
                if t1 == 3 :
                    model_RNN.test_RNN_LSTM(test_set,model,seuil)           
            
            btn_val = Button(win, text='valider',command = lambda : valider_test(test_set,model,float(seuil.get()),t1)).place(x=80, y = 180)

            
    def tester_journee():
        global liste_df,filename,data_frame_day
        
        # clear frame in case of selecting another time the button of the test
        clear_frame(win)
        Label(win, text="Importer le modèle à utiliser : ", font=('Georgia 10')).place(x=70,y=10)
        b1 = Button(win, text='ANN',command = lambda t =1 : load_model(t)).place(x=50,y=50)
        b2 = Button(win, text='RNN',command = lambda t =2 : load_model(t)).place(x=150, y = 50)
        b3 = Button(win, text='LSTM',command = lambda t =3 : load_model(t)).place(x=250, y = 50)
        Button(win, text='Tester sur des navires',command = lambda : tester_navire(win)).place(x=15, y = 110)
        Button(win, text='Tester sur une journée',command = tester_journee).place(x=180, y = 110)
        
        # import the day file
        filename = str(filedialog.askopenfilename())
        liste_df,data_frame_day = util.process_day(filename)

        Label(win, text="Matrice de confusion : choisir le seuil : ", font=('Georgia 10')).place(x=10,y=155)
        seuil = Entry(win,width=10)
        seuil.place(x=10,y=180)
        
        # a funtion to plot the histogram
        def tracer_hist(liste_count_1):
            x = [i for i in range(len(liste_count_1))]
            y = liste_count_1
            
            mngr = plt.get_current_fig_manager()
            mngr.window.setGeometry(1100,100,640, 545)
            plt.bar(x,y,align='center') 
            plt.xlabel('Navire')
            plt.ylabel('nbre de 1')
            plt.title('Histogramme représentant la distribution des msgs prédits à 1')
            for i in range(len(y)):
                plt.vlines(x[i],0,y[i]) 
            plt.show()
        
        def valider_test_journee(liste,model,seuil,t1):
            global prediction,prediction_fw
            # test the ANN model on a specific day
            if t1 == 1 :
                prediction,liste_count_1 = model_ANN.test_ANN_journee(liste,model,seuil)
                data_frame_day.insert(len(data_frame_day.columns), "Prédiction", prediction, True)
                data_frame_day.to_excel(str(filename.split('.')[0])+'-prédiction.xlsx',index=False) 
                c = util.process_sitrep(filename.split('/')[-1].split('.')[0])
                if c !=0 :
                    Label(win, text='nombre d\'interventions : '+ str(c), font=('Georgia 10')).place(x=10,y=215)
                else :
                    Label(win, text='nombre d\'interventions non connu ou 0', font=('Georgia 10')).place(x=10,y=215)
                Label(win, text='nombre de navires : '+ str(len(liste_df)), font=('Georgia 10')).place(x=10,y=240)
                Label(win, text="Prédiction à \'0\' : "+ str(prediction.count(0)), font=('Georgia 10')).place(x=10,y=265)
                Label(win, text="Prédiction à \'1\' : "+ str(prediction.count(1)), font=('Georgia 10')).place(x=10,y=290)
                tracer_hist(liste_count_1)
            
            # test the RNN model on a specific day    
            if t1 == 2 :
                prediction,liste_count_1,prediction_fw = model_RNN.test_RNN_LSTM_journee(liste,model,seuil)
                data_frame_day.insert(len(data_frame_day.columns), "Prédiction",prediction_fw, True)
                data_frame_day.to_excel(str(filename.split('.')[0])+'-prédiction.xlsx',index=False) 
                c = util.process_sitrep(filename.split('/')[-1].split('.')[0])
                if c !=0 :
                    Label(win, text='nombre d\'interventions : '+ str(c), font=('Georgia 10')).place(x=10,y=215)
                else :
                    Label(win, text='nombre d\'interventions non connu ou 0', font=('Georgia 10')).place(x=10,y=215)
                Label(win, text='nombre de navires : '+ str(len(liste_df)), font=('Georgia 10')).place(x=10,y=240)
                Label(win, text="Prédiction à \'0\' : "+ str(prediction.count(0)), font=('Georgia 10')).place(x=10,y=265)
                Label(win, text="Prédiction à \'1\' : "+ str(prediction.count(1)), font=('Georgia 10')).place(x=10,y=290)
                tracer_hist(liste_count_1)
            
            # test the LSTM model on a specific day    
            if t1 == 3 :
                prediction,liste_count_1,prediction_fw = model_RNN.test_RNN_LSTM_journee(liste,model,seuil) 
                data_frame_day.insert(len(data_frame_day.columns), "Prédiction",prediction_fw, True)
                data_frame_day.to_excel(str(filename.split('.')[0])+'-prédiction.xlsx',index=False) 
                c = util.process_sitrep(filename.split('/')[-1].split('.')[0])
                if c !=0 :
                    Label(win, text='nombre d\'interventions : '+ str(c), font=('Georgia 10')).place(x=10,y=215)
                else :
                    Label(win, text='nombre d\'interventions non connu ou 0', font=('Georgia 10')).place(x=10,y=215)
                Label(win, text='nombre de navires : '+ str(len(liste_df)), font=('Georgia 10')).place(x=10,y=240)
                Label(win, text="Prédiction à \'0\' : "+ str(prediction.count(0)), font=('Georgia 10')).place(x=10,y=265)
                Label(win, text="Prédiction à \'1\' : "+ str(prediction.count(1)), font=('Georgia 10')).place(x=10,y=290)
                tracer_hist(liste_count_1)
                
        btn_val = Button(win, text='valider',command = lambda : valider_test_journee(liste_df,model,float(seuil.get()),t1)).place(x=80, y = 180)



    b3 = Button(win, text='Tester sur des navires',command = lambda : tester_navire(win)).place(x=15, y = 110)
    b3 = Button(win, text='Tester sur une journée',command = tester_journee).place(x=180, y = 110)
    
    

if __name__ == '__main__':
    
    root = Tk()
    
    # set the title
    root.title('Application Dérive')
    
    # Set the geometry of tkinter frame
    root.geometry('360x260')
    
    # make window appear in the middle of the screen
    root.eval('tk::PlaceWindow . center')
    
    x = root.winfo_x()
    y = root.winfo_y()
    
    # Set the logo 
    root.iconbitmap("ship.ico")
    
    # create buttons for the main window
    btn1 = Button(root,text='Annotation Dérive',command=annotation_derive,width=33, height=2).pack(pady= 15, padx= 20)
    btn2 = Button(root, text ='Entraîner et sauvegarder un modèle',command=train_and_save,width=33, height=2).pack(pady= 15, padx= 20)
    btn3 = Button(root, text = 'Importer un modèle et le tester',command = load_and_test,width=33, height=2).pack(pady= 15, padx= 20)
    
    root.mainloop()