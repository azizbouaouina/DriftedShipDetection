import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button,MultiCursor
import pandas
from datetime import datetime
from matplotlib import dates
import os

def liste_derive(y):
    i=0
    b=0
    liste=[]
    for i in range(len(y)):
        if y[i]==1 and b==0:
            a=i
            b=1
            #liste.append(a)
        if y[i]==0 and b==1:
            c=i-1
            b=0
            liste.append([a,c])
        if y[i]==1 and i==len(y)-1 and b==1 :
            c=i
            liste.append([a,c])
    return liste



class Index:
    ind = 0
    

    def next(self, event):
        global cursor_2,ax2,ax3,ax6,ax7,liste_plot,liste_pts,liste_lines
        liste_pts=[]
        self.ind += 1
        print(self.ind)
        if self.ind < len(liste_path1): 
            i = self.ind % len(liste_path1)
            df = pandas.read_excel(liste_path1[i])
            file_path=liste_path1[i]
            
            df_copy = df.copy()
            
            df.loc[df["Cap () :"] > 180 ,"Cap () :" ] = 360 - df["Cap () :"]
            df.loc[df["Cap vrai () :"] > 180 ,"Cap vrai () :" ] = 360 - df["Cap vrai () :"]
            df['diff']=abs(df['Cap vrai () :']-df['Cap () :'])
                
                
            # sauvegarder les valeurs à changer    
            columns = df.columns
            m=0
            for i in columns:
                if i == 'avis':
                    m=1
            
            if m == 0 :
                liste = [0 for i in df.index]
                df_copy.insert(len(df_copy.columns),'avis',liste,True)
                df_copy.to_excel(file_path,index=False) 
            
            
                
    
            y = df['Vitesse (noeuds) :']
            y2= df['Statut de navigation :']
            y1= df["diff"]

    
            liste_heure=[]
            for k in df.index:
                liste_heure.append(datetime.utcfromtimestamp(df['Horodatage Epoch :'][k]))
                
    
            dates_1 = dates.date2num(liste_heure)
            
            
            ax1[0].cla()
            
            ax2.cla()
            ax2.set_axis_off()
            
            ax3.cla()
            ax3.set_axis_off()
            
            ax1[1].cla()

            
            ax1[0].plot_date(dates_1,y,'-' ,label='vitesse',color="red")
            ax1[0].set_ylabel("Vitesse",color='red')
            ax1[0].tick_params(axis ='y', labelcolor = 'red') 
            ax1[0].set_title("Message ID : " + file_path.split("/")[-1].split('.')[0] +'    |    MMSI : ' +str(df['MMSI:'][0] ) +'    |    Année : ' + str(liste_heure[0]).split('-')[0] +'     |    nb de msg : ' +str(len(y)))
            ax1[0].set_xlim(dates_1[0],dates_1[-1])
            ax1[0].set_ylim(ymin=min(y)-0.5, ymax=max(y)+0.5)
            ax1[0].grid()
            try:
                ax6.cla()
                ax6.set_axis_off()
            except Exception as e1:
                print( e1.__class__, "occurred.")   
            try:
                ax7.cla()
                ax7.set_axis_off()
            except Exception as e2:
                print( e2.__class__, "occurred.") 
            
    
            ax4 = ax1[0].twinx() 
            ax4.plot_date(dates_1,y2,'-', color = 'green', label ='Statut de navigation') 
            ax4.set_ylim(ymin=min(y2)-0.5,ymax=max(y2)+0.5)
    
            ax4.tick_params(axis ='y', labelcolor = 'green') 
            ax4.set_ylabel("Statut de navigation",color='green')
            ax4.set_xlim(dates_1[0],dates_1[-1])
            ax6=ax4
    
    
            
            
            
            ax1[1].grid()
            ax1[1].plot_date(dates_1,y1,'-' ,label='ecart',color="blue")
            ax1[1].set_ylabel("Ecart",color='blue')
            ax1[1].tick_params(axis ='y', labelcolor = 'blue') 
            ax1[1].set_ylim(ymin=min(y1)-0.5, ymax=max(y1)+0.5)
            ax1[1].set_xlim(dates_1[0],dates_1[-1])
            
            
            ax5 = ax1[1].twinx() 
            ax5.axis()
            ax5.plot_date(dates_1,y2,'-', color = 'green', label ='Statut de navigation') 
            ax5.set_ylim(ymin=min(y2)-0.5,ymax=max(y2)+0.5)
            
    
            ax5.tick_params(axis ='y', labelcolor = 'green') 
            ax5.set_ylabel("Statut de navigation",color='green')
            ax5.set_xlim(dates_1[0],dates_1[-1])
            ax7=ax5
            
            #------------------------------------------------------------------
            # si il exite déjà des cas de dérive les visualiser en noir
            liste_lines=[]
            if m == 1:
                for cas in liste_derive(df['avis']):
                    y_vitesse_derive = df['Vitesse (noeuds) :'][cas[0]:cas[1]+1]
                    ax1[0].plot_date(dates_1[cas[0]:cas[1]+1],y_vitesse_derive,'-',color="black",scalex=False, scaley=False)
                    y_ecart_derive = df['diff'][cas[0]:cas[1]+1]
                    ax1[1].plot_date(dates_1[cas[0]:cas[1]+1],y_ecart_derive,'-',color="black",scalex=False, scaley=False)
                    
                    
                    line1=ax1[0].axvline(x = dates_1[cas[0]], color = 'black', label = 'axvline - full height')
                    line2=ax1[0].axvline(x = dates_1[cas[1]], color = 'black', label = 'axvline - full height')

                    line3=ax1[1].axvline(x = dates_1[cas[0]], color = 'black', label = 'axvline - full height')
                    line4=ax1[1].axvline(x = dates_1[cas[1]], color = 'black', label = 'axvline - full height')
                    
                    liste_lines.append(line1)
                    liste_lines.append(line2)
                    liste_lines.append(line3)
                    liste_lines.append(line4)


            
            #------------------------------------------------------------------
    
            
            cursor_2 = MultiCursor(fig.canvas, (ax4, ax5), color='dimgrey', lw=2, horizOn=False, vertOn=True)
    
            
            plt.draw()
            fig.canvas.draw()
            
            liste_pts=[]
            liste_plot=[]
            fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, df,df_copy,ax4,ax5,file_path))
            try:
                axes2.cla()
                axes2.set_axis_off()
                
                axes3.cla()
                axes3.set_axis_off()
            except Exception as e0:
                print( e0.__class__, "occurred.")   
            axes1 = plt.axes([0.15, 0.05, 0.12, 0.075])
            Button(axes1, 'Supprimer les cas de dérive')
            axes2=axes1
            
            fig.canvas.mpl_connect('button_press_event', lambda event: supprimer(event, df,df_copy,axes1,file_path))
            
            axes_deplacer1 = plt.axes([0.15, 0.9, 0.075, 0.05])
            Button(axes_deplacer1, 'Exclure le fichier')
            axes3=axes_deplacer1
            
            fig.canvas.mpl_connect('button_press_event', lambda event: deplacer(event,df_copy,axes_deplacer1,file_path))
            
            fig.canvas.draw()
            plt.draw()

        else :
            plt.close()

            print("fin")
            
            

            


 
    

def onclick(event,df,df_copy,a,b,file_path):
    

    x = event.xdata
    if event.inaxes == a or event.inaxes == b :
        print('position souris : ',x)
    
        liste_heure=[]
        for i in df.index:
            liste_heure.append(datetime.utcfromtimestamp(df['Horodatage Epoch :'][i]))
            
    
        dates_1 = dates.date2num(liste_heure)
        
        idx = (np.abs(dates_1-x)).argmin()
        x=dates_1[idx]
        
        if event.inaxes == a or event.inaxes == b :
            
            plot1=ax1[0].axvline(x = x, color = 'black', label = 'axvline - full height')
            plot2=ax1[1].axvline(x = x, color = 'black', label = 'axvline - full height')
            liste_plot.append(plot1)
            liste_plot.append(plot2)
            
            fig.canvas.draw()
    
            liste_pts.append(idx)
            if len(liste_pts)%2 == 0 : 
                        
                debut = []
                fin = []
        
                for i in range(len(liste_pts)):
                    if i % 2 == 0:
                        debut.append(liste_pts[i])
                    else : 
                        fin.append(liste_pts[i])
                        
                for j in range(len(debut)):
                    if debut[j]>fin[j]:
                        c = fin[j]
                        fin[j]=debut[j]
                        debut[j]=c
                    
    
                    intervale=dates_1[np.arange(debut[j],fin[j]+1)]
                    df_copy['avis'][np.arange(debut[j],fin[j]+1)]=1
                    y = df['Vitesse (noeuds) :'][np.arange(debut[j],fin[j]+1)]
                    y1 = df['diff'][np.arange(debut[j],fin[j]+1)]
                    
                    
                    ax1[0].plot_date(intervale,y,'-',color="black",scalex=False, scaley=False)
                    ax1[1].plot_date(intervale,y1,'-',color="black",scalex=False, scaley=False)
        
                    fig.canvas.draw()
                df_copy.to_excel(file_path,index=False)
                

def supprimer(event,df,df_copy,axe,file_path):
        
    global liste_pts,liste_plot,liste_lines
    event.xdata
    if event.inaxes == axe :
        
        liste_heure=[]
        liste_pts=[]
        for i in df.index:
            liste_heure.append(datetime.utcfromtimestamp(df['Horodatage Epoch :'][i]))
            

        dates_1 = dates.date2num(liste_heure)
        

        
        # x = df.index
        y = df['Vitesse (noeuds) :']
        y1= df["diff"]
        ax1[0].plot_date(dates_1,y,'-' ,label='vitesse',color="red",scalex=False, scaley=False)
        ax1[1].plot_date(dates_1,y1 ,'-',label='Écart',color='blue',scalex=False, scaley=False)
        for i in liste_plot:
            i.remove()
        liste_plot=[]
        for i in liste_lines:
            i.remove()
        liste_lines=[]
        
        fig.canvas.draw()
        
        df_copy['avis']=0
        df_copy.to_excel(file_path,index=False)
        

def deplacer(event,df_copy,axe,file_path):
    
    event.xdata
    if event.inaxes == axe:
        
        if os.path.exists(file_path):
            os.remove(file_path)
        
        liste_file_path = file_path.split('/')
        new_path = ''
        for i in range(len(liste_file_path)-2):
            new_path = new_path + liste_file_path[i]+'/'
        new_path = new_path + 'fichiers_exclus_' + liste_file_path[-2]+'/'
        
        try:
            os.mkdir(new_path)
        except Exception as e:
            print( e.__class__, "occurred.") 
        file_number = file_path.split("/")[-1]
        df_copy.to_excel(new_path+file_number,index=False)
        callback.next
                
        

def run(liste_path):
    
    global cursor,cursor1,ax1,fig,ax2,ax3,callback
    global liste_path1,l,l1,l2,l3,bprev,bnext,liste_plot,liste_lines,liste_pts
    liste_path1=liste_path
    path1=liste_path[0]
    liste_pts=[]
    liste_plot=[]
    df = pandas.read_excel(path1)

    # copie de la data_frame
    df_copy = df.copy()

    # calcul de la différence entre le COG et le TH
    df.loc[df["Cap () :"] > 180 ,"Cap () :" ] = 360 - df["Cap () :"]
    df.loc[df["Cap vrai () :"] > 180 ,"Cap vrai () :" ] = 360 - df["Cap vrai () :"]
    df['diff']=abs(df['Cap vrai () :']-df['Cap () :'])
        
        
    # sauvegarder les valeurs à changer    
    columns = df.columns
    m=0
    for i in columns:
        if i == 'avis':
            m=1
    
    if m == 0 :
        liste = [0 for i in df.index]
        df_copy.insert(len(df_copy.columns),'avis',liste,True)
        df_copy.to_excel(path1,index=False)    
        
    
    y = df['Vitesse (noeuds) :']
    y1= df["diff"]
    y2= df['Statut de navigation :']

    liste_heure=[]
    for i in df.index:
        liste_heure.append(datetime.utcfromtimestamp(df['Horodatage Epoch :'][i]))
        
    
    dates_1 = dates.date2num(liste_heure)
    
    
    fig = plt.figure()
    ax1 = fig.subplots(2) 
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(50,150,1800, 780)
    # mngr.window.setGeometry(50,150,900, 480)
    plt.subplots_adjust(top = 0.85, bottom = 0.2)


    ax1[0].set_title("Message ID : " + path1.split("/")[-1].split('.')[0] +'    |    MMSI : ' +str(df['MMSI:'][0] ) +'    |    Année : ' + str(liste_heure[0]).split('-')[0] +'     |    nb de msg : ' +str(len(y)))
    
    #----------------------- Vitesse -------------------------------------------
    
    ax1[0].set_ylim(ymin=min(y)-0.5, ymax=max(y)+0.5)
    ax1[0].set_xlim(dates_1[0],dates_1[-1])
    l, = ax1[0].plot_date(dates_1,y,'-' ,label='vitesse',color="red")
    ax1[0].set_ylabel("Vitesse",color='red')
    ax1[0].tick_params(axis ='y', labelcolor = 'red') 
    
    
    ax2 = ax1[0].twinx() 
    l1, = plt.plot_date(dates_1,y2,'-', color = 'green', label ='Statut de navigation') 
    ax2.set_ylim(ymin=min(y2)-0.5,ymax=max(y2)+0.5)
    ax2.tick_params(axis ='y', labelcolor = 'green') 
    ax2.set_ylabel("Statut de navigation",color='green')
    ax1[0].grid()
    
    #------------------------- Ecart -----------------------------------------
    
    ax1[1].set_ylim(ymin=min(y1)-0.5, ymax=max(y1)+0.5)
    l2, = ax1[1].plot_date(dates_1,y1,'-' ,label='Écart',color="blue")
    ax1[1].set_ylabel("Écart",color='blue')
    ax1[1].tick_params(axis ='y', labelcolor = 'blue') 
    ax1[1].set_xlim(dates_1[0],dates_1[-1])
    
    
    ax3 = ax1[1].twinx() 
    l3, = plt.plot_date(dates_1,y2,'-', color = 'green', label ='Statut de navigation') 
    ax3.set_ylim(ymin=min(y2)-0.5,ymax=max(y2)+0.5)
    ax3.tick_params(axis ='y', labelcolor = 'green') 
    ax3.set_ylabel("Statut de navigation",color='green')
    ax1[1].grid()


    #------------------------------------------------------------------
    # si il exite déjà des cas de dérive les visualiser en noir
    liste_lines=[]
    if m == 1:
        for cas in liste_derive(df['avis']):
            y_vitesse_derive = df['Vitesse (noeuds) :'][cas[0]:cas[1]+1]
            ax1[0].plot_date(dates_1[cas[0]:cas[1]+1],y_vitesse_derive,'-',color="black",scalex=False, scaley=False)
            y_ecart_derive = df['diff'][cas[0]:cas[1]+1]
            ax1[1].plot_date(dates_1[cas[0]:cas[1]+1],y_ecart_derive,'-',color="black",scalex=False, scaley=False)
            
            line1=ax1[0].axvline(x = dates_1[cas[0]], color = 'black', label = 'axvline - full height')
            line2=ax1[0].axvline(x = dates_1[cas[1]], color = 'black', label = 'axvline - full height')

            line3=ax1[1].axvline(x = dates_1[cas[0]], color = 'black', label = 'axvline - full height')
            line4=ax1[1].axvline(x = dates_1[cas[1]], color = 'black', label = 'axvline - full height')
            liste_lines.append(line1)
            liste_lines.append(line2)
            liste_lines.append(line3)
            liste_lines.append(line4)

    
    #------------------------------------------------------------------
    
    cursor = MultiCursor(fig.canvas, (ax2, ax3), color='dimgrey', lw=2, horizOn=False, vertOn=True)
    
    fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, df,df_copy,ax2,ax3,path1))

    
    
    callback = Index()
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Suivant')
    bnext.on_clicked(callback.next)

    
    axes = plt.axes([0.15, 0.05, 0.12, 0.075])
    Button(axes, 'Supprimer les cas de dérive')
    
    
    fig.canvas.mpl_connect('button_press_event', lambda event: supprimer(event, df,df_copy,axes,path1))
    
    axes_deplacer = plt.axes([0.15, 0.9, 0.075, 0.05])
    Button(axes_deplacer, 'Exclure le fichier')
    
    fig.canvas.mpl_connect('button_press_event', lambda event: deplacer(event,df_copy,axes_deplacer,path1))
    
    plt.show()
    
    
    file_path = 'C:/Users/azizb/Desktop/data_2014_2018/annee_2014/555.xlsx'
    liste_file_path = file_path.split('/')
    new_path = ''
    for i in range(len(liste_file_path)-1):
        new_path = new_path + new_path[i]+'/'
    print(new_path)
    