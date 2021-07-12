from os import name
from tkinter import *
import tkinter as tk 
from tkinter import Button, Tk ,Label , filedialog , ttk
from tkinter.filedialog import askopenfilename, test
from numpy import place
from numpy.core.fromnumeric import size

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

bayesiano=Tk()
bayesiano.title("BAYESIANO")
bayesiano.geometry("570x640")
bayesiano.resizable(False, False)

v_size = StringVar()


  
class Bayesiano:
  def __init__(self, x_train , y_train):
    self.nb = GaussianNB()
    self.nb.fit(x_train, y_train)

  def predecir(self, x_test):
    return self.nb.predict(x_test)

def Archivo():
    name = filedialog.askopenfilename(title= "Abrir", initialdir="D:/", filetypes =(("All Files","*.*"),("jpeg", "*.jpg"))  )
   
    datos = pd.read_csv(name, header=0)     
    file = open (name,'r')  
    print(file.read())
    print (datos)
    file.close()
    print(name)    
    ruta.delete(0,"end")    
    ruta.insert(0,name)
    text_cont.insert(tk.INSERT,datos)
    return datos
def mostrar():
  
  df_data = pd.read_csv('dataset_general_covid_19.csv')
  df_data.head()
    
  X = df_data.drop(["recuperados"], axis=1)
  y = df_data["recuperados"]


  x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)
  bayesiano = Bayesiano(x_train, y_train)
  
  y_predict = bayesiano.predecir(x_test)

  cm = confusion_matrix(y_test, y_predict)
  

  plt.figure(figsize=(3, 3))
  colormap = plt.cm.viridis
  sns.heatmap(cm, annot=True, fmt='g', cmap=colormap)
  recuperar_img = plt.show()


  print(classification_report(y_test, y_predict))

  repor = classification_report(y_test, y_predict)

  
  cont_matris.insert(tk.INSERT,cm)
  contreporte.insert(tk.INSERT, repor)
  imagen_b.insert(tk.INSERT, recuperar_img)
  

#name = filedialog.askopenfilename(title= "Abrir", initialdir="D:/", filetypes =(("All Files","*.*"),("jpeg", "*.jpg"))  )
   
frame1 = LabelFrame(bayesiano)
frame2 = LabelFrame(bayesiano)

frame1.pack(fill="both", expand="yes", padx=10, pady=10)
frame2.pack(fill="both", expand="yes", padx=10, pady=10)

btn_examinar = Button(frame1, text="Examinar", command=Archivo)
btn_examinar.pack()
btn_examinar.place(x=430,y=10,width=100,height=30)
ruta = Entry(frame1)
ruta.pack()
ruta.place(x=30,y=10,width=370,height=30)
ruta = Entry(frame1)
ruta.pack()
ruta.place(x=30,y=10,width=370,height=30)

resultado = Label(frame1, text = "Dataset"  )
resultado.pack()
resultado.place(x=10,y=40,width=100,height=30)

text_cont = Text(frame1)
text_cont.pack()
text_cont.place(x=30,y=65,width=300,height=160)

sise = Label(frame1, text = "Test_size"  )
sise.pack()
sise.place(x=330,y=60,width=100,height=30)

regulacion= Entry( frame1 ,textvariable= v_size )
regulacion.pack()
regulacion.place(x=360,y=90,width=120,height=30)


btn_eje = Button(frame1, text="Ejecutar", command=mostrar)
btn_eje.pack()
btn_eje.place(x=100,y=260,width=100,height=30)

reporte_matriz = Label(frame2, text = "Matriz"  )
reporte_matriz.pack()
reporte_matriz.place(x=30,y=40,width=130,height=30)

cont_matris = Text(frame2)
cont_matris.pack()
cont_matris.place(x=30,y=65,width=180,height=100)

imagen_b = Text( frame2)
imagen_b.pack()
imagen_b.place(x=31,y=100,width=180,height=100)

reporte = Label(frame2, text = "Reporte"  )
reporte.pack()
reporte.place(x=320,y=30,width=130,height=30)

contreporte = Text(frame2)
contreporte.pack()
contreporte.place(x=250,y=65,width=220,height=160)




bayesiano.mainloop()


