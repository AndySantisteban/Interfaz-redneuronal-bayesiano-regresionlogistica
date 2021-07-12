from os import path
from tkinter import *
from tkinter import ttk
from numpy import random, array
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from typing import List
from numpy import integer
from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


red = tk.Tk()
red.title("Red_neuronal")
red.geometry("1200x600") 
red.pack_propagate(False) 
red.resizable(0, 0)
red.configure(background='RoyalBlue1')

 # Frame for TreeView
frame1 = tk.LabelFrame(red, text="Parámetros")
frame1.place(height=250, width=490,x=20,rely=0.5)

frame2 = tk.LabelFrame(red, text="Dataset")
frame2.place(height=250, width=490,x=20,y=5)

frame3 = tk.LabelFrame(red, text="Dataset Carged")
frame3.place(height=250, width=490,relx=.5,y=5)

frame4= tk.LabelFrame(red, text="Resultados")
frame4.place(height=250, width=490,relx=.5,rely=0.5)
button1 = tk.Button(frame2, text="Browse A File", command=lambda: File_dialog(),bg="gold")
button1.place(rely=0.8, relx=0.51)
button2 = tk.Button(frame2, text="Load File", command=lambda: Load_excel_data(),bg="SeaGreen1")
button2.place(rely=0.8, relx=0.30)
button2 = tk.Button(frame1, text="Cancel", command=lambda: Load_excel_data(),bg="orange red")
button2.place(rely=0.8, relx=0.6)
#
entrada_batch_size=StringVar()
campo_batch_size=Entry(frame1,textvariable=entrada_batch_size).place(rely=0.01, relx=0.4)
entrada_funct=StringVar()
campo_func=Entry(frame1,textvariable=entrada_funct).place(rely=0.2, relx=0.4)
entrada_power=StringVar()
campo_power=Entry(frame1,textvariable=entrada_power).place(rely=0.4, relx=0.4)
entrada_momentum=StringVar()
campo_momentum=Entry(frame1,textvariable=entrada_momentum).place(rely=0.6, relx=0.4)

label_file = ttk.Label(frame1, text="batch size")
label_file.place(rely=0.01, relx=0.01)
label_file2 = ttk.Label(frame1, text="power")
label_file2.place(rely=0.4, relx=0.01)
label_file3 = ttk.Label(frame1, text="function")
label_file3.place(rely=0.2, relx=0.01)
label_file3 = ttk.Label(frame1, text="momentum")
label_file3.place(rely=0.6, relx=0.01)





label_file = ttk.Label(frame2, text="No File Selected")
label_file.place(rely=0, relx=0)
tv1 = ttk.Treeview(frame3)

tv1.place(relheight=1, relwidth=1) 
treescrolly = tk.Scrollbar(frame3, orient="vertical", command=tv1.yview) 
treescrollx = tk.Scrollbar(frame3, orient="horizontal", command=tv1.xview) 
tv1.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set) 
treescrollx.pack(side="bottom", fill="x") 
treescrolly.pack(side="right", fill="y") 

label_resultados = ttk.Label(frame4, text="")
label_resultados.place(rely=0, relx=0)
""""""
def File_dialog():
    """This Function will open the file explorer and assign the chosen file path to label_file"""
    filename = filedialog.askopenfilename(
                            initialdir="/", 
                            filetypes =(("All Files","*.*"),("jpeg", "*.jpg")                    ) ,                          
                                            )
    label_file["text"] = filename
    return None
def graficar():
  df=pd.read_csv("dataset_general_covid_19_red_neuronal.csv",sep=";")
  X=df.drop(['recuperados','Unnamed: 0'], axis='columns')
  Y=df.recuperados
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
  print(len(X_train))
  print(len(X_test))
  
  func = 'tanh'
  model = MLPClassifier(activation=func,solver='sgd',learning_rate_init=0.01,
                        hidden_layer_sizes=5,momentum=0.2,
                        random_state=1, max_iter=1000,power_t=0.5,shuffle=True,
                        verbose=True,batch_size=200,n_iter_no_change=200,nesterovs_momentum=True,
                        learning_rate='constant').fit(X_train, Y_train)
  #model.predict_proba(X_test[:1])
  model.predict(X_test)
  #Obtener y mostrar la precisión del modelo entrenado
  model.score(X_train, Y_train)
  percentage = model.score(X_test, Y_test)
  #Probando el modelo con el dataset de prueba (X_test)
  predictions = model.predict(X_test)
  #mostrar el vector de resultados generados por el modelo para cada ejemplo del dataset de prueba
  from sklearn.metrics import confusion_matrix
  res = confusion_matrix(Y_test, predictions)
  Precision= percentage
  from sklearn.metrics import classification_report
  from matplotlib import pyplot as plt
  from sklearn.metrics import plot_roc_curve
  target_names = ['SI', 'NO']
  report = classification_report(Y_test, predictions, target_names=target_names)
   
   
  lbl = Label(frame1, text=plot_roc_curve(model, X_test, Y_test)).place(rely=0, relx=0)

def exec():
  df=pd.read_csv("dataset_general_covid_19_red_neuronal.csv",sep=";")
  X=df.drop(['recuperados','Unnamed: 0'], axis='columns')
  print(X)
  Y=df.recuperados
  Y.head()
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
  print(len(X_train))
  print(len(X_test))
  batch =int(entrada_batch_size.get())
  pw=int(entrada_power.get())/100
  func=entrada_funct.get()
  momem=int(entrada_momentum.get())/100
  model = MLPClassifier(activation=func,solver='sgd',learning_rate_init=0.01,
                        hidden_layer_sizes=(5),momentum=momem,
                        random_state=1, max_iter=1000,power_t=pw,shuffle=True,
                        verbose=False,batch_size=batch,n_iter_no_change=200,nesterovs_momentum=True,
                        learning_rate='constant').fit(X_train, Y_train)
  #model.predict_proba(X_test[:1])
  model.predict(X_test)
  #Obtener y mostrar la precisión del modelo entrenado
  model.score(X_train, Y_train)
  percentage = model.score(X_test, Y_test)
  percentage
  #Probando el modelo con el dataset de prueba (X_test)
  predictions = model.predict(X_test)
  #mostrar el vector de resultados generados por el modelo para cada ejemplo del dataset de prueba
  print(predictions)
  from sklearn.metrics import confusion_matrix
  res = confusion_matrix(Y_test, predictions)
  print("Matrix de confusión")
  print(res)
  #cantidad de ejemplos de prueba
  print(f"Dataset de prueba: {len(X_test)}")
  print(f"Accuracy = {percentage*100} %") #Precisión
  Precision= percentage
  from sklearn.metrics import classification_report
  from matplotlib import pyplot as plt
  from sklearn.metrics import plot_roc_curve
  target_names = ['SI', 'NO']
  report = classification_report(Y_test, predictions, target_names=target_names)
  plt.show()
  lb3 = Label(frame4, text=report).place(rely=0.3, relx=0.4)
  lbl = Label(frame4, text="report:").place(rely=0.3, relx=0.01)
  lb4 = Label(frame4, text=Precision).place(rely=0.15, relx=0.4)
  lbl2 = Label(frame4, text="Precision").place(rely=0.15, relx=0.01)
def Load_excel_data():
    button3 = tk.Button(frame1, text="Ejecutar", command=lambda: exec(), bg="coral1")
    button3.place(rely=0.8, relx=0.01)
    """If the file selected is valid this will load the file into the Treeview"""
    file_path = label_file["text"]
    try:
        excel_filename = r"{}".format(file_path)
        if excel_filename[-4:] == ".csv":
            df = pd.read_csv(excel_filename, sep=";")
            
        else:
            df = pd.read_excel(excel_filename)

    except ValueError:
        tk.messagebox.showerror("Information", "The file you have chosen is invalid")
        return None
    except FileNotFoundError:
        tk.messagebox.showerror("Information", f"No such file as {file_path}")
        return None

    clear_data()
    tv1["column"] = list(df.columns)
    tv1["show"] = "headings"
    for column in tv1["columns"]:
        tv1.heading(column, text=column) 

    df_rows = df.to_numpy().tolist() 
    for row in df_rows:
        tv1.insert("", "end", values=row) 
    return None


def clear_data():
    tv1.delete(*tv1.get_children())
    return None





red.mainloop()