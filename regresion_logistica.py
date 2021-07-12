import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import seaborn as sb
from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

root = tk.Tk()
root.title("Regresion Logistica Intefaz")
root.geometry("500x600") # set the root dimensions
root.pack_propagate(False) # tells the root to not let the widgets inside it determine its size.
root.resizable(0, 0)
root.configure(background='LightBlue3')
# Frame for TreeView
frame1 = tk.LabelFrame(root, text="Parametros")
frame1.place(height=250, width=490,x=5,y=5, )

file_frame = tk.LabelFrame(root, text="Open File")
file_frame.place(height=100, width=400, rely=0.5, relx=0.08,x=5,y=5,)

frame_resultado= tk.LabelFrame(root,text="Resultados", )
frame_resultado.place(height=100,width=490,rely=0.7,relx=0.01,)
label1=ttk.Label(frame1,text="Seed:")
label1.place(rely=0.01, relx=0.01)

label2=ttk.Label(frame1,text="n_splits:")
label2.place(rely=0.2, relx=0.01)

label3=ttk.Label(frame1,text="Validation_size:")
label3.place(rely=0.4, relx=0.01)

nick = tk.StringVar()
entrada_seed=StringVar()
campo_seed=Entry(frame1,textvariable=entrada_seed).place(rely=0.01, relx=0.4)
entrada_test=StringVar()
campo_test=Entry(frame1,textvariable=entrada_test).place(rely=0.2, relx=0.4)
entrada_validation=StringVar()
campo_validation=Entry(frame1,textvariable=entrada_validation).place(rely=0.4, relx=0.4)


label_file = ttk.Label(file_frame, text="No File Selected")
label_file.place(rely=0, relx=0)


button2 = tk.Button(file_frame, text="Load File", command=lambda: Load_excel_data(), bg="coral1")
button2.place(rely=0.5, relx=0.30)
button1 = tk.Button(file_frame, text="Browse A File", command=lambda: File_dialog(), bg="SeaGreen1")
button1.place(rely=0.5, relx=0.51)



def Calculo():
  dataframe = pd.read_csv('dataset_general_covid_19_red_neuronal.csv',sep=";")  
  dataframe=dataframe.drop(['Unnamed: 0'],axis="columns")
  dataframe.drop(['recuperados'],1).hist()
  X = np.array(dataframe.drop(['recuperados'], axis="columns"))
  y = np.array(dataframe.recuperados)
  X.shape
  model = linear_model.LogisticRegression()
  model.fit(X,y)
  predictions = model.predict(X)
  model.score(X,y)
  seed =int(entrada_seed.get())
  validation_size=int(entrada_validation.get())/100
  
  X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
  name='Logistic Regression'
  kfold = model_selection.KFold(n_splits=20, random_state=seed,shuffle=True,)
  cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  predictions = model.predict(X_validation)
  score=accuracy_score(Y_validation, predictions)
  matriz=confusion_matrix(Y_validation, predictions)
  report=classification_report(Y_validation, predictions)
  resultado_regresion=StringVar()
  resultado_regresion.set(matriz[0])
  entrada_regresion=Entry(frame_resultado,textvariable=resultado_regresion).place(rely=0.01, relx=0.5)
  resultado_regresion2=StringVar()
  resultado_regresion2.set(matriz[1])
  entrada_regresion2=Entry(frame_resultado,textvariable=resultado_regresion2).place(rely=0.3, relx=0.5)

  resultado_regresion3=StringVar()
  resultado_regresion3.set(score)
  entrada_regresion3=Entry(frame_resultado,textvariable=resultado_regresion3).place(rely=0.6, relx=0.5, )
  print(score)
  print (matriz)
  print(report)

  label1=ttk.Label(frame_resultado,text="Reporte")
  label1.place(rely=0.1, relx=0.01)
  label2=ttk.Label(frame_resultado,text="Regresion Logistica:")
  label2.place(rely=0.5, relx=0.01)

  matriz.mainloop()  
def grafico():
  dataframe = pd.read_csv('dataset_general_covid_19_red_neuronal.csv',sep=";")
  dataframe.head()
  dataframe=dataframe.drop(['Unnamed: 0'],axis="columns")
  dataframe.head()
  dataframe.describe()
  print(dataframe.groupby('recuperados').size())
  dataframe.drop(['recuperados'],1).hist()
  X = np.array(dataframe.drop(['recuperados'], axis="columns"))
  y = np.array(dataframe.recuperados)
  X.shape
  model = linear_model.LogisticRegression()
  model.fit(X,y)
  predictions = model.predict(X)
  print(predictions)
  model.score(X,y)
  validation_size = 0.20
  seed =7
  X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
  name='Logistic Regression'
  kfold = model_selection.KFold(n_splits=20, random_state=seed,shuffle=True,)
  cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)
  predictions = model.predict(X_validation)
  score=accuracy_score(Y_validation, predictions)
  matriz=confusion_matrix(Y_validation, predictions)
  report=classification_report(Y_validation, predictions)
  resultado_regresion=StringVar()
  resultado_regresion.set(matriz[0])
  entrada_regresion=Entry(frame_resultado,textvariable=resultado_regresion).place(rely=0.01, relx=0.5)
  resultado_regresion2=StringVar()
  resultado_regresion2.set(matriz[1])
  entrada_regresion2=Entry(frame_resultado,textvariable=resultado_regresion2).place(rely=0.3, relx=0.5)

  resultado_regresion3=StringVar()
  resultado_regresion3.set(score)
  entrada_regresion3=Entry(frame_resultado,textvariable=resultado_regresion3).place(rely=0.6, relx=0.5, )
  print(score)
  print (matriz)
  print(report)

  label1=ttk.Label(frame_resultado,text="Reporte")
  label1.place(rely=0.1, relx=0.01)
  label2=ttk.Label(frame_resultado,text="Regresion Logistica:")
  label2.place(rely=0.5, relx=0.01)


  

  
  
def File_dialog():
    """This Function will open the file explorer and assign the chosen file path to label_file"""
    filename = filedialog.askopenfilename(
                            initialdir="/", 
                            filetypes =(("All Files","*.*"),("jpeg", "*.jpg")) ,                          
                                            )
    label_file["text"] = filename
    return None


def Load_excel_data():
    button3= tk.Button(frame1, text="Insertar",command=lambda: Calculo(), bg="coral1")
    button3.place(rely=0.6, relx=0.6)
    button3= tk.Button(frame1, text="Por defecto",command=lambda: grafico(), bg="SeaGreen1")
    button3.place(rely=0.6, relx=0.1)
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
    return None




root.mainloop()