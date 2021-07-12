from typing import List
from numpy import integer
import pandas as pd
from sklearn.model_selection import train_test_split
from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from tkinter import filedialog, messagebox, ttk

import pandas as pd

# initalise the tkinter GUI
root = tk.Tk()
root.geometry("500x500") 
root.pack_propagate(False) 
root.resizable(0, 0) 
root.title("Dataset Tratamiento")

# Frame for TreeView
frame1 = tk.LabelFrame(root, text="Csv Data")
frame1.place(height=250, width=500,x=5,y=5, )

# Frame for open file dialog
file_frame = tk.LabelFrame(root, text="Open File")
file_frame.place(height=200, width=400, rely=0.5, relx=0.08,x=5,y=5)

label_file = ttk.Label(file_frame, text="No File Selected")
label_file.place(rely=0, relx=0)
# Buttons
button1 = tk.Button(file_frame, text="Browse A File", command=lambda: File_dialog(),bg="gold")
button1.place(rely=0.8, relx=0.1)

button2 = tk.Button(file_frame, text="Load File", command=lambda: Load_excel_data(),bg="SeaGreen1")
button2.place(rely=0.8, relx=0.40)


# The file/file path text
label_file = ttk.Label(file_frame, text="No File Selected")
label_file.place(rely=0, relx=0)

label_test=ttk.Label(file_frame,text="Length Test:")
label_test.place(rely=0.2, relx=0)

label_train=ttk.Label(file_frame,text="Length Train:")
label_train.place(rely=0.4, relx=0)

label_validation=ttk.Label(file_frame,text="Length Validation:")
label_validation.place(rely=0.6, relx=0)
## Treeview Widget
tv1 = ttk.Treeview(frame1)
tv1.place(relheight=1, relwidth=1) 

treescrolly = tk.Scrollbar(frame1, orient="vertical", command=tv1.yview) 
treescrollx = tk.Scrollbar(frame1, orient="horizontal", command=tv1.xview) 
tv1.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set) 
treescrollx.pack(side="bottom", fill="x") 
treescrolly.pack(side="right", fill="y") 

def Dividir_dataset():
    df= pd.read_csv('dataset_general_covid_19_red_neuronal.csv', sep=";")
    df = df.drop(['Unnamed: 0'], axis=1)
    print(df)
    train, test_validation = train_test_split(df, test_size = 0.30)
    n= df.__len__
    print("Ejemplos usados para entrenar: ", len(train))
    print("Ejemplos usados para test: ", len(test_validation))

    test, validation = train_test_split(test_validation, test_size = 0.33)

    print("Ejemplos usados para entrenar: ", len(test))
    print("Ejemplos usados para test: ", len(validation))
    
    entrada=StringVar()
    entrada.set(len(train))
    campo=Entry(file_frame,textvariable=entrada).place(rely=0.2, relx=0.3)

    entrada2=StringVar()
    entrada2.set(len(test))
    campo2=Entry(file_frame,textvariable=entrada2).place(rely=0.4, relx=0.3)

    entrada3=StringVar()
    entrada3.set(len(validation))
    campo3=Entry(file_frame,textvariable=entrada3).place(rely=0.6, relx=0.3)

    test.to_csv('test.csv')
    train.to_csv('train.csv')
    validation.to_csv('validation.csv')
    return None
def File_dialog():
    """This Function will open the file explorer and assign the chosen file path to label_file"""
    filename = filedialog.askopenfilename(
                            initialdir="/", 
                            filetypes =(("All Files","*.*"),("jpeg", "*.jpg")                    ) ,                          
                                            )
    label_file["text"] = filename
    return None


def Load_excel_data():
    """If the file selected is valid this will load the file into the Treeview"""
    button3 = tk.Button(file_frame, text="Split Dataset", command=lambda: Dividir_dataset(),bg="coral1")
    button3.place(rely=0.8, relx=0.7)
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


root.mainloop()