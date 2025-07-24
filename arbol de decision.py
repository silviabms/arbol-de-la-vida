# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import seaborn as sb
import streamlit as st 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# titulo para que salga en la pagina web
st.title("Modelo de prediccion de aprobacionn de estudiante con arbol de decision ")
st markdown(Este modelo usa notas: parciales, proyecto y examen final para predecir)
# acragar los datos (cargar el conjunto de datps para su analisis)
@st.cache datos():
     
return pd.read_csv("estudiantes_notas_finales_arboles.csv")
# recibiendo los datos cargados en la variable df.(antes llamda dataset)
df=cargar_datos()
# mostrar los primeros datos (cinco primeros)
st.subheader("datos cargados")
st.write(df.head()) #esta intruccion muestra los primero cinco datos
# Gráficos simples
st.subheader("Distribución de notas") # Titulo que aparece en la pagina web
st.bar_chart(df[["Primer Parcial", "Segundo Parcial", "Proyecto", "Examen Final", "Nota Final"]].mean())
