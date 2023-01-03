# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 22:51:10 2022

@author: CCD
"""

import pandas as pd
import os
import datetime
import numpy as np
from sklearn.cluster import AgglomerativeClustering as HAC

ruta_output = 'Datos/Output/'
fechas = [f.name for f in os.scandir(ruta_output) if f.is_dir()]
fechas.remove('20210726')

df_votos_agregado = pd.DataFrame({'fecha':[], 'n_votacion':[],
                                  'nombre':[], 'partido':[],
                                  'voto':[]})

i= 0
for f in fechas:
    date = datetime.datetime.strptime(f, '%Y%m%d')
    ruta_fecha = os.path.join(ruta_output, f)
    votos = [v.name for v in os.scandir(ruta_fecha) if v.is_dir()]
    for v in votos:
        voto_num = int(v.split('_')[1])
        ruta_voto = os.path.join(ruta_fecha, v)
        df_votos = pd.read_json(ruta_voto + '/votos_final.json')
        df_votos['fecha'] = date
        df_votos['n_votacion'] = voto_num
        
        df_votos_agregado = pd.concat([df_votos_agregado, df_votos])
        i += 1
print(i)
df_votos_agregado = df_votos_agregado.pivot(index=['fecha', 'n_votacion'], columns=['nombre'], values='voto').sort_index()
df_votos_agregado.to_csv('Datos/Analisis/Votos.csv')
#Limpiando votos
def EtiquetaVoto(x):
    map_votos = {'SI +++': 1, 'NO ---': -1, 'ABST.': 0}
    if x not in map_votos:
        return np.nan
    else:
        return map_votos[x]
    
df_votos_agregado = df_votos_agregado.applymap(EtiquetaVoto)


#Limpiando declarativos
def LimpiaDeclarativo(row):
    favor = row[row==1].count()
    contra = row[row==-1].count()
    if favor >= 104 and contra < 10:
        return row.apply(lambda x: np.nan)
    else:
        return row

df_votos_agregado = df_votos_agregado.apply(LimpiaDeclarativo, axis = 1)


#1: Calculando % de veces que votaron igual
def CalculaSimilitud(df):
    print('Evaluando similitud de voto')
    congresistas = list(df.columns)
    # similitud = {}
    matriz_similitud = np.zeros((len(congresistas), len(congresistas)))
    for i, c1 in enumerate(congresistas):
        # similitud[c1] = {}
        print('\t' +'(' + str(i+1) + '/' + str(len(congresistas)) + ') Calculando similitud de ' + c1 )
        for j, c2 in enumerate(congresistas[i+1:]):
            congs = [c1, c2]
            df_comp = df.loc[:,congs]
            df_comp['check'] = df_comp[c1] == df_comp[c2]
            df_comp['sum'] = df_comp[c1] + df_comp[c2]
            votos_validos = len(df_comp) - df_comp['sum'].isna().sum()
            
            thresh = 80
            if votos_validos < thresh:
                simil = np.nan
            else:
                simil = df_comp['check'].sum()/votos_validos
            # similitud[c1][c2] = df_comp['check'].sum()/len(df_comp)
            matriz_similitud[i, i + j + 1] = simil
            
    matriz_similitud = matriz_similitud + matriz_similitud.T 
    np.fill_diagonal(matriz_similitud, 1)
    df_similitud = pd.DataFrame(data = matriz_similitud, index = congresistas, columns = congresistas)
    return df_similitud

matriz_similitud = CalculaSimilitud(df_votos_agregado)
matriz_distancias = 1 - matriz_similitud


#2 Limpia Congresistas con insuficiente info
def LimpiaCongresista(df):
    df_final = df.copy(deep=True)
    for cong in df:
        if np.sum(np.isnan(df[cong]))== len(df)-1:
            print('Congresista ' + cong + ' se elimina por falta de datos')
            df_final = df_final.drop(index=cong, columns=cong)
    return df_final
matriz_distancias = LimpiaCongresista(matriz_distancias)


#3 Corriendo modelo de agrupamiento
Agglom = HAC(linkage = 'complete', affinity='precomputed', n_clusters=10)
# Agglom = HAC(linkage = 'complete', affinity='precomputed', distance_threshold=0.25, n_clusters = None)

Clustering = Agglom.fit(matriz_distancias)

def SortMatriz(df, model):
    cong = df.index
    labels = model.labels_
    dict_clusters = dict(zip(cong, labels))
    df_clusters = pd.DataFrame.from_dict(dict_clusters, orient='index', columns=['Cluster']).reset_index()
    df_clusters.sort_values(by=['Cluster', 'index'], inplace=True)
    ordered_cong = list(df_clusters['index'])
    return df.reindex(index = ordered_cong, columns = ordered_cong), df_clusters
matriz_distancias_ordered, df_clusters = SortMatriz(matriz_distancias, Clustering)

matriz_distancias_ordered.to_excel('Datos/Analisis/Distancias Sorted.xlsx')
df_clusters.to_excel('Datos/Analisis/Clusters.xlsx')


import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['axes.facecolor'] = (1,1,1)
plt.rcParams['figure.facecolor'] = (1,1,1)
plt.rcParams['figure.figsize'] = 6, 5
plt.rcParams['figure.dpi'] = 120

plt.figure()
plt.title('Matriz de distancias')
ax = sns.heatmap(matriz_distancias_ordered,cmap="rocket_r", yticklabels=False, xticklabels=False)
plt.show()

plt.figure()
plt.title('Matriz de distancias')
ax = sns.heatmap(matriz_distancias,cmap="rocket_r", yticklabels=False, xticklabels=False)
plt.show()






# from matplotlib import pyplot as plt
# from scipy.cluster.hierarchy import dendrogram

# def plot_dendrogram(model, **kwargs):
#     # Create linkage matrix and then plot the dendrogram

#     # create the counts of samples under each node
#     counts = np.zeros(model.children_.shape[0])
#     n_samples = len(model.labels_)
#     for i, merge in enumerate(model.children_):
#         current_count = 0
#         for child_idx in merge:
#             if child_idx < n_samples:
#                 current_count += 1  # leaf node
#             else:
#                 current_count += counts[child_idx - n_samples]
#         counts[i] = current_count

#     linkage_matrix = np.column_stack(
#         [model.children_, model.distances_, counts]
#     ).astype(float)

#     # Plot the corresponding dendrogram
#     dendrogram(linkage_matrix, **kwargs)

# plt.title("Hierarchical Clustering Dendrogram")
# # plot the top three levels of the dendrogram
# plot_dendrogram(Clustering, truncate_mode="level", p=3)
# plt.xlabel("Number of points in node (or index of point if no parenthesis).")
# plt.show()

# hierarchy.dendrogram(Z, orientation="right", labels=df.index)


matriz_similitud.to_excel('Datos/Analisis/similitud.xlsx')
