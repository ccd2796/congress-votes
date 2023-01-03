# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 07:05:40 2022

@author: Asus
"""

import pandas as pd
import os
import datetime
import numpy as np

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

#Limpiando votos
def LimpiaVotos(x):
    map_votos = {'SI +++': 1, 'NO ---': -1, 'ABST.': 0}
    if x not in map_votos:
        return np.nan
    else:
        return map_votos[x]
    
df_votos_agregado = df_votos_agregado.applymap(LimpiaVotos)


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
matriz_similitud.to_excel('Datos/Analisis/similitud.xlsx')
