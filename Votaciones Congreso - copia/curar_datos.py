# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 10:21:47 2022

@author: Asus
"""
def main(fecha):
    import cv2
    import pandas as pd
    import jellyfish
    import matplotlib.pyplot as plt
    import math
    
    plt.rcParams['figure.figsize'] = 7, 3
 
    plt.rcParams['figure.dpi'] = 120
    # plt.rcParams['figure.facecolor'] = (1,1,1)
    # plt.rcParams['axes.facecolor'] = (1,1,1)
    
    
    def read_args():
        ruta_args = "Datos/Input/" + fecha + ".txt"
        print("Leyendo argumentos en " + ruta_args)      
        args = {}
        with open(ruta_args) as f:
            for line in f:
               line_l = line.split()
               key = line_l[0]
               val = line_l[1:]
               val = [int(v) for v in val]
               
               if len(val) == 1:
                   args[key] = val[0]
               else:
                   args[key] = val
        return args 
        
    #funcion para ajustar imagen
    def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]
    
        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))
    
        return cv2.resize(image, dim, interpolation=inter)
    
    def ShowRow(image, y, col, val):
        img_width = math.floor(image.shape[1]/3)
        img_slice = image[y-5:y+20, img_width*(col-1):img_width*col]
        img_slice = cv2.cvtColor(img_slice, cv2.COLOR_BGR2RGB)
        
        slc_height = math.floor(img_slice.shape[0])
        slc_width = math.floor(img_slice.shape[1])
        if val == 'partido':
            img_slice = cv2.rectangle(img_slice, (0,slc_height), (math.floor(0.15*slc_width),0), (255,0,0), 4)
        elif val == 'voto':
            img_slice = cv2.rectangle(img_slice, (math.floor(0.85*slc_width),slc_height), (slc_width,0), (255,0,0), 4)
        plt.imshow(img_slice)
        plt.axis('off')
        plt.show()
    
    # fecha = "20210819"
    
    args = read_args()
    pags = args['pags']
    
    ruta_fecha = 'Datos/Output/' + fecha + '/'    
    
    #Guardando checks
    ruta_checks = 'Datos/Checks/'
    with open(ruta_checks + 'check_nombre.txt', encoding='utf-8') as f:
        check_nombre = f.readlines()
    with open(ruta_checks + 'check_voto.txt', encoding='utf-8') as f:
        check_voto = f.readlines()
    with open(ruta_checks + 'check_partido.txt', encoding='utf-8') as f:
        check_partido = f.readlines()
    
    check_nombre = [check.strip().upper() for check in check_nombre]
    check_voto = [check.strip().upper() for check in check_voto]
    check_partido = [check.strip().upper() for check in check_partido]
    
    if type(pags) == int:
        n_pags = 1
    else:
        n_pags = len(pags)
    #leyendo los votos
    
    for v in range(1, n_pags + 1):
    # for v in range(1, 2):
        print('Curando datos de la votación n° ' + str(v) + ' de ' + str(n_pags) + ' de la fecha ' + fecha)
        ruta_voto = ruta_fecha + 'voto_' + str(v) + '/'
        df_voto = pd.read_json(ruta_voto + 'votos_raw.json')
        df_coord = pd.read_json(ruta_voto + 'votos_coord.json')
        
        tabla_votos = cv2.imread(ruta_voto + 'tabla.jpg')
        # cv2.imshow("Tabla", ResizeWithAspectRatio(tabla_votos, width=700))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        for c in df_voto.index:
            #ubicando coordenada y columna
            yCoord = math.floor(df_coord.loc[c, 'yCoord'])
            col = df_coord.loc[c, 'col']
            
            #curando nombre
            nombre_raw = df_voto.loc[c, 'nombre'].upper()
            
            similarity = 0
            
            for n in check_nombre:
                sim = jellyfish.jaro_distance(n,nombre_raw)
                if sim > similarity:
                    similarity = sim
                    nombre_final = n
            
            df_voto.loc[c, 'nombre'] = nombre_final
            
            #curando partido
            partido_raw = df_voto.loc[c, 'partido'].upper()
            if partido_raw not in check_partido:
                ShowRow(tabla_votos, yCoord, col, 'partido')
                while True:
                    partido_final = input('PARTIDO - Ingresar partido de congresista ' + nombre_final  + '\n' +
                                          'Texto detectado: ' + partido_raw + '\n').upper()
                    if partido_final not in check_partido:
                        print('Partido no esta en lista check')
                    else:
                        plt.close()
                        break
            else:
                partido_final = partido_raw
            df_voto.loc[c, 'partido'] = partido_final
            
            #curando voto
            dict_check = {'SI': 'SI +++', 'NO':'NO ---'}
            
            voto_raw = df_voto.loc[c, 'voto'].upper()
            
            if voto_raw[:2] in dict_check:
                voto_raw = dict_check[voto_raw[:2]]
            
            if voto_raw not in check_voto:
                ShowRow(tabla_votos, yCoord, col, 'voto')
                while True:    
                    voto_final = input('VOTO - Ingresar voto de congresista ' + nombre_final + '\n' +
                                       'Texto detectado: ' + voto_raw + '\n').upper()
                    
                    if voto_final in dict_check:
                        voto_final = dict_check[voto_final]
                    
                    if voto_final.upper() not in check_voto:
                        print('Voto no esta en lista check')
                    else:
                        plt.close()
                        break
            else:
                voto_final = voto_raw
            df_voto.loc[c, 'voto'] = voto_final
        
        
        print('Curación completa - Fecha ' + fecha)
        df_voto.to_json(ruta_voto + 'votos_final.json')

if __name__ == '__main__':
    fecha = input('Ingresar fecha con datos a curar:\n')
    main(fecha)