# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 12:16:09 2022

@author: carlo
"""

def main(fecha):
    
    import cv2
    import pandas as pd
    import matplotlib.pyplot as plt
    import math
    
    plt.rcParams['figure.figsize'] = 20, 8
     
    plt.rcParams['figure.dpi'] = 480
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
    
    
    #Guardando checks
    ruta_checks = 'Datos/Checks/'
    with open(ruta_checks + 'check_voto.txt', encoding='utf-8') as f:
        check_voto = f.readlines()
    check_voto = [check.strip().upper() for check in check_voto]
    
    args = read_args()
    
    
    dict_check = {'SI': 'SI +++', 'NO':'NO ---'}
    
    
    if 'constancia-voto' in args:
        pags_const = args['constancia-voto']
        pags = args['pags']
        
        if type(pags_const) == int:
            n_pags_const = 1
            pags_const = [pags_const]
        else:
            n_pags_const = len(pags_const)
            
        if type(pags) == int:
            pags = [pags]

        ruta_fecha = 'Datos/Output/' + fecha + '/'    
        
        for p in range(1, n_pags_const + 1):
        # for v in range(1, 2):
            v = pags.index(pags_const[p-1]) + 1
    
            print('Curando constancias en votación n° ' + str(v) + ' de la fecha ' + fecha)
            ruta_voto = ruta_fecha + 'voto_' + str(v) + '/'
            df_voto = pd.read_json(ruta_voto + 'votos_final.json')
            
            imagen_votos = cv2.imread(ruta_voto + 'voto.jpg')
            
            img_height = imagen_votos.shape[0]
            plt.imshow(imagen_votos[math.floor(img_height*0.75): ,:])
            plt.axis('off')
            plt.show()
            
            while True:
                nombre = input('\tIngresar nombre congresista a curar constancia voto:\n')
                 
                while True:
                    voto_final = input('Ingresar nuevo voto de congresista ' + nombre + ':\n').upper()
                    
                    if voto_final in dict_check:
                        voto_final = dict_check[voto_final]
                    
                    if voto_final not in check_voto:
                        print('Voto no esta en lista check')
                    else:
                        break
                    
                df_voto.loc[df_voto.nombre == nombre, 'voto'] = voto_final
                
                while True:
                    check = input('\t¿Corregir más votos? y/n:\n')
                    if check not in ['y', 'n']:
                        pass
                    else:
                        break
                
                if check == 'y':
                    pass
                elif check == 'n':
                    break
                
            df_voto.to_json(ruta_voto + 'votos_final.json')
    else:
        print('No se encontró constancias a curar')
        
        
        
    
    
    
    