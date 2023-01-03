# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 06:38:56 2022

@author: Asus
"""

def main(fecha):
    import os
    from pdf2image import convert_from_path
    
    #Script para crear carpetas. Input fecha y numero de paginas que tienen votos
    def rutas_imgs():
        # Directory
        parent_dir = r"D:\Python\Scripts\Ushnu\Votaciones Congreso\Datos\Output"
        # fecha = "20210805"
        if type(pags) == int:
            votos = 1
        else:
            votos = len(pags)
        path_fecha = os.path.join(parent_dir, fecha)
        
        try:
            os.mkdir(path_fecha)
        except OSError:
            print("Carpeta " + fecha + ' ya existe')
        finally:  
            for voto in range(1, votos+1):
                path_voto = os.path.join(parent_dir, fecha, "voto_" + str(voto))
                try:
                    os.mkdir(path_voto)
                except OSError:
                    print("Carpeta " + fecha + "/" + str(voto)+ ' ya existe')
                finally:
                    if type(pags) == int:
                        n_pag = pags
                    else:
                        n_pag = pags[voto-1]
                    print("\tGuardando pág " + str(n_pag) + ' como JPEG')
                    images[n_pag-1].save(path_voto + '/voto.jpg', 'JPEG')
                    
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
        
    
    # fecha = "20210819"
    
    print("Trabajando PDF con fecha " + fecha)
    
    args = read_args()
    
    pags = args['pags']
    
    ruta_pdf = 'Datos/Input/' + fecha + '.pdf'
    print("Convirtiendo PDF: " + ruta_pdf + ' a JPG - ...')
    images = convert_from_path(ruta_pdf, poppler_path=r'D:\Python\Scripts\Ushnu\Votaciones Congreso\poppler-0.68.0_x86\poppler-0.68.0\bin')
    print("Convirtiendo PDF: " + ruta_pdf + ' a JPG - Completado')
    
    rutas_imgs()

