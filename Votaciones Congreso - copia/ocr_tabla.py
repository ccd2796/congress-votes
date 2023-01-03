# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 04:21:31 2022

@author: Asus
"""

def main(fecha):
    from sklearn.cluster import AgglomerativeClustering
    from pytesseract import Output
    import pandas as pd
    import pytesseract
    import cv2
    import numpy as np
    
    pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'
        
    
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
    
    # fecha = "20210819"
    
    args = read_args()
    pags = args['pags']
    
    ruta_fecha = 'Datos/Output/' + fecha + '/'    
    
    if type(pags) == int:
        n_pags = 1
    else:
        n_pags = len(pags)
    
    
    # set the PSM mode to detect sparse text, and then localize text in
    # the table
    options = "--psm 6"
    args["min-conf"] = 0
    
    #leyendo los votos
    for v in range(1, n_pags + 1):
    # for v in range(1, 2):    
        ruta_voto = ruta_fecha + 'voto_' + str(v) + '/'
        print('\tExtrayendo voto n° ' + str(v) + ' de ' + str(n_pags) + ' de la fecha ' + fecha)
        #leyendo las columnas
        # for c in range(1,4):
        voto_cong = {}
        voto_coord = {}
        v_c = 0
        for c in range(1,4):
            print('\t\tLeyendo columna ' + str(c))
            table_col = cv2.imread(ruta_voto + 'col'+ str(c) + '.jpg')
            print('\t\t\tAplicando OCR - ...')
            results = pytesseract.image_to_data(cv2.cvtColor(table_col, cv2.COLOR_BGR2RGB), lang="spa", config=options, output_type=Output.DICT)
            print('\t\t\tAplicando OCR - Completado')
            # initialize a list to store the (x, y)-coordinates of the detected
            # text along with the OCR'd text itself
            coords = []
            ocrText = []
            
            # loop over each of the individual text localizations
            for r in range(0, len(results["text"])):
            	# extract the bounding box coordinates of the text region from
            	# the current result
            	x = results["left"][r]
            	y = results["top"][r]
            	w = results["width"][r]
            	h = results["height"][r]
            	# extract the OCR text itself along with the confidence of the
            	# text localization
            	text = results["text"][r]
            	conf = float(results["conf"][r])
            	# filter out weak confidence text localizations
            	if conf > args["min-conf"]:
                # if conf > 50:
            		# update our text bounding box coordinates and OCR'd text,
            		# respectively
            		coords.append((x, y, w, h))
            		ocrText.append(text)
                
            #clasificando filas y columnas:
            print('\t\t\tClasificando filas y columnas')
            #delimitadores de columnas
            coord_col_1 = round(0.12*table_col.shape[1])
            coord_col_2 = round(0.85*table_col.shape[1])
                
            #filas
            # extract all x-coordinates from the text bounding boxes, setting the
            # y-coordinate value to zero
            yCoords = [(0, c[1]) for c in coords]
            # apply hierarchical agglomerative clustering to the coordinates
            clustering = AgglomerativeClustering(
            	n_clusters=None,
            	affinity="manhattan",
            	linkage="complete",
            	distance_threshold=args["dist-thresh"])
            clustering.fit(yCoords)
            # initialize our list of sorted clusters
            sortedClusters = []
            
            # loop over all clusters
            for l in np.unique(clustering.labels_):
            	# extract the indexes for the coordinates belonging to the
            	# current cluster
            	idxs = np.where(clustering.labels_ == l)[0]
            	# verify that the cluster is sufficiently large
            	if len(idxs) > args["min-size"]:
            		# compute the average x-coordinate value of the cluster and
            		# update our clusters list with the current label and the
            		# average x-coordinate
            		avg = np.average([coords[i][1] for i in idxs])
            		sortedClusters.append((l, avg))
            # sort the clusters by their average x-coordinate and initialize our
            # data frame
            sortedClusters.sort(key=lambda x: x[1])
            
            for (l, avg) in sortedClusters:
                idxs = np.where(clustering.labels_ == l)[0]
                xCoords = [coords[i][0] for i in idxs]
                sortedIdxs = idxs[np.argsort(xCoords)]
                
                row = {'partido':[], 'nombre':[], 'voto':[]}
                for i in sortedIdxs:
                        (x, y, w, h) = coords[i]
    
                        if x < coord_col_1:
                            row['partido'].append(ocrText[i].strip())
                        elif coord_col_1 <= x < coord_col_2:
                            row['nombre'].append(ocrText[i].strip())
                        elif coord_col_2 <= x:
                            row['voto'].append(ocrText[i].strip())
                            
                for r, data in row.items():
                    row[r] = ' '.join(data)
                    row[r] = row[r].strip()

                voto_cong[v_c] = {'partido': row['partido'], 'nombre': row['nombre'], 'voto': row['voto']}
                voto_coord[v_c] = {'yCoord': avg, 'col': c}
                v_c += 1
        
        df_voto = pd.DataFrame.from_dict(voto_cong, orient = 'index')
        df_voto.to_json(ruta_voto + 'votos_raw.json')
        
        df_coord = pd.DataFrame.from_dict(voto_coord, orient = 'index')
        df_coord.to_json(ruta_voto + 'votos_coord.json')
        

