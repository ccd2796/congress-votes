# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 12:36:51 2022

@author: Asus
"""

def main(fecha):

    import numpy as np
    import pytesseract
    import imutils
    import cv2
    import math
    
    pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'
    
    #Script para procesar imagenes
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
    
    print("Trabajando imágenes de votaciones con fecha " + fecha)
    
    args = read_args()
    pags = args['pags']
    
    if type(pags) == int:
        n_pags = 1
    else:
        n_pags = len(pags)
    
    ruta_fecha = 'Datos/Output/' + fecha + '/'    
    for v in range(1, n_pags + 1):
        print('\tExtrayendo resultados de votación n° ' + str(v) )
        ruta_voto = ruta_fecha + 'voto_' + str(v) + '/'
        
    
        #Codigo para borrar el titulo y el contorno del sello
        #Se tienen que borrar porque interfieren al ubicar la tabla
        
        # set a seed for our random number generator
        np.random.seed(42)
        # load the input image and convert it to grayscale
        image = cv2.imread(ruta_voto + 'voto.jpg')
        shp = image.shape
        #recortando sello a mano, crop 80% del alto
        image = image[:round(shp[0]*0.8),:,:]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # initialize a rectangular kernel that is ~5x wider than it is tall,
        # then smooth the image using a 3x3 Gaussian blur and then apply a
        # blackhat morphological operator to find dark regions on a light
        # background
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 11))
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        # compute the Scharr gradient of the blackhat image and scale the
        # result into the range [0, 255]
        grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad = np.absolute(grad)
        (minVal, maxVal) = (np.min(grad), np.max(grad))
        grad = (grad - minVal) / (maxVal - minVal)
        grad = (grad * 255).astype("uint8")
        # apply a closing operation using the rectangular kernel to close
        # gaps in between characters, apply Otsu's thresholding method, and
        # finally a dilation operation to enlarge foreground regions
        grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        # cv2.imshow("Thresh", ResizeWithAspectRatio(thresh, width=400))
        
        # find contours in the thresholded image and grab the largest one,
        # which we will assume is the stats table
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts_area = {cv2.contourArea(cnt):cnt for cnt in cnts}
        areas_sorted = sorted(cnts_area.keys())
        tableCnt = cnts_area[areas_sorted[-1]]
        
        titulo_area = areas_sorted[-1]
        tituloCnt = cnts_area[titulo_area]
        
        (x_titulo, y_titulo, w_titulo, h_titulo) = cv2.boundingRect(tituloCnt)
        title = image[y_titulo:y_titulo + h_titulo, x_titulo:x_titulo + w_titulo]
        
        
        #De nuevo sin el titulo
        image = image[y_titulo + h_titulo :, :]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # initialize a rectangular kernel that is ~5x wider than it is tall,
        # then smooth the image using a 3x3 Gaussian blur and then apply a
        # blackhat morphological operator to find dark regions on a light
        # background
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 11))
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        # compute the Scharr gradient of the blackhat image and scale the
        # result into the range [0, 255]
        grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad = np.absolute(grad)
        (minVal, maxVal) = (np.min(grad), np.max(grad))
        grad = (grad - minVal) / (maxVal - minVal)
        grad = (grad * 255).astype("uint8")
        # apply a closing operation using the rectangular kernel to close
        # gaps in between characters, apply Otsu's thresholding method, and
        # finally a dilation operation to enlarge foreground regions
        grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        thresh = cv2.dilate(thresh, None, iterations=10)
        
        # find contours in the thresholded image and grab the largest one,
        # which we will assume is the stats table
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts_area = {cv2.contourArea(cnt):cnt for cnt in cnts}
        areas_sorted = sorted(cnts_area.keys())
        tableCnt = cnts_area[areas_sorted[-1]]
        
        (x, y, w, h) = cv2.boundingRect(tableCnt)
        table = image[y:y + h, x:x + w]
        
        #partiendo la tabla en 3
        table_width = table.shape[1]
        cutoff = math.floor(table_width/3)
        
        for col in range(1, 4):
            table_col = table[:, cutoff*(col-1):cutoff*col]
            cv2.imwrite(ruta_voto + 'col' +  str(col) +'.jpg', table_col)
        
        
        # show the original input image and extracted table to our screen
        # cv2.imshow("Input", ResizeWithAspectRatio(image, width=400))
        # cv2.imshow("Table", ResizeWithAspectRatio(table, width=400))
        # cv2.imshow("Title", ResizeWithAspectRatio(title, width=600))
        
        cv2.imwrite(ruta_voto + 'tabla.jpg', table)
        cv2.imwrite(ruta_voto + 'titulo.jpg', title)
        
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
