# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 15:29:31 2022

@author: Kmedrano101
"""

import serial
import json
import predictor as md
import cv2 as cv


arduino = serial.Serial(port='COM3') # open serial port

#print(arduino.name) # check which port was really used
modelo = md.predictorModel()
modelo.load_model()
json_tx = json.dumps({'C_P1': 0,'T_P1': 1})

# Start Video
cap = cv.VideoCapture(1)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
image = None

while True:
    _, img = cap.read()
    if arduino.in_waiting>0:
        cadena = arduino.readline()
    try:
        json_cad = json.loads(cadena)
        print(json_cad)
    except:
        pass
    if json_cad['S_P1'] == 1 and json_cad['S_P2'] == 1:
        image = img
        print(modelo.make_prediction(img))
    if image is not None:
        cv.imshow('Windows Img',image)
        if cv.waitKey(1) == ord('q'):
            break
    #arduino.write(json_tx.encode())
cap.release()
cv.destroyAllWindows()
arduino.close()   