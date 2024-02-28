# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 19:54:26 2023

@author: ulas0
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, you can also specify a video file

model= load_model("model3.h5")

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Preprocess the frame to match the model's input requirements
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    frame = cv2.resize(frame, (64, 64))  # Resize to match model input size
    frame = frame/255.0 # Normalize pixel values to [0, 1]
    frame2 = np.expand_dims(frame, axis=0)  # Add batch dimension
    
    # Predict smile using the model
    prediction = model.predict(frame2)
    
    #stay close to camera
    # Determine the prediction result
    if prediction[0][0] > 0.5:
        label = "Smile"
        color = (0, 255, 0)  # Green
    else:
        label = "No Smile"
        color = (0, 0, 255)  # Red
    
    # Overlay the result on the frame
    cv2.putText(frame, label, (2,15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, thickness=1)
    
    # Display the frame with the result
    cv2.imshow('Frame',cv2.resize(frame, (190,140)))
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()


















